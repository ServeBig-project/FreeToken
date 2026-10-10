"""Standalone public startup contract for an installed, outdated CPU extension.

The coordinator prepares NOWAG_STALE_CPU_SOURCE as a separate candidate installation
containing the old extension; this test never replaces a library or changes that directory.
NOWAG_STALE_CPU_PYTHON selects its interpreter. A valid public config/index with absent
weight shards must still report the extension-rebuild error first (confirmed 2026-10-10).
Only CLI process behavior is observed; no production modules are imported by the test.
"""

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

CLI = "import sys; from freetoken.cli import main; sys.argv = ['ft', *sys.argv[1:]]; sys.exit(main())"
MODES = [("cpu", ["--moe-backend", "cpu"]),
         ("hybrid", ["--moe-backend", "hybrid", "--moe-cache-size", "1536"]),
         ("cpu_layers", ["--moe-backend", "offload", "--moe-cpu-layers", "8",
                         "--moe-cache-size", "1536"])]


@pytest.mark.parametrize("name,mode", MODES)
def test_stale_cpu_extension_rejected_before_weight_loading(name, mode, tmp_path):
    if os.environ.get("NOWAG_GPU_OK") != "1" or not os.environ.get("NOWAG_GPU"):
        pytest.skip("GPU use not approved for the startup subprocess")
    source, python = (os.environ.get(key) for key in ("NOWAG_STALE_CPU_SOURCE", "NOWAG_STALE_CPU_PYTHON"))
    if not source or not python:
        pytest.skip("separate old-extension source/interpreter environment not provided")
    assert Path(source).is_dir() and Path(python).is_file()
    original = Path(os.environ.get("NOWAG_QWEN36_BASE", "/data1/lmcache_kv/models/Qwen3.6-35B-A3B-NVFP4"))
    base = tmp_path / "metadata-only-base"
    base.mkdir()
    metadata = ["config.json", "generation_config.json", "model.safetensors.index.json",
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "vocab.json", "merges.txt", "added_tokens.json"]
    for filename in metadata:
        if (original / filename).is_file():
            shutil.copy2(original / filename, base / filename)
    assert (base / "config.json").is_file()
    index = json.loads((base / "model.safetensors.index.json").read_text())
    assert index["weight_map"] and all(not (base / shard).exists() for shard in index["weight_map"].values())
    command = [python, "-c", CLI, "serve", "--model", str(base), "--gpu", os.environ["NOWAG_GPU"],
               "--batching-policy", "legacy", "--port", os.environ.get("NOWAG_PORT", "31791"), *mode]
    process = subprocess.Popen(command, env={**os.environ, "PYTHONPATH": source},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               start_new_session=True)
    try:
        output, _ = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"{name}: old extension did not reject startup before loading weights\n{output[-4000:]}")
    logs = Path(os.environ.get("NOWAG_LOG_DIR", str(tmp_path)))
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"stale_cpu_extension_{name}.log").write_text(output)
    lowered = output.lower()
    assert process.returncode != 0, output[-4000:]
    assert "cpu" in lowered and "rebuild" in lowered and "extension" in lowered, output[-4000:]
    assert "filenotfounderror" not in lowered, "missing weight shards were accessed before the extension check"

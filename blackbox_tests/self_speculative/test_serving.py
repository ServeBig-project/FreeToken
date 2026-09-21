"""Source-blind acceptance tests: only ft serve, HTTP, and checkpoint inputs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, suppress
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest


PROJECT = Path("/home/nengneng/AIPrometheus/servebig/servebig-project")
CANDIDATE = Path(os.environ.get("FT_SD_CANDIDATE", PROJECT / ".sd-worktrees/freetoken-s2-sd/python"))
BASELINE = Path(os.environ.get("FT_SD_BASELINE", PROJECT / "FreeToken/python"))
MODEL = Path(os.environ.get("FT_SD_MODEL", "/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B"))
GPU = os.environ.get("FT_SD_GPU")
CONTEXT = 256
WIDTH = 4
SAMPLES = 128
LONG = "Continue the list of integers without explanation: 1, 2, 3, 4, 5,"
SAMPLED = {"prompt": "Continue this sequence using only A or B separated by spaces:\n"
                     "A B B A A B A B B B A A B A B B A A B A", "temperature": 1.5,
           "top_k": 2, "top_p": 1.0, "max_tokens": 8, "ignore_eos": True}
MODES = [("disabled", 0, 3, "radix", "offload"),
         ("reduced", 4, 3, "radix", "offload"),
         ("equal", 4, "target", "radix", "offload"),
         ("single", 1, 1, "naive", "auto")]


def cli(package, *args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package)
    return [sys.executable, "-c", "from freetoken.cli import main; main()", "serve", *args], env


def test_cli_help():
    """Missing advertised flags blocks use of the feature before any GPU run."""
    command, env = cli(CANDIDATE, "--help")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--speculative-num-steps" in result.stdout
    assert "--speculative-draft-experts" in result.stdout


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    path = Path(os.environ.get("FT_SD_ARTIFACTS", tmp_path_factory.mktemp("self-speculative")))
    path.mkdir(parents=True, exist_ok=True)
    print(f"\nBlackbox evidence: {path}")
    return path


@pytest.fixture(scope="session")
def checkpoint():
    if not GPU:
        pytest.skip("GPU not assigned; set FT_SD_GPU only after coordinator assignment")
    from tokenizers import Tokenizer

    config = json.loads((MODEL / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    assert config["model_type"] == "qwen3_moe", "Acceptance requires the real Qwen3MoE checkpoint"
    return config, tokenizer


def common_args(port):
    return ["--model-path", str(MODEL), "--gpu", GPU, "--host", "127.0.0.1", "--port", str(port),
            "--batching-policy", "legacy", "--max-running-requests", str(WIDTH),
            "--max-seq-len-override", str(CONTEXT), "--max-prefill-length", "128",
            "--num-tokens", "1024", "--moe-cache-size", "512", "--num-tokenizer", "0",
            "--cuda-graph-max-bs", "4", "--sampling-defaults", "none", "--reasoning-parser", "off",
            "--served-model-name", "sd-blackbox", "--enable-cache-report"]


class Server:
    def __init__(self, name, client):
        self.name, self.client = name, client
        self.requests, self.metrics, self.snapshots = [], [], []

    def stats(self):
        response = self.client.get("/v1/stats")
        assert response.status_code == 200, response.text
        stats = response.json()
        self.snapshots.append(stats)
        return stats

    def idle(self):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            stats = self.stats()
            if stats["requests"]["active"] == 0:
                return stats
            time.sleep(0.2)
        pytest.fail(f"{self.name}: request resources remain active after 60 seconds: {stats}")

    def call(self, request, stream=False):
        body = {"model": "sd-blackbox", **request, "stream": stream}
        path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
        started = time.monotonic()
        if stream:
            body["stream_options"] = {"include_usage": True}
            text, endings, usage, done = [], [], None, False
            with self.client.stream("POST", path, json=body) as response:
                assert response.status_code == 200, response.read().decode()
                for line in response.iter_lines():
                    if not line or line.startswith(":"):
                        continue
                    assert line.startswith("data: "), line
                    if line == "data: [DONE]":
                        done = True
                        break
                    chunk = json.loads(line[6:])
                    if not chunk["choices"]:
                        assert usage is None, "Duplicate streamed usage"
                        usage = chunk["usage"]
                        continue
                    assert len(chunk["choices"]) == 1, chunk
                    choice = chunk["choices"][0]
                    piece = choice.get("text", choice.get("delta", {}).get("content")) or ""
                    assert not (endings and piece), "Text emitted after termination"
                    if choice.get("finish_reason") is not None:
                        assert not piece, "Terminal chunk must contain no output text"
                        endings.append(choice["finish_reason"])
                    text.append(piece)
            assert done and len(endings) == 1 and usage is not None, (done, endings, usage)
            result = {"text": "".join(text), "finish_reason": endings[0], "usage": usage}
        else:
            response = self.client.post(path, json=body)
            assert response.status_code == 200, f"{self.name}: {body!r}: {response.text}"
            data = response.json()
            assert len(data["choices"]) == 1, data
            choice = data["choices"][0]
            result = {"text": choice.get("text", choice.get("message", {}).get("content")) or "",
                      "finish_reason": choice["finish_reason"], "usage": data["usage"]}
        result["usage"] = {key: result["usage"][key] for key in
                           ("prompt_tokens", "completion_tokens", "total_tokens")}
        usage = result["usage"]
        assert all(isinstance(value, int) and value >= 0 for value in usage.values()), usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], usage
        limit = body.get("max_completion_tokens", body["max_tokens"])
        assert usage["completion_tokens"] <= limit, "Discarded drafts escaped the output limit"
        assert result["finish_reason"] in {"stop", "length"}, result
        elapsed = time.monotonic() - started
        self.requests.append({"request": body, "response": result, "seconds": elapsed})
        return result

    def batch(self, requests, label):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            results = list(pool.map(self.call, requests))
        elapsed = time.monotonic() - started
        tokens = sum(result["usage"]["completion_tokens"] for result in results)
        self.metrics.append({"label": label, "requests": len(requests), "seconds": elapsed,
                             "committed_tokens": tokens, "committed_tokens_per_second": tokens / elapsed})
        return results


@contextmanager
def serve(name, package, artifacts, steps=0, experts=3, cache="radix", backend="offload", extra_args=()):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    args = common_args(port) + ["--cache-type", cache, "--moe-backend", backend, *extra_args]
    if steps:
        args += ["--speculative-num-steps", str(steps), "--speculative-draft-experts", str(experts)]
    command, env = cli(package, *args)
    log_path = artifacts / f"{name}.log"
    with log_path.open("w") as log, httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=600) as client:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        server = Server(name, client)
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                assert process.poll() is None, f"{name} exited during startup; public process log: {log_path}"
                try:
                    health = client.get("/health", timeout=2).json()
                    if health["status"] == "error":
                        pytest.fail(f"{name} startup failed: {health['message']}")
                    if health["status"] == "ok":
                        models = client.get("/v1/models", timeout=2)
                        assert models.status_code == 200, models.text
                        server.models = models.json()
                        break
                except httpx.TransportError:
                    pass
                time.sleep(1)
            else:
                pytest.fail(f"{name} failed to start within 900 seconds; see {log_path}")
            server.stats()
            yield server
        finally:
            (artifacts / f"{name}.json").write_text(json.dumps(
                {"command": command, "model": str(MODEL), "requests": server.requests,
                 "metrics": server.metrics, "stats": server.snapshots}, ensure_ascii=False, indent=2))
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def long_prompt(tokenizer, label, count):
    ids = tokenizer.encode(label + " alpha" * count, add_special_tokens=False).ids[:count]
    prompt = tokenizer.decode(ids, skip_special_tokens=False)
    assert len(tokenizer.encode(prompt, add_special_tokens=False).ids) == count
    return prompt


def test_long_context_reuse(checkpoint, artifacts):
    """Real document prefixes must preserve target outputs across concurrent reuse."""
    tokenizer = checkpoint[1]
    source = tokenizer.encode((MODEL / "README.md").read_text(), add_special_tokens=False).ids
    assert len(source) >= 3072, "Long-context input requires the complete public checkpoint README"
    requests = []
    for count, task in [(2048, "Summarize the total and activated expert counts in one short sentence."),
                        (3072, "Summarize the native and extended context lengths in one short sentence.")]:
        excerpt = tokenizer.decode(source[:count], skip_special_tokens=False)
        prompt = f"Read this excerpt from the model's public README:\n<readme>\n{excerpt}\n</readme>\n{task}\nAnswer:"
        requests.append({"prompt": prompt, "temperature": 0, "max_tokens": 16})
    limits = ["--max-seq-len-override", "4096", "--num-tokens", "8192", "--max-prefill-length", "2048"]
    with serve("long-baseline", BASELINE, artifacts, extra_args=limits) as ordinary:
        expected = ordinary.batch(requests, "long-context-first")
        assert all(result["usage"]["prompt_tokens"] >= count for result, count in zip(expected, (2048, 3072)))
        assert ordinary.batch(requests, "long-context-reuse") == expected
        ordinary.idle()
    with serve("long-reduced", CANDIDATE, artifacts, steps=4, experts=3, extra_args=limits) as enabled:
        before = enabled.idle()["speculative"]
        for label in ("long-context-first", "long-context-reuse"):
            assert enabled.batch(requests, label) == expected, "Long prefixes or cached history changed target output"
        after = enabled.idle()["speculative"]
        assert after["enabled"] and after["verify_steps"] > before["verify_steps"], "Long-context SD coverage missing"


@pytest.fixture(scope="session")
def baseline(checkpoint, artifacts):
    _, tokenizer = checkpoint
    cases = [{"prompt": LONG, "temperature": 0, "max_tokens": limit, "ignore_eos": True}
             for limit in (1, 2, 3, 4, 5, 7, 9, 17)]
    cases += [{"prompt": "把这句话翻译成英语：清晨的花园很安静。\n英语：", "temperature": 0, "max_tokens": 16},
              {"messages": [{"role": "user", "content": "Reply only with hello. /no_think"}],
               "temperature": 0, "max_tokens": 32},
              {"prompt": LONG, "temperature": 0, "max_tokens": 2, "max_completion_tokens": 7,
               "ignore_eos": True},
              {"prompt": long_prompt(tokenizer, "Near context limit:", CONTEXT - 2),
               "temperature": 0, "max_tokens": 2, "ignore_eos": True},
              {"prompt": long_prompt(tokenizer, "Near context limit:", CONTEXT - 2),
               "temperature": 0, "max_tokens": 7, "ignore_eos": True}]
    pressure = [{"prompt": long_prompt(tokenizer, f"Independent request {i}: ", CONTEXT - 10),
                 "temperature": 0, "max_tokens": 8, "ignore_eos": True} for i in range(WIDTH + 1)]
    eos = {"prompt": "<|im_start|>user\nReply only with OK. /no_think<|im_end|>\n"
                     "<|im_start|>assistant\n<think>\n\n</think>\n\n",
           "temperature": 0, "max_tokens": 64}
    evidence = os.environ.get("FT_SD_BASELINE_EVIDENCE")
    recorded = json.loads(Path(evidence).read_text())["requests"] if evidence else None

    def replay(request):
        body = {"model": "sd-blackbox", **request, "stream": False}
        for index, row in enumerate(recorded):
            if row["request"] == body:
                return recorded.pop(index)["response"]
        raise AssertionError(f"Recorded baseline has no remaining response for {request!r}")

    with (nullcontext() if evidence else serve("baseline", BASELINE, artifacts)) as server:
        call = replay if evidence else server.call
        outputs = [call(case) for case in cases]
        assert outputs[-1]["usage"]["completion_tokens"] == 2, "Baseline must clamp generation to context"
        eos_output = call(eos)
        assert eos_output["finish_reason"] == "stop", "EOS coverage missing: baseline did not emit EOS"
        text = outputs[7]["text"]
        assert len(text) >= 10, "Baseline sequence prompt must supply a real multi-character stop"
        stop = text[3:9]
        stopped = [{**cases[7], "stop": stop}, {**cases[7], "stop": [stop, "unseen stop marker"]}]
        stop_outputs = [call(case) for case in stopped]
        assert all(result["finish_reason"] == "stop" and stop not in result["text"] for result in stop_outputs)
        if evidence:
            pressure_outputs = [call(request) for request in pressure]
            samples = [call(SAMPLED) for _ in range(SAMPLES)]
        else:
            pressure_outputs = server.batch(pressure, "cache-capacity")
            samples = []
            for _ in range(SAMPLES // WIDTH):
                samples.extend(server.batch([SAMPLED] * WIDTH, "sampling"))
            server.idle()
    return {"cases": cases, "outputs": outputs, "eos": eos, "eos_output": eos_output,
            "stopped": stopped, "stop_outputs": stop_outputs, "pressure": pressure,
            "pressure_outputs": pressure_outputs, "samples": samples}


@pytest.fixture(scope="module", params=MODES, ids=[mode[0] for mode in MODES])
def server(request, checkpoint, baseline, artifacts):
    name, steps, experts, cache, backend = request.param
    if experts == "target":
        experts = checkpoint[0]["num_experts_per_tok"]
    with serve(name, CANDIDATE, artifacts, steps, experts, cache, backend) as instance:
        instance.steps, instance.experts = steps, experts
        yield instance


def test_greedy_boundaries(server, baseline):
    """A mismatch blocks delivery: drafts, limits, or routing changed target decoding."""
    for request, expected in zip(baseline["cases"], baseline["outputs"]):
        assert server.call(request) == expected, request


def test_eos_and_stops(server, baseline):
    """EOS and both public stop forms must hide discarded suffixes and usage."""
    requests = [baseline["eos"], *baseline["stopped"]]
    expected = [baseline["eos_output"], *baseline["stop_outputs"]]
    for request, result in zip(requests, expected):
        assert server.call(request) == result, request
        assert server.call(request, stream=True) == result, request


def test_streaming_completion_and_chat(server, baseline):
    """A stream must end once, include usage, and expose exactly committed text."""
    for index in (0, 3, 6, 9, 11, 12):
        assert server.call(baseline["cases"][index], stream=True) == baseline["outputs"][index]


def test_context_overflow_error(server, checkpoint):
    """A full prompt must return the public error, without retaining a request slot."""
    for count in (CONTEXT, CONTEXT + 1):
        body = {"model": "sd-blackbox", "prompt": long_prompt(checkpoint[1], "Full context: ", count),
                "max_tokens": 1, "temperature": 0}
        response = server.client.post("/v1/completions", json=body)
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error", error
        assert str(count) in error["message"] and str(CONTEXT) in error["message"], error
    server.idle()


def test_heterogeneous_concurrency(server, baseline):
    """Different finish times, prompts, and sampling settings must stay isolated."""
    indices = (0, 7, 8, 9)
    requests = [baseline["cases"][index] for index in indices]
    requests += [{**baseline["cases"][6], "temperature": 0.8, "top_k": 1, "top_p": 0.9}, SAMPLED]
    expected_groups = [[baseline["outputs"][index] for index in indices] + [baseline["outputs"][6]]]
    reference_path = os.environ.get(f"FT_SD_CONCURRENT_REFERENCE_{server.name.upper()}")
    if reference_path:
        reference = json.loads(Path(reference_path).read_text())
        assert reference["speculative"]["enabled"] is False, "Concurrency reference must use ordinary inference"
        assert [row["request"] for row in reference["mixed"]] == [
            {"model": "sd-blackbox", **request, "stream": False} for request in requests]
        expected_groups = [[row[key] for row in reference["mixed"][:5]]
                           for key in ("ordinary_reference", "ordinary_eager")]
        server.metrics.append({"label": "concurrent-reference", "path": reference_path})
    results = server.batch(requests, "heterogeneous-concurrency")
    assert results[:5] in expected_groups, "Deterministic responses must match one complete ordinary group"
    assert results[5]["usage"]["completion_tokens"] == SAMPLED["max_tokens"]


def test_sampling_filters(server, baseline, checkpoint):
    """Single-token support gives a deterministic oracle for top-k and top-p."""
    greedy = baseline["cases"][6]
    assert server.call({**greedy, "temperature": 1.4, "top_k": 1, "top_p": 1}) == baseline["outputs"][6]
    # The most likely token has mass >= 1/vocabulary_size, so this retains one token.
    tiny_p = 0.5 / checkpoint[0]["vocab_size"]
    assert server.call({**greedy, "temperature": 0.9, "top_p": tiny_p}) == baseline["outputs"][6]


def permutation_p(left, right):
    categories = {value: index for index, value in enumerate(sorted(set(left + right)))}
    pooled = [categories[value] for value in left + right]
    n = len(left)

    def distance(values):
        counts = [0] * len(categories)
        for index, value in enumerate(values):
            counts[value] += 1 if index < n else -1
        return sum(abs(count) for count in counts)

    observed, extreme = distance(pooled), 1
    rng = random.Random(1731)
    for _ in range(4999):
        rng.shuffle(pooled)
        extreme += distance(pooled) >= observed
    return extreme / 5000


def sample_bucket(result):
    text = result["text"]
    if not re.fullmatch(r"\s*[AB](?:\s+[AB])*\s*", text):
        return "other"
    return f"A={text.split().count('A')}"


def test_stochastic_distribution(server, baseline):
    """Detect material sampling bias, without assuming seed or sampled-text identity."""
    before = server.idle()["speculative"]
    samples = []
    for _ in range(SAMPLES // WIDTH):
        samples.extend(server.batch([SAMPLED] * WIDTH, "sampling"))
    after = server.idle()["speculative"]
    if server.steps:
        for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps"):
            assert after[key] > before[key], f"Sampling coverage missing: no speculative {key}"
    left = [sample_bucket(result) for result in baseline["samples"]]
    right = [sample_bucket(result) for result in samples]
    assert len(set(left) - {"other"}) > 1, "Sampling coverage missing: baseline A-count projection had no variation"
    p = permutation_p(left, right)
    server.metrics.append({"label": "distribution", "samples_per_mode": SAMPLES,
                           "baseline_histogram": dict(Counter(left)), "candidate_histogram": dict(Counter(right)),
                           "permutation_p": p})
    assert p >= 0.001, f"Observed target sampling distribution differs (permutation p={p}); inspect recorded outputs"
    streamed = server.call(SAMPLED, stream=True)
    assert streamed["usage"]["completion_tokens"] == SAMPLED["max_tokens"]


def test_cache_capacity_and_history(server, baseline):
    """Repeated admission at fixed capacity exposes leaks and stale draft history."""
    for _ in range(2):
        assert server.batch(baseline["pressure"], "cache-capacity") == baseline["pressure_outputs"]
        stats = server.idle()
        if stats["kv"] is not None:
            assert 0 <= stats["kv"]["used_pages"] <= stats["kv"]["total_pages"], stats
    assert server.call(baseline["cases"][7]) == baseline["outputs"][7]


def test_cancellation_and_recovery(server, baseline):
    """Aborted streams must release admission slots and preserve other requests."""
    body = {"model": "sd-blackbox", "prompt": LONG, "temperature": 0,
            "max_tokens": CONTEXT - 32, "ignore_eos": True, "stream": True}
    for _ in range(3):
        before, prefix = server.idle(), []
        with ThreadPoolExecutor(max_workers=1) as pool:
            with server.client.stream("POST", "/v1/completions", json=body) as response:
                assert response.status_code == 200, response.read().decode()
                for line in response.iter_lines():
                    if not line.startswith("data: {"):
                        continue
                    chunk = json.loads(line[6:])
                    prefix.extend(choice.get("text", "") for choice in chunk["choices"])
                    if not any(choice.get("text") for choice in chunk["choices"]):
                        continue
                    during = server.stats()
                    verified = during["speculative"]["verify_steps"] > before["speculative"]["verify_steps"]
                    if during["requests"]["active"] == 1 and (not server.steps or verified):
                        survivor = pool.submit(server.call, baseline["cases"][7])
                        break
                else:
                    pytest.fail("Cancellation coverage missing: no active streamed request with completed verification")
            assert survivor.result() == baseline["outputs"][7]
        server.requests.append({"request": body, "cancelled": True, "received_prefix": "".join(prefix)})
        after = server.idle()
        for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps"):
            assert after["speculative"][key] >= during["speculative"][key], "Cancelled work disappeared from counters"
        assert server.batch(baseline["pressure"], "post-cancellation-capacity") == baseline["pressure_outputs"]
    with server.client.stream("POST", "/v1/completions", json=body) as response:
        assert response.status_code == 200, response.read().decode()
    server.idle()
    assert server.call(baseline["cases"][7]) == baseline["outputs"][7]


def test_speculation_counters_and_rejection_path(server):
    """Exercise retained and rejected drafts; missing coverage cannot claim acceptance."""
    before = server.idle()["speculative"]
    prompts = ["Continue a detailed story: The lighthouse keeper opened a sealed letter and found",
               "Explain in detail why the sky is blue and how scattering varies with wavelength:",
               "Write Python code for merging sorted lists with a docstring:\ndef merge(left, right):"]
    for prompt in prompts:
        result = server.call({"prompt": prompt, "temperature": 0, "max_tokens": 64, "ignore_eos": True})
        assert result["finish_reason"] == "length" and result["usage"]["completion_tokens"] == 64
    after = server.idle()["speculative"]
    assert after["enabled"] is bool(server.steps), after
    for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps"):
        assert isinstance(after[key], int) and after[key] >= before[key] >= 0, after
    assert after["accepted_draft_tokens"] <= after["draft_tokens"], after
    if not server.steps:
        assert all(after[key] == 0 for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps")), after
        return
    delta = {key: after[key] - before[key] for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps")}
    assert delta["verify_steps"] > 0 and delta["accepted_draft_tokens"] > 0, delta
    server.metrics.append({"label": "speculation", **delta,
                           "draft_acceptance": delta["accepted_draft_tokens"] / delta["draft_tokens"]})
    if server.name != "equal":
        discarded = delta["draft_tokens"] - delta["accepted_draft_tokens"]
        assert discarded > server.steps * len(prompts), (
            f"Rejection coverage missing: {delta}; discards could all be final-boundary clipping")


@pytest.mark.parametrize("extra,reason", [
    (["--speculative-num-steps", "-1"], "nonnegative|non-negative|>=.?0|negative"),
    (["--speculative-num-steps", "1.5"], "integer|int"),
    (["--speculative-draft-experts", "0"], "expert|>=.?1|positive"),
    (["--speculative-draft-experts", "too-many"], "expert|range|between|token"),
    (["--speculative-draft-experts", "1.5"], "integer|int"),
    *[(["--batching-policy", policy], "legacy|policy|support")
      for policy in ("mixed", "layered", "joint", "layered-pipeline")],
    (["--tensor-parallel-size", "2"], "single|one|tensor|gpu"),
    (["--moe-backend", "cpu"], "cpu|backend"),
    (["--moe-backend", "hybrid"], "hybrid|backend"),
    (["--moe-cpu-layers", "1"], "cpu|layer"),
    (["--model-path", "unsupported"], "qwen|architecture|model"),
])
def test_enabled_startup_errors(checkpoint, artifacts, extra, reason):
    """Unsupported combinations must reject clearly, not hang or start serving."""
    extra = list(extra)
    if extra[-1] == "too-many":
        extra[-1] = str(checkpoint[0]["num_experts_per_tok"] + 1)
    if extra[-1] == "unsupported":
        path = os.environ.get("FT_SD_UNSUPPORTED_MODEL", "/data1/lmcache_kv/models/Qwen3.6-35B-A3B-NVFP4")
        if not Path(path).is_dir():
            pytest.skip("Unsupported-architecture real checkpoint is unavailable")
        extra[-1] = path
    command, env = cli(CANDIDATE, *common_args(18997), "--speculative-num-steps", "4", *extra)
    label = "error-" + re.sub(r"[^a-zA-Z0-9_-]", "_", "-".join(extra))
    with (artifacts / f"{label[-180:]}.log").open("w+") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            pytest.fail(f"Unsupported startup combination did not reject within 90 seconds: {extra}")
        log.seek(0)
        output = log.read()
    assert result != 0, f"Unsupported startup combination was accepted: {extra}"
    assert "speculat" in output.lower() and re.search(reason, output, re.I), output

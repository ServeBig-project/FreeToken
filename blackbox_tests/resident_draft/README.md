# Resident-only draft: independent HTTP acceptance

These tests use the public CLI, HTTP responses and cumulative statistics only.
The author has not read production source, implementation diffs or other tests.
The coordinator starts servers and assigns GPUs; the HTTP runner never starts one.
Run against a dedicated service with no unrelated requests.

## Inputs and server command

The real BF16 Qwen3-30B-A3B checkpoint has 48 layers and 128 experts per layer.
Write an input list with six fixed experts per layer, leaving more candidates than
the draft's K=3. The expert list is a test input, not a residency policy requirement:

```sh
export TEST_PYTHON=/home/nengneng/miniconda3/envs/freetoken-dev/bin/python
"$TEST_PYTHON" -c 'import json; from pathlib import Path; Path("/tmp/resident-draft-six.json").write_text(json.dumps({"gpu_experts": [[layer, expert] for layer in range(48) for expert in range(6)]}))'
```

Set `TEST_GPU` to the coordinator-assigned free GPU. For the baseline, omit the
new residency flag so its default is exercised:

```sh
PYTHONPATH=python "$TEST_PYTHON" -c 'from freetoken.cli import main; main()' serve \
  --model-path /data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B \
  --gpu "$TEST_GPU" --host 127.0.0.1 --port 30000 \
  --moe-backend offload --moe-cache-size 512 --moe-collect-stats \
  --disable-moe-prefill-overlap --moe-resident-experts /tmp/resident-draft-six.json \
  --batching-policy legacy --max-running-requests 4 --cache-type radix \
  --max-seq-len-override 4096 --num-tokens 8192 --max-prefill-length 2048 \
  --sampling-defaults none --reasoning-parser off \
  --speculative-num-steps 3 --speculative-draft-experts 3
```

The active configurations add `--speculative-draft-residency router` or
`--speculative-draft-residency affinity`. The shortage configurations also remove
`--moe-resident-experts` and change `--moe-cache-size` to `128`. With 48×3 > 128,
at least one layer must have fewer than K cached experts; the single-layer prefill
area still fits because overlap is disabled. Restart only between configurations.

## Matrix and HTTP commands

The baseline must actually draft, verify and load draft experts. Every active
mode must draft and verify with zero draft loads and zero residency stops;
affinity must perform replacements. Shortage must increase residency stops while
draft and verification counters stay unchanged. Failure changes the acceptance
result; missing draft/replacement/load coverage is a failure, not a passing zero.

```sh
"$TEST_PYTHON" blackbox_tests/resident_draft/check_http.py \
  --url http://127.0.0.1:30000 --scenario baseline --mode off \
  --reference /tmp/resident-draft-reference.json > /tmp/resident-draft-baseline.json

# After restarting with each enabled mode:
"$TEST_PYTHON" blackbox_tests/resident_draft/check_http.py \
  --url http://127.0.0.1:30000 --scenario active --mode router \
  --reference /tmp/resident-draft-reference.json > /tmp/resident-draft-router.json

# Repeat for affinity, then for shortage/router and shortage/affinity.
```

Each run checks greedy text and committed usage at output limits 1, 2, 4 and 8,
then four concurrent sampled requests with different prompts, limits 1/2/3/5,
temperature, top-k and top-p. Active/baseline runs also check streaming equality,
a stop string spanning the output prefix, chat output accounting, disconnect
during streaming, and successful identical greedy completion after disconnect.
Shortage runs omit those repeated lifecycle checks.

An explicitly `off` service can use `--scenario active --mode off` against the
default-off reference to check compatibility. To cover combinations, rerun an
active configuration with the existing measured adaptive profile and/or reuse
cap, passing `--adaptive` and/or `--reuse` to the checker. Reuse changes target
routing, so that run checks local streaming/recovery consistency and skips exact
comparison with the original target. A service without `--moe-collect-stats`
requires `--collect-stats off`; both optional counters must be null. Such a run
does not replace either zero-transfer acceptance run.

## CPU checks

Compile only these independent test files to catch syntax errors before spending
GPU time. The CLI checks catch a missing option and unclear invalid-mode or
SD-disabled errors; fix the CLI if those fail after implementation is complete.
`check_cli.py` hides CUDA devices and does not start a valid service.

```sh
"$TEST_PYTHON" -m py_compile blackbox_tests/resident_draft/check_http.py blackbox_tests/resident_draft/check_cli.py
PYTHONPATH=python "$TEST_PYTHON" blackbox_tests/resident_draft/check_cli.py
```

## Evidence boundaries

This matrix covers the real BF16 offload checkpoint. It does not claim fused or
packed-format acceptance, exact probability-distribution measurement, mathematical
validation of router rankings or full-weight L2 nearest neighbors, or quantitative
resource-leak detection. Target verification can hide an incorrectly chosen draft
expert; final text and replacement counts cannot establish its exact identity.
Timing is reported without a performance threshold. Draft cannot change expert
cache contents within one round, so mid-round cache-loss fallback is unreachable
under the public contract and is not manufactured here.

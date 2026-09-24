# Resident-only draft: independent HTTP acceptance

These tests use the public CLI, HTTP responses and cumulative statistics only.
The author has not read production source, implementation diffs or other tests.
The coordinator starts servers and assigns GPUs; the HTTP runner never starts one.
Run against a dedicated service with no unrelated requests.

## Server command

The real BF16 Qwen3-30B-A3B checkpoint has 48 layers and 128 experts per layer.
Set `TEST_GPU` to the coordinator-assigned free GPU. For the baseline, omit the
residency flag so its default is exercised:

```sh
export TEST_PYTHON=/home/nengneng/miniconda3/envs/freetoken-dev/bin/python
PYTHONPATH=python "$TEST_PYTHON" -c 'from freetoken.cli import main; main()' serve \
  --model-path /data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B \
  --gpu "$TEST_GPU" --host 127.0.0.1 --port 30000 \
  --moe-backend offload --moe-cache-size 512 --moe-collect-stats \
  --disable-moe-prefill-overlap \
  --batching-policy legacy --max-running-requests 4 --cache-type radix \
  --max-seq-len-override 4096 --num-tokens 8192 --max-prefill-length 2048 \
  --sampling-defaults none --reasoning-parser off \
  --speculative-num-steps 3 --speculative-draft-experts 3
```

The active configuration adds `--speculative-draft-residency router`. The
shortage configuration also changes `--moe-cache-size` to `128`. With 48×3 > 128,
at least one layer must have fewer than K cached experts; the single-layer prefill
area still fits because overlap is disabled. Restart only between configurations.

Shortage needs a separate ordinary-target reference with the same cache budget:
use cache `128`, no resident list, `--speculative-num-steps 0`, residency `off`,
and `--cuda-graph-max-bs 0`. Keep the other model, sampling, concurrency and eager
execution settings identical. The cache512 reference is only for active cases.

## Matrix and HTTP commands

The baseline must actually draft, verify and load draft experts, with zero
residency stops. Router must draft and verify with zero draft loads; its
residency stops are recorded, not failures, because nothing pins experts while
the cache warms. Shortage must increase residency stops while draft and
verification counters stay unchanged. Failure changes the acceptance result;
missing draft/load coverage is a failure, not a passing zero.

```sh
"$TEST_PYTHON" blackbox_tests/resident_draft/check_http.py \
  --url http://127.0.0.1:30000 --scenario baseline --mode off \
  --reference /tmp/resident-draft-reference.json > /tmp/resident-draft-baseline.json

# After restarting with router:
"$TEST_PYTHON" blackbox_tests/resident_draft/check_http.py \
  --url http://127.0.0.1:30000 --scenario active --mode router \
  --reference /tmp/resident-draft-reference.json > /tmp/resident-draft-router.json

# After starting the cache128 ordinary target:
"$TEST_PYTHON" blackbox_tests/resident_draft/check_http.py \
  --url http://127.0.0.1:30000 --scenario ordinary-reference --mode off \
  --reference /tmp/resident-draft-ordinary128.json > /tmp/resident-draft-ordinary128-result.json

# After restarting cache128 with router and SD enabled:
"$TEST_PYTHON" blackbox_tests/resident_draft/check_http.py \
  --url http://127.0.0.1:30000 --scenario shortage --mode router \
  --reference /tmp/resident-draft-ordinary128.json > /tmp/resident-draft-shortage-router.json
```

Each run checks greedy text and committed usage at output limits 1, 2, 4 and 8,
then four concurrent sampled requests with different prompts, limits 1/2/3/5,
temperature, top-k and top-p. Active/baseline runs also check streaming equality,
a stop string spanning the output prefix, chat output accounting, disconnect
during streaming, and successful identical greedy completion after disconnect.
Shortage and ordinary-reference runs omit those repeated lifecycle checks. The
ordinary reference requires SD disabled and zero draft/accept/verify/load/stop
counters. Shortage requires exact text/usage equality and the same
`/v1/cache/status` expert-cache size as that ordinary reference.

An explicitly `off` service can use `--scenario active --mode off` against the
default-off reference to check compatibility. Use the fresh-service diagnostic
below to compare stream text under equal initial conditions. A service without
`--moe-collect-stats` requires `--collect-stats off`; `draft_expert_loads` must
be null. Such a run does not replace the zero-transfer router run.

## CPU checks

Compile only these independent test files to catch syntax errors before spending
GPU time. The CLI checks catch unclear invalid-mode or SD-disabled errors; fix the CLI if those fail after implementation is complete.
`check_cli.py` hides CUDA devices and does not start a valid service.

```sh
"$TEST_PYTHON" -m py_compile blackbox_tests/resident_draft/check_http.py blackbox_tests/resident_draft/check_cli.py
PYTHONPATH=python "$TEST_PYTHON" blackbox_tests/resident_draft/check_cli.py
```

## Stream/history diagnosis

On two separately started services with identical settings, run:
`python blackbox_tests/resident_draft/diagnose_stream.py --url URL --first plain --output plain.json`,
then use `--first stream --output stream.json` on the second fresh service.
The script retains responses, complete stream events and before/after statistics.
Only cold first requests have identical initial conditions; later calls include
sampled history and reveal whether repeated plain requests also change with
cache history. This records evidence without changing acceptance assertions.

## Evidence boundaries

This matrix covers the real BF16 offload checkpoint. It does not claim fused or
packed-format acceptance, exact probability-distribution measurement, mathematical
validation of router rankings, or quantitative resource-leak detection. Target
verification can hide an incorrectly chosen draft expert; final text cannot
establish its exact identity.
Timing is reported without a performance threshold. Draft cannot change expert
cache contents within one round, so mid-round cache-loss fallback is unreachable
under the public contract and is not manufactured here.

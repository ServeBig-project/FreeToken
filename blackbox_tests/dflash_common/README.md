# Independent DFlash acceptance

These tests use only the public contract in `docs/dflash-public-contract.md`
and the HTTP service. Their author has not read production code, implementation
diffs, or existing tests. The coordinator owns service startup, GPU allocation,
and execution against the baseline and changed revisions.

## Phase 1: existing self SD

Run the same requests on both supported target models, with fixed and adaptive
SD and Graph enabled. Keep model, launch arguments, GPU allocation, cache
capacity, and sampling settings identical when comparing revisions.

The service suite covers:

- Greedy and sampled responses, response structure, usage, and output limits.
- Streaming completion, explicit stop, and client cancellation followed by reuse.
- Concurrent requests at C1, C4, and C16, plus mixed lengths that create tail
  batches; Graph execution must be confirmed by its public counters.
- Cold and repeated prefixes, shared prefixes, separate cache groups, and a
  long input whose length is supplied by the coordinator for the configured
  prefill chunk size.
- Idle cache rebuild, rejected busy rebuild, and rejected invalid rebuild with
  a successful subsequent request.

Protocol, resource-lifecycle, or Graph coverage failures fail acceptance. Greedy
text differences between revisions are recorded separately for numerical and
quality review; equality alone does not establish numerical correctness.

Public statistics are saved before and after each workload so the coordinator
can assess actual draft, verification, acceptance, and Graph activity. Tests do
not infer private counters or use production modules as a reference.

## Run

The coordinator starts one isolated service with cache reporting enabled, waits
for `/health`, and runs:

```sh
python blackbox_tests/dflash_common/run_self_service.py \
  --url http://127.0.0.1:PORT --label qwen3-fixed-before \
  --adaptive no --graph yes --max-draft-steps 8 \
  --prefill-chunk-size PREFILL_CHUNK_SIZE --output /tmp/qwen3-fixed-before.json
```

Repeat for Qwen3 and Qwen3.6, fixed and adaptive, before and after the refactor.
Use the service's configured prefill chunk size. `--long-prompt-file` supplies a
different real long-input fixture when needed for the selected cache capacity.
Add `--require-c16-n8` only when the selected mode and capacity support eight
draft tokens at C16. A 96-slot non-Replay baseline can cap that batch at N4.

```sh
python blackbox_tests/dflash_common/compare_service.py \
  /tmp/qwen3-fixed-before.json /tmp/qwen3-fixed-after.json \
  --output /tmp/qwen3-fixed-comparison.json
```

Exit 0 means all asserted behavior and requested coverage passed. Exit 1 means
an observable behavior failed. Exit 2 means coverage was not demonstrated (or,
for comparison, the request inputs differed). Greedy differences are saved
separately, including cold-versus-reused comparisons within a run, and do not
change the exit code automatically. Sampling requests test protocol and limits;
they do not claim a distributional equivalence result.

The runner records every request body and public response needed to reproduce
a failure. It never starts or stops a service and never reads model code.

## DFlash numerical phase

New DFlash numerical tests wait for the documented public calculation entrypoint
and reference equations.

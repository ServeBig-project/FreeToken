# Runtime pool merge acceptance

Status: **pending**. This report uses only the public contract, HTTP results,
black-box drivers and their artifacts. No production source or implementation
review was read. The result reviewer does not start services or GPU work.

## Acceptance results

| Stage | Source | Result | Evidence to review |
|---|---|---|---|
| J1–J6 | `f4647b2` | Pending completion; coordinator reports J1/J2: 4 passed, J3 loading | [pytest log](/tmp/claude-1003/-home-nengneng-AIPrometheus-servebig-servebig-project/16c14e97-5fd3-4de1-a41f-67f277900f86/scratchpad/blackbox_j.log); [public artifacts](/home/nengneng/AIPrometheus/servebig/servebig-project/.worktrees/runtime-pool-blackbox/blackbox_tests/runtime_pool/_results) |
| Original 58-request trace | `17bbe02` | Pending | [trace artifacts](/data2/servebig-envs/runtime_pool_closeout_20261009/merge-final-trace) |
| Joint idle maintenance | `17bbe02` | Pending | [maintenance artifacts](/data2/servebig-envs/runtime_pool_closeout_20261009/joint-rebuild-final); [driver](../blackbox_tests/runtime_pool/test_service_joint_rebuild.py) |

Results from different source revisions remain attributed separately. Pending
stages do not establish acceptance of the final revision.

## Fixed public configuration

- Model: Qwen3.6-35B-A3B-NVFP4; RTX 4090; hybrid expert backend.
- Initial expert capacity 5000; runtime 8.626953125 GiB; concurrency 8;
  context 49152; Graph maximum batch 8; ReplaySSM; layered-pipeline;
  DFlash with 4 draft steps.
- The final trace reuses the original 58 requests without reducing prompts or
  output limits. Graph, SD and concurrency remain enabled at the same settings.
- Maintenance runs once after the trace is idle: runtime 4.75 GiB and expert
  capacity 7200, supplied together to `POST /v1/cache/rebuild`.

## Public resource calculation

The [reference status](/data2/servebig-envs/runtime_pool_closeout_20261009/hybrid-trace-v4/final-cache-status.json)
publishes 1,775,616 bytes per expert, a cache budget of 18,684,425,011 bytes,
expert limits 256–10240 and 10240 model experts.

| Configuration | Runtime bytes | Expert bytes | Combined bytes |
|---|---:|---:|---:|
| Before: 8.626953125 GiB / 5000 | 9,263,120,384 | 8,878,080,000 | 18,141,200,384 |
| Requested: 4.75 GiB / 7200 | 5,100,273,664 | 12,784,435,200 | 17,884,708,864 |

The requested total decreases by 256,491,520 bytes and fits the published
budget and expert limits. This establishes the resource relationship, not a
successful rebuild. The driver repeats this calculation against live geometry.

## Final evidence still required

- J1–J6: final test counts, public failures and any missing cases.
- Trace: request identity and unchanged input/output limits; all completions,
  errors, finish reasons and usage; actual Graph/SD execution; unchanged
  expert/runtime budgets and configured concurrency.
- Maintenance: a recorded 64-token reference; one successful joint rebuild;
  exact requested geometry with concurrency/context/Replay/Graph/DFlash kept;
  the same prompt and eight new requests completing with valid SSE/usage;
  increasing Graph/SD counters and matching public request/token totals.

Cross-plan text equality and a prescribed number of pauses are not acceptance
conditions. The driver records text comparison without requiring equality.

## Changes

Driver commit `2bccd62`: production +0/-0; black-box tests +117/-0, net +117.
CPU syntax and CLI entry checks passed; service acceptance has not run here.

# Independent public HTTP acceptance

This client reads only HTTP responses and its own earlier reports. Start the
server separately; the client neither imports server code nor launches a GPU process.

```sh
python blackbox_tests/sd_portability/run_http.py \
  --url http://127.0.0.1:8000 --mode graph --expected-steps 3 \
  --resources --lifecycle --output reports/sd-http-qwen36-n3.json
```

Use `--mode eager` for an explicitly eager server. Use `--expected-steps 0` only
to collect an AR reference; such a run cannot establish SD coverage. Every SD run
requires real draft/verify activity, and Graph runs require both replay counters
to advance. The initial counters may be nonzero; checks use the current run's delta.

The first phase covers health/model discovery, text and chat output, SSE completion,
exact final token usage, same-group and other-group prefixes, fixed resource geometry,
invalid rebuild mode, idle rebuild, and subsequent generation. `--resources` permits
cache rebuilds, which discard cached prefixes. `--state-slots N` selects a capacity
within the public limits; the original capacity is restored even on failure. It can
increase a naive cache's state pool to exercise SD or reduce an existing pool. A smaller
pool records its complete draft histogram; this alone cannot attribute shorter drafts
to capacity because the output tail can also shorten them. A rebuilt
pool must exercise SD and, when configured, draft/verify Graph to claim that coverage.

`--reference FILE` compares complete text with an earlier report made using identical
public requests. Same-group, different-group, stream and reference text differences
remain investigation items without a similarity threshold. Every input, full response,
usage value and statistics snapshot is saved in the JSON report.

Exit status: `0` means these checks passed; `1` is a public-contract failure; `2` means
text differences need investigation; `3` means a required execution path was not observed.
The report always retains all categories, even when several apply. Passing this phase
does not claim the full configuration matrix.

`--lifecycle` adds string/array stop and EOS attempts followed by identical prefix
requests, closes a live stream and verifies subsequent service, and sends four concurrent
requests of 8/21/48/65 tokens. It reports whether all four were actually active and
whether Graph replayed both full and uneven batches. Stop, EOS or cancellation that did
not actually happen remain uncovered. This phase also checks text-list prompts and
the explicit HTTP 400 rejection of token-ID input followed by successful service.
`--only eos prompt-input` rechecks just these behaviors without repeating core,
stop, cancellation or concurrent generation.

`--only generated-prefix` tests the stricter generated-prefix requirement separately.
Each stop/EOS/cancellation source gets a fresh cache group; the follow-up includes its
retained output and is compared with the same complete request in another fresh group.
Stop is placed after retained text, and cancellation waits for a retained text segment.
Coverage requires public `usage.prompt_tokens_details.cached_tokens` greater than the
source prompt's token count. Missing counters or hits confined to the original prompt
remain uncovered. The original same-input tests establish recovery and comparisons,
but cannot on their own prove reuse of previously generated content. Lifecycle mode
also includes these probes (at most 544 requested tokens plus a cancelled stream).

## Fixed task quality

```sh
python blackbox_tests/sd_portability/quality.py \
  --url http://127.0.0.1:8000 --mode graph --expected-steps 0 --output reports/quality-ar.json
python blackbox_tests/sd_portability/quality.py \
  --url http://127.0.0.1:8001 --mode graph --expected-steps 8 \
  --reference reports/quality-ar.json --output reports/quality-sd.json
```

Both servers must expose the same model id. The eight fixed tasks comprise two
arithmetic answers, two JSON transformations, two Python execution results and two
short functions checked on fixed examples. JSON answers must parse in full. Function
answers may have a single Markdown code fence; the generated functions run in a
separate Python process with a two-second limit. Each generation allows at most 256
tokens. Public `geometry.reasoning.kwargs.off` fixes the checkpoint's thinking-off
template settings identically for AR and SD; the EOS lifecycle check also uses them.
`--only coding_alias` isolates that unchanged task; its own before/after counters must
still demonstrate SD and, when requested, draft/verify Graph activity.

The report records each task's complete request, output, usage and boolean score.
Task failures are visible even in the AR reference. A previously correct task becoming
wrong is a regression (exit 1). Any remaining incorrect task gives exit 4; complete
text differences still give exit 2 even when every task is correct. No partial-credit
or text-similarity threshold is used. At most 2,048 tokens are requested per quality run.

Runtime depends on model throughput. The normal phase generates 288 output tokens;
resource checks add 80 tokens and two cache rebuilds. Lifecycle checks request at most
718 tokens plus a stream closed after its first output; prompt input checks add 32 tokens.
Request timeout defaults to 180 s.

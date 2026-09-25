# Independent public HTTP acceptance

This client reads only HTTP responses and its own earlier reports. Start the
server separately; the client neither imports server code nor launches a GPU process.

```sh
python blackbox_tests/sd_portability/run_http.py \
  --url http://127.0.0.1:8000 --mode graph --expected-steps 3 \
  --resources --output reports/sd-http-qwen36-n3.json
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
pool without observed shortened drafts is reported as uncovered, not passed. A rebuilt
pool must exercise SD and, when configured, draft/verify Graph to claim that coverage.

`--reference FILE` compares complete text with an earlier report made using identical
public requests. Same-group, different-group, stream and reference text differences
remain investigation items without a similarity threshold. Every input, full response,
usage value and statistics snapshot is saved in the JSON report.

Exit status: `0` means these checks passed; `1` is a public-contract failure; `2` means
text differences need investigation; `3` means a required execution path was not observed.
The report always retains all categories, even when several apply. Passing this phase
does not claim concurrency, cancellation, EOS, stop or the full configuration matrix.

Runtime depends on model throughput. The normal phase generates 288 output tokens;
resource checks add 80 tokens and two cache rebuilds. Request timeout defaults to 180 s.

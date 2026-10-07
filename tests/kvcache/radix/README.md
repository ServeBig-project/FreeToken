# radix prefix-cache tests

Scenario tests for `freetoken.kvcache.radix_cache.RadixCache`, one module per component mix:

| module | tree |
|---|---|
| `test_plain_radix.py` | plain KV |
| `test_swa_radix.py` | `window=` (sliding window) |
| `test_hybrid_radix.py` | `has_state=True` (recurrent state) |

`harness.py` drives one tree through its public interface only (`match`, `insert`, `lock`/`unlock`,
`evict_kv`/`evict_window`/`evict_states`, `trim_head_window`, `evictable`/`protected`). Slot ids
are globally unique and never reused, so every returned location or state names exactly one
hand-out; the ledgers raise on a double free and `Session.check()` runs `check_integrity` plus
KV and state conservation.

`conftest.py` patches `time.monotonic_ns` with a counter so LRU order assertions never depend on
clock resolution.

# Prefix cache groups

`/v1/completions` and `/v1/chat/completions` accept `cache_group`, a string:

```json
{"model":"local-model","prompt":"Unmodified original prompt","max_tokens":4,"cache_group":"user-1"}
```

Requests can reuse prefix KV and linear-attention snapshots only within the
same group. The group is request metadata, not a token or text prefix: prompt
rendering, token counts, and model computation are unchanged. Omitting it is
equivalent to `""`, the original default group.

All groups share the same total KV capacity and eviction policy. Active
requests keep their usual locks. Model weights and the MoE expert cache remain
shared; there is no per-group GPU allocation or fixed quota.

For a multi-user replay, keep one group for all of a user's agents and cases,
and assign a different group to every other user. Observe actual reuse through
`usage.prompt_tokens_details.cached_tokens`; request IDs alone do not separate
prefix matching. Short prompts may have no reusable linear-state boundary.

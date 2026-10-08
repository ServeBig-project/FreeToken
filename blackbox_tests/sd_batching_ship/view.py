"""Read the public status objects (contract section 4).

The contract names the information but not every JSON key, so each semantic value maps to the
dotted paths observed in real `/v1/stats` responses. A value the response does not carry raises an
assertion that lists near matches: a missing contract field is a finding, not a skip.
"""
import json
import re

PATHS = {
    # execution.effective (section 4)
    "batching": ["execution.effective.batching_policy", "execution.effective.batching"],
    "drafter": ["execution.effective.drafter", "execution.effective.draft", "speculative.drafter"],
    "steps": ["execution.effective.speculative_num_steps", "execution.effective.num_steps",
              "execution.effective.steps", "speculative.num_steps"],
    "phase": ["execution.effective.speculative_phase", "execution.effective.phase", "speculative.phase"],
    "graph": ["execution.effective.cuda_graph", "execution.effective.graph"],
    "req_steps": ["execution.requested.speculative_num_steps", "execution.requested.num_steps",
                  "execution.requested.steps", "execution.requested.max_draft_steps"],
    # speculative counters (section 4, cumulative)
    "drafted": ["speculative.draft_tokens", "speculative.drafted_tokens", "speculative.num_draft_tokens"],
    "accepted": ["speculative.accepted_tokens", "speculative.num_accepted_tokens"],
    "rounds": ["speculative.verify_rounds", "speculative.rounds", "speculative.num_verify_rounds"],
    "sd_inwave": ["speculative.inwave_rounds", "speculative.phase_rounds.inwave",
                  "speculative.execution.inwave"],
    "sd_outwave": ["speculative.outwave_rounds", "speculative.phase_rounds.outwave",
                   "speculative.execution.outwave"],
    "draft_len_hist": ["speculative.draft_length_histogram", "speculative.draft_len_hist"],
    "ar_fallback": ["speculative.ar_fallbacks", "speculative.fallbacks", "speculative.ar_rounds_by_reason"],
    "exec_errors": ["speculative.execution_errors", "speculative.errors"],
    "graph_replays": ["cuda_graph.replays", "cuda_graph.replay_count", "cuda_graph.num_replays"],
}


def flat(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def _at(obj, path):
    for k in path.split("."):
        if not isinstance(obj, dict) or k not in obj:
            raise KeyError(path)
        obj = obj[k]
    return obj


def get(stats, name):
    for p in PATHS[name]:
        try:
            return _at(stats, p)
        except KeyError:
            continue
    word = name.split("_")[0]
    near = {p: v for p, v in flat(stats).items() if re.search(word, p, re.I)}
    raise AssertionError(f"/v1/stats carries no '{name}' (tried {PATHS[name]}); near: "
                         f"{json.dumps(near, default=str)[:1500]}")


def num(stats, name):
    v = get(stats, name)
    if isinstance(v, dict):
        return sum(x for x in flat(v).values() if isinstance(x, (int, float)) and not isinstance(x, bool))
    return v or 0


def reasons(stats, name="ar_fallback"):
    """Per-reason AR fallback counts as {reason: count}."""
    v = get(stats, name)
    if isinstance(v, dict):
        return flat(v)
    return {"total": v}


def reason_count(stats, pattern):
    return sum(c for k, c in reasons(stats).items()
               if re.search(pattern, k, re.I) and isinstance(c, (int, float)))


def fallback_text(stats):
    """execution.fallback_reasons rendered as text for keyword checks."""
    return json.dumps((stats.get("execution") or {}).get("fallback_reasons"), default=str).lower()


def sd_on(stats):
    d = get(stats, "drafter")
    return bool(num(stats, "steps")) and d not in (None, "", "none", "off", False)


def delta(before, after, name):
    return num(after, name) - num(before, name)


def text(obj):
    return json.dumps(obj, default=str).lower()

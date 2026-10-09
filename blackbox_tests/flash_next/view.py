"""Read values from /v1/stats and /v1/cache/status by dotted path (paths observed in real responses)."""


def flat(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def at(obj, path):
    """Value at a dotted path; a missing contract field fails and lists the nearby keys."""
    cur = obj
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            word = path.split(".")[-1].split("_")[0]
            near = sorted(p for p in flat(obj) if word in p)[:40]
            raise AssertionError(f"missing {path}; nearby: {near}")
        cur = cur[k]
    return cur

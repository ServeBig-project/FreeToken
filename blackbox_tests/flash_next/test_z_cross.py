"""Cross-session comparisons over the recorded outputs (no server launch).

Path pairs: same precision, different execution paths -> same input history (consistent outputs).
Quant pairs: fp8 dense / int8 KV vs off -> bounded quality change. Reference: candidate vs upstream.
"""
import pytest

from . import limits, tasks
from .conftest import load
from .scenarios import consistent, batch_members  # noqa: F401
from .sessions import PATH_PAIRS, QUANT_PAIRS, session


def rec(key):
    r = load(session(key)["name"])
    if r is None:
        pytest.skip(f"session {key} has no recorded results")
    return r


def checkers(tok):
    out = {n: chk for n, (_, _, chk) in tasks.single_tasks(tok).items()}
    for L in tasks.BOUNDARY_LENGTHS:
        _, ph = tasks.boundary_prompt(tok, L)
        out[f"boundary{L}"] = lambda x, ph=ph: tasks.answer_only(x).startswith(ph)
    for pid, _, gold in tasks.gsm8k():
        out[f"gsm{pid}"] = lambda x, g=gold: tasks.gsm8k_ok(x, g)
    return out


def outputs(r):
    """case -> greedy text for inputs that are identical across sessions."""
    o = {n: v["text"] for n, v in r.get("alone", {}).items()}
    o.update({f"boundary{L}": v["first"]["text"] for L, v in r.get("boundary", {}).items()})
    o.update({f"gsm{k}": v["text"] for k, v in r.get("gsm8k", {}).get("per", {}).items()})
    return o


def conv_pairs(a, b):
    """Turn i of a conversation has the same input only while all earlier replies are identical."""
    for name, ta in a.get("conv", {}).items():
        tb = b.get("conv", {}).get(name, [])
        for i, (x, y) in enumerate(zip(ta, tb)):
            chk = tasks.CONVERSATIONS[name][i][2] or (lambda _: True)
            yield f"{name}[{i}]", x["text"], y["text"], chk
            if x["text"] != y["text"]:
                break


def compare_sessions(a, b, tok):
    chk, oa, ob = checkers(tok), outputs(a), outputs(b)
    same, diff, bad = [], [], []
    rows = [(k, oa[k], ob[k], chk.get(k, lambda _: True)) for k in sorted(oa.keys() & ob.keys())]
    for k, x, y, c in rows + list(conv_pairs(a, b)):
        (same if x == y else diff).append(k)
        if not consistent(x, y, c):
            bad.append(f"{k}: {x[:120]!r} vs {y[:120]!r}")
    return same, diff, bad


@pytest.fixture(scope="module")
def tok():
    return tasks.Tok()


@pytest.mark.parametrize("left,right,what", PATH_PAIRS)
def test_path_consistency(left, right, what, tok):
    a, b = rec(left), rec(right)
    same, diff, bad = compare_sessions(a, b, tok)
    print(f"{left} vs {right} ({what}): identical {len(same)}, differing {len(diff)}: {diff}")
    assert not bad, bad


@pytest.mark.parametrize("base,quant,what", QUANT_PAIRS)
def test_quantization_quality(base, quant, what, tok):
    a, b = rec(base), rec(quant)
    ga, gb = a["gsm8k"]["correct"], b["gsm8k"]["correct"]
    same, diff, bad = compare_sessions(a, b, tok)
    print(f"{quant} vs {base} ({what}): GSM8K {gb} vs {ga}; identical {len(same)}, differing {len(diff)}")
    assert gb >= ga - limits.GSM_MAX_DROP, f"{what}: GSM8K {gb} vs {ga}"
    assert not bad, bad


def test_candidate_vs_upstream(tok):
    ref, a = rec("R"), rec("A")
    gr, ga = ref["gsm8k"]["correct"], a["gsm8k"]["correct"]
    same, diff, bad = compare_sessions(ref, a, tok)
    print(f"A vs upstream: GSM8K {ga} vs {gr}; identical {len(same)}, differing {len(diff)}: {diff}")
    assert ga >= gr - limits.GSM_MAX_DROP, f"GSM8K candidate {ga} vs upstream {gr}"
    assert not bad, bad


def geometry(r):
    return r["cache_ready"]["geometry"]


@pytest.mark.parametrize("base,quant,kind", [("A", "F", "fp8"), ("A", "E", "int8")])
def test_precision_frees_memory(base, quant, kind):
    """Same flags except the precision option: KV bytes per token (int8) and auto expert slots grow."""
    a, b = geometry(rec(base)), geometry(rec(quant))
    slots = b["moe_cache_size"] - a["moe_cache_size"]
    kv = b["unit_bytes"]["kv_per_token"] / a["unit_bytes"]["kv_per_token"]
    print(f"{quant} vs {base}: expert slots {a['moe_cache_size']} -> {b['moe_cache_size']} ({slots:+d}), "
          f"KV bytes/token {a['unit_bytes']['kv_per_token']} -> {b['unit_bytes']['kv_per_token']} ({kv:.3f}), "
          f"KV tokens {a['num_pages'] * a['page_size']} -> {b['num_pages'] * b['page_size']}")
    assert b["num_pages"] * b["page_size"] >= a["num_pages"] * a["page_size"]
    if kind == "int8":
        lo, hi = limits.KV_INT8_BYTES_RATIO
        assert lo <= kv <= hi, f"int8 KV bytes/token ratio {kv:.3f} outside [{lo}, {hi}]"
        assert slots >= limits.INT8_MIN_EXTRA_SLOTS, f"int8 KV freed only {slots} expert slots"
    else:
        assert kv == 1.0, f"fp8 dense changed KV bytes/token ratio to {kv}"
        assert slots >= limits.FP8_MIN_EXTRA_SLOTS, f"fp8 dense freed only {slots} expert slots"

"""Checks run in every session. A session module sets SESSION and does `from .scenarios import *`.

Outputs are recorded in se.rec for the cross-session comparisons in test_z_cross.py.
"""
import time

import pytest

from . import env, limits, tasks
from .client import cached, common_prefix, parallel

__all__ = ["test_01_single_tasks_alone", "test_02_conversations", "test_03_chat_api",
           "test_04_boundary_lengths", "test_05_fork_and_groups", "test_06_batched_vs_alone",
           "test_07_gsm8k", "test_08_stream_stop_eos", "test_09_bad_requests", "test_10_cancel_then_serve"]


def keep(r):
    return {k: r.get(k) for k in ("text", "finish", "usage", "logprobs", "seconds")}


def consistent(a, b, check):
    """Same input, same greedy config, different path: identical, or both correct and not degenerate."""
    if a == b:
        return True
    return check(a) and check(b) and not tasks.degenerate(a) and not tasks.degenerate(b)


def compare(se, label, got, want, check, fails):
    same = got == want
    se.rec.setdefault("pairs", {})[label] = dict(same=same, common_chars=common_prefix(got, want),
                                                 got=got[:200], want=want[:200])
    if not consistent(got, want, check):
        fails.append(f"{label}: inconsistent and not both correct: got={got[:160]!r} want={want[:160]!r}")


def radix(se):
    return se.cfg.get("radix", True)


def _single(se):
    if "single" not in se.__dict__:
        se.__dict__["single"] = tasks.single_tasks(se.tok)
    return se.__dict__["single"]


def test_01_single_tasks_alone(se):
    """Fixed short tasks, one at a time (prefix miss), including 3k/6k/12k needles across prefill chunks."""
    alone, fails = se.rec.setdefault("alone", {}), []
    for name, (p, m, chk) in _single(se).items():
        r = se.c.complete(p, m, group="alone")
        alone[name] = keep(r)
        if not chk(r["text"]) or tasks.degenerate(r["text"]):
            fails.append(f"{name}: {r['text'][:300]!r}")
        if r["usage"].get("prompt_tokens") != se.tok.n(p):
            fails.append(f"{name}: prompt_tokens {r['usage'].get('prompt_tokens')} != tokenizer {se.tok.n(p)}")
    assert not fails, fails


def run_conversation(se, name, group):
    msgs, out = [], []
    for i, (content, m, chk) in enumerate(tasks.CONVERSATIONS[name]):
        if content is None:
            content = tasks.research_first(se.tok)
        msgs.append({"role": "user", "content": content})
        p = se.tok.chat(msgs)
        r = se.c.complete(p, m, group=group)
        out.append(dict(keep(r), prompt_tokens=se.tok.n(p), ok=(chk(r["text"]) if chk else None)))
        msgs.append({"role": "assistant", "content": tasks.answer_only(r["text"])})
    return out


def test_02_conversations(se):
    """Multi-turn chat, coding and research: later turns extend the earlier prompt plus the model's reply."""
    conv, fails = se.rec.setdefault("conv", {}), []
    for name in tasks.CONVERSATIONS:
        turns = run_conversation(se, name, group="conv")
        conv[name] = turns
        for i, t in enumerate(turns):
            if t["ok"] is False or tasks.degenerate(t["text"]):
                fails.append(f"{name} turn {i}: {t['text'][:300]!r}")
            if cached(t) > t["prompt_tokens"]:
                fails.append(f"{name} turn {i}: cached {cached(t)} > prompt {t['prompt_tokens']}")
    r2 = conv["research_mt"][1]
    if radix(se) and cached(r2) < limits.HIT_MIN_PROMPT:
        fails.append(f"research turn 2 reused only {cached(r2)} tokens of a {r2['prompt_tokens']}-token prompt")
    if not radix(se) and any(cached(t) for c in conv.values() for t in c):
        fails.append("naive cache reported cached tokens")
    assert not fails, fails


def test_03_chat_api(se):
    """Server-side chat template with default thinking: reasoning split off, answer in content, EOS stop."""
    msgs = [{"role": "user", "content": "What is 17 multiplied by 23? Reply with only the number."}]
    r = se.c.chat(msgs, 4096)
    se.rec["chat_api"] = r
    assert r["finish"] == "stop", r
    assert tasks.numbers(r["text"]) == ["391"], r
    assert "<think>" not in r["text"] and "</think>" not in r["text"], r


def test_04_boundary_lengths(se):
    """Exact prompt lengths around multiples of 4/64, the prefill chunk and the QSA budget; then repeat."""
    out, fails = se.rec.setdefault("boundary", {}), []
    for L in tasks.BOUNDARY_LENGTHS:
        p, phrase = tasks.boundary_prompt(se.tok, L)
        chk = lambda x, ph=phrase: tasks.answer_only(x).startswith(ph)
        a = se.c.complete(p, 12, group="bd")
        b = se.c.complete(p, 12, group="bd")
        out[str(L)] = dict(first=keep(a), repeat=keep(b), phrase=phrase)
        if a["usage"].get("prompt_tokens") != L:
            fails.append(f"L={L}: prompt_tokens {a['usage'].get('prompt_tokens')}")
        if not chk(a["text"]):
            fails.append(f"L={L}: wrong answer {a['text']!r} (want {phrase})")
        compare(se, f"boundary{L}:repeat_vs_first", b["text"], a["text"], chk, fails)
        if cached(b) > L or cached(a) > L:
            fails.append(f"L={L}: cached {cached(a)}/{cached(b)} > prompt")
        if radix(se) and L >= limits.HIT_MIN_PROMPT and cached(b) < limits.HIT_MIN_FRACTION * L:
            fails.append(f"L={L}: exact repeat reused only {cached(b)} tokens")
        if not radix(se) and cached(b):
            fails.append(f"L={L}: naive cache reported {cached(b)} cached tokens")
    assert not fails, fails


FORK_DOC_FACTS = " Ship Orion carries 45 crates. Ship Vega carries 78 crates. "


def fork_prompts(se):
    doc = se.tok.prose(2500, 41)
    k = len(doc) // 3
    doc = doc[:k] + FORK_DOC_FACTS + doc[k:]
    q1 = se.tok.chat([{"role": "user", "content": doc + "\n\nHow many crates does Ship Vega carry? "
                                                         "Reply with only the number."}])
    q2 = se.tok.chat([{"role": "user", "content": doc + "\n\nWhich ship carries fewer crates? "
                                                         "Reply with the ship name only."}])
    return (q1, lambda x: tasks.numbers(x) == ["78"]), (q2, lambda x: "orion" in tasks.answer_only(x).lower())


def test_05_fork_and_groups(se):
    """Same prefix, different continuations; another cache_group must not reuse it; outputs agree."""
    (q1, c1), (q2, c2) = fork_prompts(se)
    a = se.c.complete(q1, 16, group="fork")
    b = se.c.complete(q2, 16, group="fork")
    iso = se.c.complete(q2, 16, group="fork-isolated")
    se.rec["fork"] = dict(q1=keep(a), q2_hit=keep(b), q2_other_group=keep(iso))
    fails = []
    for n, r, chk in (("q1", a, c1), ("q2_hit", b, c2), ("q2_other_group", iso, c2)):
        if not chk(r["text"]):
            fails.append(f"{n}: wrong {r['text']!r}")
    # reuse at a fork point depends on where recurrent states were kept, so it is recorded, not required
    compare(se, "fork:q2_after_q1_vs_q2_other_group", b["text"], iso["text"], c2, fails)
    if cached(iso):
        fails.append(f"fork: first request of another cache_group reused {cached(iso)} tokens")
    assert not fails, fails


def batch_members(se):
    single = _single(se)
    m = [(n, single[n][0], single[n][1], single[n][2]) for n in ("arith", "capital", "code_prime", "needle_3k")]
    for L in (1025, 2049):
        p, ph = tasks.boundary_prompt(se.tok, L)
        m.append((f"boundary{L}", p, 12, lambda x, ph=ph: tasks.answer_only(x).startswith(ph)))
    gid, gq, gold = tasks.gsm8k()[0]
    m.append(("gsm0", se.tok.chat([{"role": "user", "content": gq}]), 768, lambda x: tasks.gsm8k_ok(x, gold)))
    return m


def test_06_batched_vs_alone(se):
    """Non-full concurrent batch plus a 6k prompt arriving mid-decode; misses then hits; vs alone."""
    if "alone" not in se.rec:
        pytest.skip("needs test_01")
    members = batch_members(se)
    alone = dict(se.rec["alone"])
    for n, p, m, chk in members:  # members not in test_01 get an alone run here
        if n not in alone:
            alone[n] = keep(se.c.complete(p, m, group="alone"))
    long_name, (lp, lm, lchk) = "needle_6k", _single(se)["needle_6k"]
    fails, out = [], se.rec.setdefault("batched", {})
    for group in ("batch-miss", "alone"):
        jobs = [(se.c.complete, (p, m), dict(group=group)) for _, p, m, _ in members]
        jobs.append((se.c.complete, (lp, lm), dict(group=group)))
        res = parallel(jobs, stagger_s=0.05)
        names = [n for n, *_ in members] + [long_name]
        checks = [c for *_, c in members] + [lchk]
        for n, r, chk in zip(names, res, checks):
            out[f"{group}:{n}"] = keep(r)
            if not chk(r["text"]):
                fails.append(f"{group}:{n}: wrong {r['text'][:200]!r}")
            compare(se, f"batched[{group}]:{n}_vs_alone", r["text"], alone[n]["text"], chk, fails)
    se.rec["alone"].update({k: v for k, v in alone.items() if k not in se.rec["alone"]})
    assert not fails, fails


def test_07_gsm8k(se):
    """32 GSM8K test problems, thinking off, greedy, submitted 8 at a time."""
    probs = tasks.gsm8k()
    jobs = [(se.c.complete, (se.tok.chat([{"role": "user", "content": q}]), 768), dict(group="gsm"))
            for _, q, _ in probs]
    res = []
    for i in range(0, len(jobs), 8):
        res += parallel(jobs[i:i + 8])
    per = {str(pid): dict(ok=tasks.gsm8k_ok(r["text"], gold), text=r["text"], finish=r["finish"],
                          tokens=r["usage"].get("completion_tokens"))
           for (pid, _, gold), r in zip(probs, res)}
    correct = sum(v["ok"] for v in per.values())
    se.rec["gsm8k"] = dict(correct=correct, n=len(probs), per=per)
    assert correct >= limits.GSM_MIN_CORRECT, f"GSM8K {correct}/{len(probs)}"


def test_08_stream_stop_eos(se):
    """SSE equals non-stream; stop string cuts output; EOS ends early under a large max_tokens."""
    fails = []
    p, m, chk = _single(se)["needle_3k"]
    st = se.c.stream(p, m, group="alone")
    ns = se.c.complete(p, m, group="alone")
    se.rec["stream"] = dict(stream=st, nonstream=keep(ns))
    if not st["done"]:
        fails.append("stream did not end with [DONE]")
    if st["text"] != ns["text"]:
        fails.append(f"stream text {st['text']!r} != non-stream {ns['text']!r}")
    if (st["usage"] or {}).get("completion_tokens") != ns["usage"].get("completion_tokens"):
        fails.append(f"stream usage {st['usage']} != {ns['usage']}")
    cp = se.tok.chat([{"role": "user", "content": "List the integers from 1 to 30 separated by commas and "
                                                  "spaces, nothing else."}])
    s = se.c.complete(cp, 200, stop=[", 15"])
    se.rec["stop"] = keep(s)
    if s["finish"] != "stop" or ", 15" in s["text"] or "14" not in s["text"]:
        fails.append(f"stop string: {s}")
    eos = se.c.complete(_single(se)["arith"][0], 4000)
    se.rec["eos"] = keep(eos)
    if eos["finish"] != "stop" or eos["usage"].get("completion_tokens", 9999) > 16:
        fails.append(f"EOS early stop: {eos['finish']} {eos['usage']}")
    exact = se.c.complete(_single(se)["capital"][0], 37, ignore_eos=True)
    se.rec["ignore_eos"] = keep(exact)
    if exact["finish"] != "length" or exact["usage"].get("completion_tokens") != 37:
        fails.append(f"ignore_eos max_tokens=37: {exact['finish']} {exact['usage']}")
    assert not fails, fails


def test_09_bad_requests(se):
    """Over-long prompt and malformed bodies get a readable 4xx; the server keeps serving correctly."""
    if se.reference:
        pytest.skip("errors are a candidate contract item")
    over = se.tok.chat([{"role": "user", "content": se.tok.prose(env.CTX + 300, 51)}])
    bad = {
        "over_context": dict(model=se.c.model, prompt=over, max_tokens=8, temperature=0),
        "no_prompt": dict(model=se.c.model, max_tokens=8),
    }
    fails, out = [], {}
    for k, body in bad.items():
        r = se.c.post("/v1/completions", body)
        out[k] = dict(status=r.status_code, body=r.text[:400])
        if not 400 <= r.status_code < 500 or len(r.text.strip()) < 10:
            fails.append(f"{k}: HTTP {r.status_code} {r.text[:200]!r}")
    se.rec["bad_requests"] = out
    p, m, chk = _single(se)["arith"]
    after = se.c.complete(p, m, group="after-errors")
    compare(se, "after_errors:arith_vs_alone", after["text"], se.rec["alone"]["arith"]["text"], chk, fails)
    assert not fails, fails


def wait_idle(se, timeout=limits.IDLE_TIMEOUT_S):
    """Idle per /v1/stats requests.active; returns (ok, last stats.requests)."""
    end, last = time.time() + timeout, None
    while time.time() < end:
        last = se.c.get("/v1/stats").get("requests") or {}
        if last.get("active") == 0:
            return True, last
        time.sleep(1)
    return False, last


def test_10_cancel_then_serve(se):
    """Cancel in decode, in a long prefill, while queued and while sharing a prefix; then serve again."""
    if se.reference:
        pytest.skip("cancellation is a candidate contract item")
    single, fails = _single(se), []
    code_p = single["code_prime"][0]
    se.c.stream(code_p, 1500, group="cancel", ignore_eos=True, stop_after_chunks=10)  # in decode
    se.c.abort_after(single["needle_12k"][0], 16, 0.5, group="cancel")  # in prefill
    queued = [(se.c.abort_after, (code_p, 300, 0.05), dict(group=f"cancel-q{i}")) for i in range(12)]
    parallel(queued)
    # two requests sharing a 2.5k-token prefix; cancel one mid-flight, the other must finish correctly
    (q1, c1), (q2, c2) = fork_prompts(se)
    res = parallel([(se.c.abort_after, (q1, 200, 1.0), dict(group="cancel-share")),
                    (se.c.complete, (q2, 16), dict(group="cancel-share"))])
    fails += [] if c2(res[1]["text"]) else [f"surviving shared-prefix request wrong: {res[1]['text']!r}"]
    ok, last = wait_idle(se)
    se.rec["cancel"] = dict(idle=ok, last=last, survivor=keep(res[1]))
    if not ok:
        fails.append(f"not idle {limits.IDLE_TIMEOUT_S}s after cancels: {str(last)[:600]}")
    for n in ("arith", "needle_3k"):
        p, m, chk = single[n]
        r = se.c.complete(p, m, group="after-cancel")
        compare(se, f"after_cancel:{n}_vs_alone", r["text"], se.rec["alone"][n]["text"], chk, fails)
    assert not fails, fails

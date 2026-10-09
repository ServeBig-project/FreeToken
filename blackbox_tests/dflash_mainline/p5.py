"""P5 (small performance regression): fixed N4, compact vs full DFlash storage, identical explicit
expert/KV/GDN capacities. Contract section 6 item 2 gate: compact decode <= 2% slower than full.

Frozen workloads (greedy, fresh cache group per wave, one warm-up wave excluded):
  C1 short  3 reps x 4 sequential code prompts, 192 output tokens
  C4 long   3 reps x 4 concurrent ~5.8k-token prompts, 256 output tokens
  C16 long  2 reps x 16 concurrent ~5.8k-token prompts, 128 output tokens
Metric: mean over requests of (output tokens - 1) / (last chunk - first chunk), median over reps.
Decision (fixed before running): pass if compact >= 0.98 x full; fail if below and the gap exceeds
both arms' rep spread (max - min); otherwise inconclusive.
"""

import json
import statistics
import subprocess
import threading
import time
import urllib.request

from transformers import AutoTokenizer

from harness import GPU, dflash, document, qwen36
from p2 import fresh
from workloads import CODE_HEADER, CODE_TASKS, acceptance, spec_delta

GATE = 0.98
BIG = dict(policy="layered-pipeline", tokens=131072, running=16, graph=16, gdn=4600000000)

CONFIGS = {
    "perf_compact": qwen36(**BIG) + dflash(4),
    "perf_full": qwen36(**BIG) + dflash(4, "--no-dflash-compact-kv"),
}


def timed_stream(server, prompt, max_tokens):
    body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "cache_group": fresh()}
    req = urllib.request.Request(server.url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out, first, last = "", None, None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            chunk = json.loads(line[5:])
            piece = chunk["choices"][0].get("text", "") if chunk.get("choices") else ""
            if piece:
                last = time.monotonic()
                first = first or last
                out += piece
    return {"text": out, "first": first, "last": last}


def gpu_mib():
    rows = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.splitlines()
    return max(int(r.split(",")[1]) for r in rows if r.split(",")[0].strip() == GPU)


def wave(server, tokenizer, prompts, max_tokens, concurrent):
    peak, stop = [gpu_mib()], threading.Event()

    def poll():
        while not stop.is_set():
            peak.append(gpu_mib())
            time.sleep(0.5)
    watcher = threading.Thread(target=poll)
    watcher.start()
    before = server.stats()
    calls = [lambda p=p: timed_stream(server, p, max_tokens) for p in prompts]
    results = server.parallel(calls) if concurrent else [call() for call in calls]
    after = server.wait_idle()
    stop.set()
    watcher.join()
    rates = []
    for r in results:
        n = len(tokenizer(r["text"])["input_ids"])
        if n > 1 and r["last"] > r["first"]:
            rates.append((n - 1) / (r["last"] - r["first"]))
    delta = spec_delta(after, before)
    graph = {p: after["cuda_graph"].get(p, 0) - before["cuda_graph"].get(p, 0) for p in ("draft", "verify",
                                                                                          "target_decode")}
    return {"decode_tok_s": statistics.mean(rates), "acceptance": acceptance(delta), "delta": delta,
            "texts": [r["text"] for r in results],
            "graph_replays": graph, "peak_gpu_mib": max(peak)}


def perf_run(server, c):
    tokenizer = AutoTokenizer.from_pretrained(server.flag("--model-path"))
    hybrid = server.flag("--moe-backend") == "hybrid"
    long_prompt = lambda seed: document(seed, 220) + " Note"  # noqa: E731
    wave(server, tokenizer, [long_prompt(1), CODE_HEADER + CODE_TASKS[0][0]], 64, True)  # warm-up, excluded
    plan = {"c4_long": (3, lambda r: [long_prompt(300 + 4 * r + k) for k in range(4)], 256, True)}
    if not hybrid:
        plan["c1_short"] = (3, lambda r: [CODE_HEADER + t[0] for t in CODE_TASKS], 192, False)
        plan["c16_long"] = (2, lambda r: [long_prompt(400 + 16 * r + k) for k in range(16)], 128, True)
    out = {}
    for name, (reps, prompts, max_tokens, concurrent) in plan.items():
        out[name] = [wave(server, tokenizer, prompts(r), max_tokens, concurrent) for r in range(reps)]
        eager = [w["delta"]["verify_steps"] - w["graph_replays"]["verify"] for w in out[name]]
        c.note(f"perf:{name}", decode_tok_s=[round(w["decode_tok_s"], 2) for w in out[name]],
               acceptance=[round(w["acceptance"], 3) for w in out[name]],
               peak_gpu_mib=max(w["peak_gpu_mib"] for w in out[name]), verify_rounds_not_on_graph=eager)
        c.check(f"perf_graph_used:{name}", all(w["graph_replays"]["verify"] > 0 for w in out[name]),
                replays=[w["graph_replays"] for w in out[name]])
    out["status"] = server.status()["geometry"]["dflash"]
    return out


PLAN = {name: [perf_run] for name in CONFIGS}


def compact_vs_full(sessions, c):
    for suffix in ("", "_hybrid"):
        a = sessions.get(f"perf_compact{suffix}", {}).get("artifacts", {}).get("perf_run")
        b = sessions.get(f"perf_full{suffix}", {}).get("artifacts", {}).get("perf_run")
        if not (a and b):
            continue
        for name in a:
            if name == "status" or name not in b:
                continue
            x = [w["decode_tok_s"] for w in a[name]]
            y = [w["decode_tok_s"] for w in b[name]]
            mx, my = statistics.median(x), statistics.median(y)
            spread = max(max(x) - min(x), max(y) - min(y))
            detail = dict(compact=x, full=y, ratio=mx / my, spread=spread,
                          acceptance=(statistics.median(w["acceptance"] for w in a[name]),
                                      statistics.median(w["acceptance"] for w in b[name])),
                          peak_gpu_mib=(max(w["peak_gpu_mib"] for w in a[name]),
                                        max(w["peak_gpu_mib"] for w in b[name])))
            if mx >= GATE * my:
                c.check(f"compact_decode_gate{suffix}:{name}", True, **detail)
            elif my - mx > spread:
                c.check(f"compact_decode_gate{suffix}:{name}", False, **detail)
            else:
                c.note(f"compact_decode_gate_inconclusive{suffix}:{name}", **detail)
        if "c1_short" in a and "c1_short" in b:  # single sequential requests: same scheduling in both arms
            same = [x == y for wa, wb in zip(a["c1_short"], b["c1_short"]) for x, y in zip(wa["texts"], wb["texts"])]
            c.check(f"compact_full_token_identical{suffix}:c1_short", all(same), identical=same)
        c.check(f"compact_reports_less_context{suffix}", a["status"]["context_bytes"] < b["status"]["context_bytes"],
                compact=a["status"]["context_bytes"], full=b["status"]["context_bytes"])


COMPARE = [compact_vs_full]

"""Public HTTP checks; the coordinator owns server startup and GPU allocation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def connect(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    return urlopen(Request(url + path, data=data,
                           headers={"Content-Type": "application/json"}), timeout=600)


def api(url, path, body=None):
    with connect(url, path, body) as response:
        return json.load(response)


def validate(response, limit, chat=False):
    require(len(response["choices"]) == 1, "expected one independent completion")
    choice = response["choices"][0]
    text = choice["message"]["content"] if chat else choice["text"]
    require(isinstance(text, str), "completion text must be a string")
    require(choice["finish_reason"] in ("length", "stop"), "unexpected finish reason")
    usage = response["usage"]
    count = usage["completion_tokens"]
    require(isinstance(count, int) and 0 <= count <= limit, "output token limit exceeded")
    require(usage["prompt_tokens"] > 0, "missing prompt token accounting")
    require(usage["total_tokens"] == count + usage["prompt_tokens"], "inconsistent usage")
    if choice["finish_reason"] == "length":
        require(count == limit, "length termination before the output limit")
    return {"text": text, "finish_reason": choice["finish_reason"],
            "completion_tokens": count, "prompt_tokens": usage["prompt_tokens"]}


def complete(url, body, chat=False):
    path = "/v1/chat/completions" if chat else "/v1/completions"
    return validate(api(url, path, body), body["max_tokens"], chat)


def stream(url, body, abort=False):
    pieces, finish, done, usage = [], None, False, None
    request = dict(body, stream=True, stream_options={"include_usage": True})
    with connect(url, "/v1/completions", request) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            require("error" not in event, f"stream error: {event}")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                pieces.append(choice.get("text", ""))
                finish = choice.get("finish_reason") or finish
            if abort and any(pieces):
                require(finish is None, "cancellation fixture already finished")
                return
    require(done and finish is not None, "stream did not terminate cleanly")
    require(not abort, "cancellation fixture emitted no text before completion")
    require(isinstance(usage, dict), "include_usage stream omitted committed token usage")
    return validate({"choices": [{"text": "".join(pieces), "finish_reason": finish}],
                     "usage": usage}, body["max_tokens"])


def run(args):
    url = args.url.rstrip("/")
    before = api(url, "/v1/stats")["speculative"]
    ordinary = args.scenario == "ordinary-reference"
    require(before["draft_residency"] == args.mode, "server residency mode differs")
    require(before["enabled"] == (not ordinary), "unexpected SD enabled setting")
    cache_slots = api(url, "/v1/cache/status")["geometry"]["moe_cache_size"]
    model = api(url, "/v1/models")["data"][0]["id"]
    common = {"model": model, "temperature": 0, "top_k": -1, "top_p": 1.0}
    cases = [
        dict(common, prompt="The sequence is: 2, 4, 6, 8,", max_tokens=8),
        dict(common, prompt="A short sentence about the ocean:\n", max_tokens=1),
        dict(common, prompt="Translate 'good morning' into French:\n", max_tokens=2),
        dict(common, prompt="The capital of Japan is", max_tokens=4),
    ]
    greedy = [complete(url, case) for case in cases]
    if args.scenario not in ("baseline", "ordinary-reference"):
        reference = json.loads(args.reference.read_text())
        require(reference["model"] == model, "reference model differs")
        require(reference["cases"] == cases, "reference input contract differs")
        if args.scenario == "shortage":
            require(reference["scenario"] == "ordinary-reference",
                    "shortage needs an independent ordinary target reference")
            require(reference["cache_slots"] == cache_slots,
                    "shortage and ordinary reference cache budgets differ")
        require(greedy == reference["greedy"],
                f"greedy target differs: expected {reference['greedy']}, got {greedy}")

    sampled = [dict(case, temperature=0.7 + index * 0.1, top_k=20, top_p=0.9,
                    max_tokens=limit)
               for index, (case, limit) in enumerate(zip(cases, (1, 2, 3, 5)))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda body: complete(url, body), sampled))

    lifecycle = {}
    if args.scenario in ("baseline", "active"):
        streamed = stream(url, cases[0])
        require(streamed == greedy[0], f"stream differs: expected {greedy[0]}, got {streamed}")
        require(bool(greedy[0]["text"]), "stop fixture needs nonempty output")
        stop = greedy[0]["text"][:max(1, len(greedy[0]["text"]) // 2)]
        stopped = complete(url, dict(cases[0], stop=[stop]))
        require(stopped["text"] == "" and stopped["finish_reason"] == "stop",
                f"stop prefix escaped into output: {stopped}")
        complete(url, dict(common, temperature=0.8, top_k=20, top_p=0.9,
                           messages=[{"role": "user", "content": "Say hello briefly."}],
                           max_tokens=2), chat=True)
        stream(url, dict(cases[0], max_tokens=256), abort=True)
        recovered = complete(url, cases[0])
        require(recovered == greedy[0], f"after cancellation expected {greedy[0]}, got {recovered}")
        lifecycle = {"stream": streamed, "stop": stopped, "after_cancellation": recovered}

    after = api(url, "/v1/stats")["speculative"]
    keys = ("draft_tokens", "accepted_draft_tokens", "verify_steps", "residency_stops")
    delta = {key: after[key] - before[key] for key in keys}
    require(all(value >= 0 for value in delta.values()), "cumulative counters regressed")
    require(delta["accepted_draft_tokens"] <= delta["draft_tokens"],
            "accepted drafts exceed proposed drafts")
    require(after["draft_residency"] == args.mode, "residency mode changed")
    if args.collect_stats == "off":
        require(before["draft_expert_loads"] is None and after["draft_expert_loads"] is None,
                "draft_expert_loads must be null")
    else:
        require(type(before["draft_expert_loads"]) is int and type(after["draft_expert_loads"]) is int,
                "draft_expert_loads must be a collected count")
        delta["draft_expert_loads"] = after["draft_expert_loads"] - before["draft_expert_loads"]
        require(delta["draft_expert_loads"] >= 0, "draft_expert_loads regressed")
    if ordinary:
        require(all(after[key] == 0 for key in keys + ("draft_expert_loads",)),
                "ordinary service has SD activity")
    elif args.scenario == "shortage":
        require(delta["residency_stops"] > 0, "insufficient-cache fallback was not exercised")
        require(delta["draft_tokens"] == 0 and delta["verify_steps"] == 0,
                "insufficient cache still executed drafting or verification")
    else:
        require(delta["draft_tokens"] > 0 and delta["verify_steps"] > 0,
                "coverage failure: workload never drafted and verified")
        if args.mode == "off":
            require(delta["residency_stops"] == 0, "residency off refused drafting")
    if args.collect_stats == "on":
        if args.mode == "off" and not ordinary:
            require(delta["draft_expert_loads"] > 0,
                    "coverage failure: off control never loaded a draft expert")
        else:
            require(delta["draft_expert_loads"] == 0, "resident-only draft loaded an expert")
    report = {"model": model, "mode": args.mode, "scenario": args.scenario,
              "cases": cases, "greedy": greedy, "sampled": concurrent, "delta": delta,
              "cache_slots": cache_slots, "lifecycle": lifecycle}
    if args.scenario in ("baseline", "ordinary-reference"):
        args.reference.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--scenario", required=True,
                        choices=("baseline", "ordinary-reference", "active", "shortage"))
    parser.add_argument("--mode", choices=("off", "router"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--collect-stats", choices=("on", "off"), default="on")
    args = parser.parse_args()
    if args.scenario in ("baseline", "ordinary-reference") and args.mode != "off":
        parser.error("reference capture requires mode=off")
    if args.scenario == "ordinary-reference" and args.collect_stats == "off":
        parser.error("ordinary reference requires statistics on")
    if args.scenario == "shortage" and args.mode == "off":
        parser.error("shortage requires router")
    started = time.monotonic()
    result = run(args)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))

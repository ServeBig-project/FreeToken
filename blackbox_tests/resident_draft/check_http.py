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
    pieces, finish, done = [], None, False
    with connect(url, "/v1/completions", dict(body, stream=True)) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            require("error" not in event, f"stream error: {event}")
            for choice in event.get("choices", []):
                pieces.append(choice.get("text", ""))
                finish = choice.get("finish_reason") or finish
            if abort and any(pieces):
                require(finish is None, "cancellation fixture already finished")
                return
    require(done and finish is not None, "stream did not terminate cleanly")
    require(not abort, "cancellation fixture emitted no text before completion")
    return {"text": "".join(pieces), "finish_reason": finish}


def run(args):
    url = args.url.rstrip("/")
    before = api(url, "/v1/stats")["speculative"]
    require(before["draft_residency"] == args.mode, "server residency mode differs")
    require(before["enabled"], "this matrix requires SD to be enabled")
    require(before["adaptive_enabled"] == args.adaptive, "unexpected adaptive setting")
    require(before["reuse_enabled"] == args.reuse, "unexpected reuse setting")
    model = api(url, "/v1/models")["data"][0]["id"]
    common = {"model": model, "temperature": 0, "top_k": -1, "top_p": 1.0}
    cases = [
        dict(common, prompt="The sequence is: 2, 4, 6, 8,", max_tokens=8),
        dict(common, prompt="A short sentence about the ocean:\n", max_tokens=1),
        dict(common, prompt="Translate 'good morning' into French:\n", max_tokens=2),
        dict(common, prompt="The capital of Japan is", max_tokens=4),
    ]
    greedy = [complete(url, case) for case in cases]
    if args.scenario != "baseline" and not args.reuse:
        reference = json.loads(args.reference.read_text())
        require(reference["model"] == model, "reference model differs")
        require(reference["cases"] == cases, "reference input contract differs")
        require(greedy == reference["greedy"],
                f"greedy target differs: expected {reference['greedy']}, got {greedy}")

    sampled = [dict(case, temperature=0.7 + index * 0.1, top_k=20, top_p=0.9,
                    max_tokens=limit)
               for index, (case, limit) in enumerate(zip(cases, (1, 2, 3, 5)))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda body: complete(url, body), sampled))

    if args.scenario != "shortage":
        streamed = stream(url, cases[0])
        require(streamed == {key: greedy[0][key] for key in streamed},
                "stream emitted different committed text or termination")
        require(bool(greedy[0]["text"]), "stop fixture needs nonempty output")
        stop = greedy[0]["text"][:max(1, len(greedy[0]["text"]) // 2)]
        stopped = complete(url, dict(cases[0], stop=[stop]))
        require(stopped["text"] == "" and stopped["finish_reason"] == "stop",
                f"stop prefix escaped into output: {stopped}")
        complete(url, dict(common, temperature=0.8, top_k=20, top_p=0.9,
                           messages=[{"role": "user", "content": "Say hello briefly."}],
                           max_tokens=2), chat=True)
        stream(url, dict(cases[0], max_tokens=256), abort=True)
        require(complete(url, cases[0]) == greedy[0],
                "a request after cancellation changed its greedy result")

    after = api(url, "/v1/stats")["speculative"]
    keys = ("draft_tokens", "accepted_draft_tokens", "verify_steps", "residency_stops")
    delta = {key: after[key] - before[key] for key in keys}
    require(all(value >= 0 for value in delta.values()), "cumulative counters regressed")
    require(delta["accepted_draft_tokens"] <= delta["draft_tokens"],
            "accepted drafts exceed proposed drafts")
    require(after["draft_residency"] == args.mode, "residency mode changed")
    for key in ("draft_expert_loads", "draft_expert_replacements"):
        if args.collect_stats == "off":
            require(before[key] is None and after[key] is None, f"{key} must be null")
        else:
            require(type(before[key]) is int and type(after[key]) is int,
                    f"{key} must be a collected count")
            delta[key] = after[key] - before[key]
            require(delta[key] >= 0, f"{key} regressed")
    if args.scenario == "shortage":
        require(delta["residency_stops"] > 0, "insufficient-cache fallback was not exercised")
        require(delta["draft_tokens"] == 0 and delta["verify_steps"] == 0,
                "insufficient cache still executed drafting or verification")
    else:
        require(delta["draft_tokens"] > 0 and delta["verify_steps"] > 0,
                "coverage failure: workload never drafted and verified")
        require(delta["residency_stops"] == 0, "pinned expert fixture unexpectedly fell back")
    if args.collect_stats == "on":
        if args.mode == "off":
            require(delta["draft_expert_loads"] > 0,
                    "coverage failure: off control never loaded a draft expert")
        else:
            require(delta["draft_expert_loads"] == 0, "resident-only draft loaded an expert")
        replacement_expected = args.mode == "affinity" and args.scenario == "active"
        require((delta["draft_expert_replacements"] > 0) if replacement_expected
                else delta["draft_expert_replacements"] == 0,
                "affinity replacement coverage/count differs from the scenario")
    report = {"model": model, "mode": args.mode, "scenario": args.scenario,
              "cases": cases, "greedy": greedy, "sampled": concurrent, "delta": delta}
    if args.scenario == "baseline":
        args.reference.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--scenario", choices=("baseline", "active", "shortage"), required=True)
    parser.add_argument("--mode", choices=("off", "router", "affinity"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--collect-stats", choices=("on", "off"), default="on")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.scenario == "baseline" and (args.mode != "off" or args.reuse):
        parser.error("baseline requires mode=off without reuse")
    if args.scenario == "shortage" and args.mode == "off":
        parser.error("shortage requires router or affinity")
    started = time.monotonic()
    result = run(args)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))

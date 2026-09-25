#!/usr/bin/env python3
"""Independent acceptance checks against the published HTTP interface."""

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def check(run, name, condition, detail=None, status="failed"):
    run["checks"].append({"name": name, "status": "passed" if condition else status,
                          "detail": detail})
    if not condition and status == "failed":
        raise AssertionError(name)


def request(run, path, body=None, expected=200):
    started = time.monotonic()
    event = {"path": path, "input": body}
    run["http"].append(event)
    req = urllib.request.Request(run["url"] + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=run["timeout"])
    except urllib.error.HTTPError as error:
        response = error
    with response:
        event["status"] = response.status
        event["raw"] = response.read().decode("utf-8")
    event["seconds"] = time.monotonic() - started
    check(run, f"{path}: HTTP {expected}", event["status"] == expected,
          {"actual": event["status"]})
    if body and body.get("stream"):
        return event["raw"]
    event["output"] = json.loads(event["raw"])
    return event["output"]


def idle(run):
    deadline = time.monotonic() + min(run["timeout"], 30)
    while True:
        stats = request(run, "/v1/stats")
        if stats["requests"]["active"] == 0:
            return stats
        check(run, "requests eventually become idle", time.monotonic() < deadline,
              stats["requests"])
        time.sleep(0.25)


def usage_check(run, name, usage, count):
    check(run, f"{name}: final output token count", usage["completion_tokens"] == count,
          usage)
    check(run, f"{name}: positive prompt usage", usage["prompt_tokens"] > 0, usage)
    check(run, f"{name}: total usage", usage["total_tokens"] ==
          usage["prompt_tokens"] + usage["completion_tokens"], usage)


def complete(run, name, prompt, group, count=48, stream=False, chat=False):
    body = {"model": run["model"], "max_tokens": count, "temperature": 0,
            "top_k": 1, "top_p": 1, "ignore_eos": True, "cache_group": group,
            "stream": stream}
    if chat:
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    if stream:
        body["stream_options"] = {"include_usage": True}
    output = request(run, "/v1/chat/completions" if chat else "/v1/completions", body)
    if stream:
        packets = [line[5:].strip() for line in output.splitlines()
                   if line.startswith("data:")]
        check(run, f"{name}: SSE terminates with DONE", bool(packets) and
              packets[-1] == "[DONE]")
        chunks = [json.loads(packet) for packet in packets[:-1]]
        text, reasons, usages = "", [], []
        for chunk in chunks:
            if chunk.get("usage") is not None:
                usages.append(chunk["usage"])
            for choice in chunk.get("choices", []):
                text += (choice.get("delta", {}).get("content") or "") if chat else (
                    choice.get("text") or "")
                if choice.get("finish_reason") is not None:
                    reasons.append(choice["finish_reason"])
        check(run, f"{name}: streaming usage present", bool(usages))
        usage = usages[-1]
        check(run, f"{name}: streaming finish reason", reasons == ["length"], reasons)
    else:
        check(run, f"{name}: one completion", len(output["choices"]) == 1)
        choice = output["choices"][0]
        text = choice["message"]["content"] if chat else choice["text"]
        check(run, f"{name}: length finish reason", choice["finish_reason"] == "length",
              choice["finish_reason"])
        usage = output["usage"]
    usage_check(run, name, usage, count)
    check(run, f"{name}: nonempty text", isinstance(text, str) and bool(text))
    run["samples"][name] = {"input": body, "text": text, "usage": usage}
    return text


def compare(run, name, left, right):
    check(run, name, left == right, {"left": left, "right": right},
          status="investigate")


def core(run, args):
    health = request(run, "/health")
    check(run, "health ready", health["status"] == "ok", health)
    models = request(run, "/v1/models")["data"]
    check(run, "served model available", bool(models))
    run["model"] = args.model or models[0]["id"]
    check(run, "requested model is served", run["model"] in [m["id"] for m in models])
    run["before"] = idle(run)
    run["geometry_before"] = request(run, "/v1/cache/status")["geometry"]
    sd = run["before"]["speculative"]
    check(run, "effective SD setting", sd["enabled"] == (args.expected_steps > 0), sd)
    check(run, "effective maximum draft length", sd["max_draft_steps"] == args.expected_steps,
          sd["max_draft_steps"])
    check(run, "effective Graph setting", run["before"]["cuda_graph"]["enabled"] ==
          (args.mode == "graph"), run["before"]["cuda_graph"])
    prompt = ("The Moon reflects sunlight. Its visible illuminated portion changes as it "
              "orbits Earth. Explain the phases of the Moon in plain language.\nAnswer:")
    first = complete(run, "prefix_first", prompt, args.group_a)
    repeated = complete(run, "prefix_repeat", prompt, args.group_a)
    isolated = complete(run, "prefix_other_group", prompt, args.group_b)
    streamed = complete(run, "completion_stream", prompt, args.group_a, stream=True)
    compare(run, "same-group output comparison", first, repeated)
    compare(run, "different-group output comparison", first, isolated)
    compare(run, "stream output comparison", first, streamed)
    chat = complete(run, "chat", "Explain why a day has daylight and darkness.", args.group_b,
                    chat=True)
    chat_stream = complete(run, "chat_stream", "Explain why a day has daylight and darkness.",
                           args.group_b, stream=True, chat=True)
    compare(run, "chat stream output comparison", chat, chat_stream)
    run["after_core"] = idle(run)
    geometry = request(run, "/v1/cache/status")["geometry"]
    for field in ("num_pages", "page_size", "moe_cache_size", "num_mamba_slots",
                  "cache_budget_bytes"):
        check(run, f"generation preserves {field}", geometry[field] ==
              run["geometry_before"][field], {"before": run["geometry_before"][field],
                                             "after": geometry[field]})
    before, after = run["before"], run["after_core"]
    check(run, "completed counter advances", after["requests"]["completed"] >
          before["requests"]["completed"])
    if args.expected_steps:
        for field in ("draft_tokens", "verify_steps"):
            check(run, f"actual SD: {field} advances", after["speculative"][field] >
                  before["speculative"][field], {"before": before["speculative"][field],
                                                "after": after["speculative"][field]},
                  status="uncovered")
        if args.mode == "graph":
            for field in ("draft", "verify"):
                check(run, f"actual Graph: {field} replay advances", after["cuda_graph"][field]
                      > before["cuda_graph"][field], {"before": before["cuda_graph"][field],
                                                       "after": after["cuda_graph"][field]},
                      status="uncovered")


def resources(run, args):
    idle(run)
    original = request(run, "/v1/cache/status")["geometry"]
    fields = ("moe_cache_size", "num_pages", "num_mamba_slots")
    restore = {field: original[field] for field in fields}
    request(run, "/v1/cache/rebuild", {"mode": "unsupported"}, expected=422)
    complete(run, "after_invalid_rebuild", "Count from one to ten in words.", args.group_a,
             count=16)
    idle(run)
    changed = False
    try:
        body = {"mode": "if_idle", "timeout": args.timeout}
        if args.state_slots is not None:
            limits = original["limits"]["mamba_slots"]
            check(run, "requested state capacity within public bounds",
                  args.state_slots >= limits["min"] and
                  (limits["max"] == 0 or args.state_slots <= limits["max"]), limits)
            body["num_mamba_slots"] = args.state_slots
        changed = True
        result = request(run, "/v1/cache/rebuild", body)
        check(run, "cache rebuild succeeds", result["status"] == "ok", result)
        geometry = request(run, "/v1/cache/status")["geometry"]
        for field in fields:
            expected = body.get(field, original[field])
            check(run, f"rebuild keeps requested {field}", geometry[field] == expected,
                  {"expected": expected, "actual": geometry[field]})
        before = idle(run)
        complete(run, "after_rebuild", "Explain why ice floats on water.\nAnswer:",
                 args.group_a, count=64)
        after = idle(run)
        current = request(run, "/v1/cache/status")["geometry"]
        for field in (*fields, "cache_budget_bytes"):
            check(run, f"limited generation preserves {field}", current[field] == geometry[field],
                  {"before": geometry[field], "after": current[field]})
        run["resource_observation"] = {"geometry": geometry, "before": before, "after": after}
        if args.state_slots is not None and args.expected_steps > 0:
            for field in ("draft_tokens", "verify_steps"):
                check(run, f"rebuilt pool actually executes SD: {field}",
                      after["speculative"][field] > before["speculative"][field],
                      status="uncovered")
            if args.mode == "graph":
                for field in ("draft", "verify"):
                    check(run, f"rebuilt pool actually executes Graph: {field}",
                          after["cuda_graph"][field] > before["cuda_graph"][field],
                          status="uncovered")
        if (args.state_slots is not None and args.state_slots < original["num_mamba_slots"]
                and args.expected_steps > 1):
            left = before["speculative"]["draft_length_histogram"]
            right = after["speculative"]["draft_length_histogram"]
            delta = [right[i] - left[i] for i in range(args.expected_steps + 1)]
            run["resource_observation"]["draft_length_delta"] = delta
            check(run, "smaller pool actually exercises shortened SD", any(delta[1:-1]),
                  delta, status="uncovered")
    finally:
        if changed:
            idle(run)
            result = request(run, "/v1/cache/rebuild",
                             dict(restore, mode="if_idle", timeout=args.timeout))
            check(run, "original cache geometry restored", result["status"] == "ok", result)
            geometry = request(run, "/v1/cache/status")["geometry"]
            for field in fields:
                check(run, f"restored {field}", geometry[field] == original[field])


def reference(run, path):
    previous = json.loads(Path(path).read_text())
    for name, sample in run["samples"].items():
        if name not in previous["samples"]:
            continue
        old = previous["samples"][name]
        check(run, f"reference {name}: identical public request", sample["input"] == old["input"],
              {"current": sample["input"], "reference": old["input"]})
        compare(run, f"reference {name}: text comparison", old["text"], sample["text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, help="Full JSON evidence path")
    parser.add_argument("--model", help="Default: first id from /v1/models")
    parser.add_argument("--mode", choices=("graph", "eager"), required=True)
    parser.add_argument("--expected-steps", type=int, choices=range(9), required=True)
    parser.add_argument("--group-a", default="sd-portability-a")
    parser.add_argument("--group-b", default="sd-portability-b")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--resources", action="store_true", help="Exercise public cache rebuild")
    parser.add_argument("--state-slots", type=int, help="State capacity within the public limits")
    parser.add_argument("--reference", help="Earlier report from the same public requests")
    args = parser.parse_args()
    if args.state_slots is not None and not args.resources:
        parser.error("--state-slots requires --resources")
    run = {"url": args.url.rstrip("/"), "timeout": args.timeout, "arguments": vars(args),
           "checks": [], "http": [], "samples": {}, "reference_only": args.expected_steps == 0}
    started = time.monotonic()
    try:
        core(run, args)
        if args.resources:
            resources(run, args)
        if args.reference:
            reference(run, args.reference)
    except Exception as error:
        run["checks"].append({"name": "run completed", "status": "failed",
                              "detail": f"{type(error).__name__}: {error}"})
    run["seconds"] = time.monotonic() - started
    counts = {status: sum(item["status"] == status for item in run["checks"])
              for status in ("passed", "failed", "investigate", "uncovered")}
    run["summary"] = counts
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), "seconds": run["seconds"], **counts}))
    return 1 if counts["failed"] else 2 if counts["investigate"] else 3 if counts["uncovered"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""HTTP evidence and assertions for the independent acceptance client."""

import json
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


def usage_check(run, name, usage, count, exact):
    check(run, f"{name}: final output token count", usage["completion_tokens"] == count if exact
          else 0 <= usage["completion_tokens"] <= count,
          usage)
    check(run, f"{name}: positive prompt usage", usage["prompt_tokens"] > 0, usage)
    check(run, f"{name}: total usage", usage["total_tokens"] ==
          usage["prompt_tokens"] + usage["completion_tokens"], usage)


def complete(run, name, prompt, group, count=48, stream=False, chat=False,
             ignore_eos=True, stop=None):
    body = {"model": run["model"], "max_tokens": count, "temperature": 0,
            "top_k": 1, "top_p": 1, "ignore_eos": ignore_eos, "cache_group": group,
            "stream": stream}
    if chat:
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    if stream:
        body["stream_options"] = {"include_usage": True}
    if stop is not None:
        body["stop"] = stop
    exact = ignore_eos and stop is None
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
        check(run, f"{name}: one streaming finish reason", len(reasons) == 1, reasons)
        reason = reasons[0]
    else:
        check(run, f"{name}: one completion", len(output["choices"]) == 1)
        choice = output["choices"][0]
        text = choice["message"]["content"] if chat else choice["text"]
        reason = choice["finish_reason"]
        usage = output["usage"]
    check(run, f"{name}: finish reason", reason == "length" if exact else
          reason in ("length", "stop"), reason)
    usage_check(run, name, usage, count, exact)
    check(run, f"{name}: text output", isinstance(text, str) and (bool(text) or stop is not None))
    run["samples"][name] = {"input": body, "text": text, "usage": usage,
                            "finish_reason": reason}
    return text


def compare(run, name, left, right):
    check(run, name, left == right, {"left": left, "right": right},
          status="investigate")


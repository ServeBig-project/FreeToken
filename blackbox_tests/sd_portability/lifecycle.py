"""Stop/cancellation reuse and mixed concurrency through public requests only."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

from http_client import check, compare, complete, idle, request


def stop_and_eos(run, args):
    original = run["samples"]["prefix_first"]
    prompt = original["input"]["prompt"]
    marker = original["text"][:24]
    for name, stop in (("stop_string", marker), ("stop_array", [marker, "END_OF_ANSWER"])):
        text = complete(run, name, prompt, args.group_a, stop=stop)
        check(run, f"{name}: stop string excluded", marker not in text, text)
        check(run, f"{name}: stopping actually observed",
              run["samples"][name]["finish_reason"] == "stop", status="uncovered")
        after = complete(run, f"after_{name}", prompt, args.group_a)
        compare(run, f"prefix reuse after {name}", original["text"], after)
    eos_prompt = "Reply with exactly the word Done and nothing else."
    off = run["geometry_before"]["reasoning"]["kwargs"]["off"]
    first = complete(run, "eos", eos_prompt, args.group_b, count=160,
                     chat=True, ignore_eos=False, chat_kwargs=off)
    check(run, "EOS actually observed", run["samples"]["eos"]["finish_reason"] == "stop",
          status="uncovered")
    repeated = complete(run, "eos_prefix_repeat", eos_prompt, args.group_b, count=160,
                        chat=True, ignore_eos=False, chat_kwargs=off)
    compare(run, "prefix reuse after EOS", first, repeated)


def cancel(run, args):
    original = run["samples"]["prefix_first"]
    body = dict(original["input"], max_tokens=512, stream=True,
                stream_options={"include_usage": True})
    event = {"path": "/v1/completions", "input": body, "raw": "", "cancelled": False}
    run["http"].append(event)
    started = time.monotonic()
    req = urllib.request.Request(run["url"] + event["path"],
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=run["timeout"]) as response:
        event["status"] = response.status
        check(run, "cancellation stream opens", response.status == 200)
        for raw in response:
            line = raw.decode("utf-8")
            event["raw"] += line
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            packet = json.loads(data)
            if any(choice.get("text") for choice in packet.get("choices", [])):
                stats = request(run, "/v1/stats")
                event["active_at_close"] = stats["requests"]["active"]
                event["cancelled"] = True
                break
    event["seconds"] = time.monotonic() - started
    check(run, "client closed an active generation", event["cancelled"] and
          event.get("active_at_close", 0) > 0, event, status="uncovered")
    run["after_cancel"] = idle(run)
    after = complete(run, "prefix_after_cancel", original["input"]["prompt"], args.group_a)
    compare(run, "prefix reuse after cancellation", original["text"], after)
    complete(run, "other_group_after_cancel", "Name the four seasons.\nAnswer:", args.group_b,
             count=16)


def shapes_delta(before, after):
    fields = ("phase", "batch_size", "query_tokens", "physical_query_tokens")
    old = {tuple(item[field] for field in fields): item["replays"]
           for item in before["cuda_graph"]["replay_shapes"]}
    return [dict(item, delta=item["replays"] - old.get(tuple(item[field] for field in fields), 0))
            for item in after["cuda_graph"]["replay_shapes"]]


def concurrency(run, args):
    before = idle(run)
    peak = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(complete, run, f"concurrent_{index}",
                            f"Explain the role of water in plant growth. Example {index}:\n",
                            args.group_a, count=count)
                for index, count in enumerate((8, 21, 48, 65))]
        while not all(job.done() for job in jobs):
            peak = max(peak, request(run, "/v1/stats")["requests"]["active"])
            time.sleep(0.1)
        for job in jobs:
            job.result()
    after = idle(run)
    run["concurrency"] = {"active_peak": peak, "before": before, "after": after}
    check(run, "four active requests observed", peak >= 4, peak, status="uncovered")
    if args.mode == "graph":
        delta = shapes_delta(before, after)
        run["concurrency"]["replay_shape_delta"] = delta
        for phase in ("draft", "verify") if args.expected_steps else ("target_decode",):
            live = [item for item in delta if item["phase"] == phase and item["delta"] > 0]
            check(run, f"{phase}: batch four Graph observed",
                  any(item["batch_size"] == 4 for item in live), live, status="uncovered")
            check(run, f"{phase}: uneven Graph tail observed",
                  any(item["batch_size"] in (2, 3) for item in live), live, status="uncovered")


def token_input(run, args):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    prompt = "Explain why the sky appears blue.\nAnswer:"
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    first = complete(run, "token_prompt", tokens, args.group_a, count=24)
    repeated = complete(run, "token_prompt_repeat", tokens, args.group_a, count=24)
    compare(run, "token input prefix reuse", first, repeated)


def lifecycle(run, args):
    stop_and_eos(run, args)
    cancel(run, args)
    concurrency(run, args)
    if args.tokenizer:
        token_input(run, args)

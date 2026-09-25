"""Require public cache-hit evidence beyond the original prompt after termination."""

import time

from http_client import cancel_stream, check, compare, complete, idle


def source_path(run, args, name, before):
    after = idle(run)
    run.setdefault("termination_paths", {})[name] = {"before": before, "after": after}
    if args.expected_steps:
        for field in ("draft_tokens", "verify_steps"):
            check(run, f"{name}: actual SD {field}", after["speculative"][field] >
                  before["speculative"][field], status="uncovered")
        if args.mode == "graph":
            for field in ("draft", "verify"):
                check(run, f"{name}: actual Graph {field}", after["cuda_graph"][field] >
                      before["cuda_graph"][field], status="uncovered")


def followup(run, args, name, prompt, group, original_tokens, chat=False, kwargs=None):
    left = complete(run, name + "_reuse", prompt, group, count=24, chat=chat, chat_kwargs=kwargs)
    right = complete(run, name + "_control", prompt, group + "-control", count=24,
                     chat=chat, chat_kwargs=kwargs)
    compare(run, name + ": generated-prefix output comparison", left, right)
    usage = run["samples"][name + "_reuse"]["usage"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    check(run, name + ": hit includes previously generated tokens",
          cached is not None and cached > original_tokens,
          {"original_prompt_tokens": original_tokens, "cached_tokens": cached},
          status="uncovered")
    control = run["samples"][name + "_control"]["usage"].get("prompt_tokens_details", {})
    if "cached_tokens" in control:
        check(run, name + ": fresh group has no prefix hit", control["cached_tokens"] == 0, control)
    else:
        check(run, name + ": cache-group isolation is observable", False, status="uncovered")


def generated_prefix(run, args):
    before = idle(run)
    group = args.group_a + "-retained-" + str(time.time_ns())
    prompt = "Describe how the Moon's visible shape changes throughout a month.\nAnswer:"
    seed = complete(run, "retained_seed", prompt, group + "-seed", count=64)
    prompt_tokens = run["samples"]["retained_seed"]["usage"]["prompt_tokens"]
    marker = seed[len(seed) // 3:len(seed) // 3 + 24]
    for name, stop in (("retained_stop_string", marker),
                       ("retained_stop_array", [marker, "END_OF_ANSWER"])):
        source_group = group + "-" + name
        source_before = idle(run)
        text = complete(run, name, prompt, source_group, count=64, stop=stop)
        source_path(run, args, name, source_before)
        source = run["samples"][name]
        check(run, name + ": actual stop retains output", bool(text) and
              source["finish_reason"] == "stop", source, status="uncovered")
        check(run, name + ": stop marker excluded", marker not in text, text)
        if text:
            followup(run, args, name, prompt + text, source_group, prompt_tokens)

    off = run["geometry_before"]["reasoning"]["kwargs"]["off"]
    eos_prompt = "Reply with exactly the word Done and nothing else."
    source_before = idle(run)
    text = complete(run, "retained_eos", eos_prompt, group + "-eos", count=160,
                    chat=True, ignore_eos=False, chat_kwargs=off)
    source_path(run, args, "retained_eos", source_before)
    source = run["samples"]["retained_eos"]
    check(run, "retained EOS actually occurred", source["finish_reason"] == "stop",
          status="uncovered")
    messages = [{"role": "user", "content": eos_prompt}, {"role": "assistant", "content": text},
                {"role": "user", "content": "Repeat the same word."}]
    followup(run, args, "retained_eos", messages, group + "-eos",
             source["usage"]["prompt_tokens"], chat=True, kwargs=off)

    body = dict(run["samples"]["retained_seed"]["input"], max_tokens=512, stream=True,
                cache_group=group + "-cancel", stream_options={"include_usage": True})
    source_before = idle(run)
    event = cancel_stream(run, body, min_chars=48)
    check(run, "retained cancellation closes active generation", event["cancelled"] and
          event.get("active_at_close", 0) > 0, event, status="uncovered")
    source_path(run, args, "retained_cancel", source_before)
    if event["retained_text"]:
        followup(run, args, "retained_cancel", prompt + event["retained_text"],
                 group + "-cancel", prompt_tokens)
    after = idle(run)
    run["generated_prefix_observation"] = {"before": before, "after": after}

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


def followup(run, name, prompt, group, original_tokens):
    left = complete(run, name + "_reuse", prompt, group, count=24)
    right = complete(run, name + "_control", prompt, group + "-control", count=24)
    compare(run, name + ": generated-prefix output comparison", left, right)
    usage = run["samples"][name + "_reuse"]["usage"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    check(run, name + ": hit includes previously generated tokens",
          cached > original_tokens,
          {"original_prompt_tokens": original_tokens, "cached_tokens": cached},
          status="uncovered")
    control = run["samples"][name + "_control"]["usage"].get("prompt_tokens_details", {})
    check(run, name + ": fresh group has no prefix hit", control.get("cached_tokens", 0) == 0, control)


def generated_prefix(run, args):
    check(run, "public tokenizer provided", bool(args.public_tokenizer),
          "Use --public-tokenizer with the checkpoint directory.")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.public_tokenizer, local_files_only=True,
                                              trust_remote_code=False)
    before = idle(run)
    group = args.group_a + "-retained-" + str(time.time_ns())
    prompt = "Describe how the Moon's visible shape changes throughout a month.\nAnswer:"
    seed = complete(run, "retained_seed", prompt, group + "-seed", count=64)
    prompt_tokens = run["samples"]["retained_seed"]["usage"]["prompt_tokens"]
    pieces = [tokenizer.decode([token], clean_up_tokenization_spaces=False)
              for token in tokenizer.encode(seed, add_special_tokens=False)]
    marker = next(piece for piece in pieces[8:] if piece.strip() and seed.find(piece) >= 24
                  and len(tokenizer.encode(piece, add_special_tokens=False)) == 1)
    run["stop_marker"] = {"text": marker, "token_count": 1}
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
            continuation = seed[seed.index(marker) + len(marker):]
            followup(run, name, prompt + text + marker + continuation + "\nContinue.",
                     source_group, prompt_tokens)

    off = run["geometry_before"]["reasoning"]["kwargs"]["off"]
    eos_prompt = "Reply with exactly the word Done and nothing else."
    source_before = idle(run)
    text = complete(run, "retained_eos", eos_prompt, group + "-eos", count=160,
                    chat=True, ignore_eos=False, chat_kwargs=off)
    source_path(run, args, "retained_eos", source_before)
    source = run["samples"]["retained_eos"]
    check(run, "retained EOS actually occurred", source["finish_reason"] == "stop",
          status="uncovered")
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": eos_prompt}],
                                            tokenize=False, add_generation_prompt=True, **off)
    check(run, "public template reproduces EOS prompt length",
          len(tokenizer.encode(rendered, add_special_tokens=False)) == source["usage"]["prompt_tokens"])
    followup(run, "retained_eos", rendered + text + tokenizer.eos_token +
             "\nRepeat the same word.\nAnswer:", group + "-eos", source["usage"]["prompt_tokens"])

    body = dict(run["samples"]["retained_seed"]["input"], max_tokens=512, stream=True,
                cache_group=group + "-cancel", stream_options={"include_usage": True})
    source_before = idle(run)
    event = cancel_stream(run, body, min_chars=48)
    check(run, "retained cancellation closes active generation", event["cancelled"] and
          event.get("active_at_close", 0) > 0, event, status="uncovered")
    source_path(run, args, "retained_cancel", source_before)
    if event["retained_text"]:
        retained = event["retained_text"]
        compare(run, "cancelled text matches observed seed prefix", seed[:len(retained)], retained)
        continuation = seed[len(retained):] if seed.startswith(retained) else ""
        followup(run, "retained_cancel", prompt + retained + continuation + "\nContinue.",
                 group + "-cancel", prompt_tokens)
    after = idle(run)
    run["generated_prefix_observation"] = {"before": before, "after": after}

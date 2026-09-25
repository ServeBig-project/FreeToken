#!/usr/bin/env python3
"""A 32K-token public text smoke probe, not the complete 58-request workload."""

import argparse
import json
from pathlib import Path
import time

from http_client import check, compare, complete, idle, request


def make_prompt(tokenizer):
    prefix = "Read this reference archive. The archive access word is ORCHID.\n"
    unit = "This archive note records a routine weather observation and does not change the access word.\n"
    suffix = "\nWhat is the archive access word? Reply with that single word only.\nAccess word:"
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    repeats = (32768 - count(prefix + suffix)) // count(unit)
    prompt = prefix + unit * repeats + suffix
    return prompt, count(prompt), count(unit)


def observe(run, tokenizer, name, prompt, group):
    before = idle(run)
    text = complete(run, name, prompt, group, count=32)
    after = idle(run)
    sample = run["samples"][name]
    sample["tokenizer_prompt_tokens"] = len(tokenizer.encode(prompt, add_special_tokens=False))
    sample["before"], sample["after"] = before, after
    check(run, name + ": actual prompt usage", sample["usage"]["prompt_tokens"] ==
          sample["tokenizer_prompt_tokens"], sample["usage"])
    for section, fields in (("speculative", ("draft_tokens", "verify_steps")),
                            ("cuda_graph", ("draft", "verify"))):
        for field in fields:
            check(run, f"{name}: actual {section} {field}", after[section][field] >
                  before[section][field], status="uncovered")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--public-tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    run = {"url": args.url.rstrip("/"), "timeout": args.timeout, "arguments": vars(args),
           "checks": [], "http": [], "samples": {}, "scope": "32K entry smoke; not 58-request workload"}
    started = time.monotonic()
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.public_tokenizer, local_files_only=True,
                                                  trust_remote_code=False)
        run["model"] = args.model or request(run, "/v1/models")["data"][0]["id"]
        initial = idle(run)
        run["before"] = initial
        run["geometry"] = request(run, "/v1/cache/status")["geometry"]
        check(run, "harness SD configuration", initial["speculative"]["enabled"] and
              initial["speculative"]["max_draft_steps"] == 4, initial["speculative"])
        check(run, "harness Graph enabled", initial["cuda_graph"]["enabled"])
        prompt, count, unit_tokens = make_prompt(tokenizer)
        run["constructed_prompt_tokens"] = count
        check(run, "constructed prompt is approximately 32768 tokens",
              abs(count - 32768) <= unit_tokens, {"actual": count, "unit_tokens": unit_tokens})
        group = "sd-long-context-" + str(time.time_ns())
        first = observe(run, tokenizer, "long_source", prompt, group)
        run["access_word"] = {"expected": "ORCHID", "first_output_word": first.strip().split()[0]}
        check(run, "source retrieves the declared access word",
              run["access_word"]["first_output_word"].rstrip(".") == "ORCHID",
              run["access_word"], status="task_failed")
        continued = prompt + first + "\nRepeat the archive access word only.\nAccess word:"
        hit = observe(run, tokenizer, "long_prefix_hit", continued, group)
        cold = observe(run, tokenizer, "long_other_group", continued, group + "-control")
        compare(run, "long prefix hit versus fresh-group output", hit, cold)
        cached = run["samples"]["long_prefix_hit"]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        check(run, "long prefix hit includes generated content", cached > count,
              {"cached_tokens": cached, "original_prompt_tokens": count}, status="uncovered")
        control = run["samples"]["long_other_group"]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        check(run, "fresh group does not reuse long prefix", control == 0, control)
    except Exception as error:
        run["checks"].append({"name": "long-context smoke completed", "status": "failed",
                              "detail": f"{type(error).__name__}: {error}"})
    run["seconds"] = time.monotonic() - started
    counts = {status: sum(item["status"] == status for item in run["checks"])
              for status in ("passed", "failed", "investigate", "uncovered", "task_failed")}
    run["summary"] = counts
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output), "seconds": run["seconds"], **counts}))
    return (1 if counts["failed"] else 4 if counts["task_failed"] else
            2 if counts["investigate"] else 3 if counts["uncovered"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())

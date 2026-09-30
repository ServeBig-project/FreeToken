"""Independent HTTP regression workload; service startup belongs to the coordinator."""

import argparse
import json
import sys
import time
from pathlib import Path

from http_client import ServiceError, concurrent_requests, request, stream


COMPLETIONS = "/v1/completions"
PROMPT = "Continue this numbered list of ordinary household objects:\n1. Table\n2. Chair\n3."
CHAT_PROMPT = (
    "<|im_start|>user\nReply with exactly the word OK and end your response."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def require(condition, message):
    if not condition:
        raise ServiceError(message)


def cached_tokens(usage):
    return usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)


def completion(result, body, cache_report=False):
    require(result["status"] == 200, f"Completion returned HTTP {result['status']}: {result['body']}")
    payload = result["body"]
    choices = payload.get("choices", [])
    require(len(choices) == 1, "Expected one completion choice")
    choice = choices[0]
    require(isinstance(choice.get("text"), str), "Completion text is missing")
    require(choice.get("finish_reason") in ("stop", "length"), f"Unexpected finish_reason: {choice}")
    usage = payload.get("usage", {})
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(isinstance(usage.get(key), int) and usage[key] >= 0, f"Invalid usage.{key}: {usage}")
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Usage sum is incorrect")
    require(usage["prompt_tokens"] > 0, "Nonempty input has zero prompt tokens")
    require(usage["completion_tokens"] <= body["max_tokens"], "Completion exceeded max_tokens")
    if body.get("ignore_eos") and not body.get("stop"):
        require(usage["completion_tokens"] == body["max_tokens"], "ignore_eos completion ended before max_tokens")
        require(choice["finish_reason"] == "length", "ignore_eos completion has wrong finish_reason")
    if cache_report:
        cached = cached_tokens(usage)
        require(isinstance(cached, int) and 0 <= cached <= usage["prompt_tokens"], "Invalid cached_tokens")
    return choice["text"]


def geometry(payload):
    source = payload["geometry"]
    return {key: source[key] for key in ("num_pages", "num_mamba_slots", "moe_cache_size")
            if key != "num_mamba_slots" or source[key] > 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--adaptive", choices=("yes", "no"), required=True)
    parser.add_argument("--graph", choices=("yes", "no"), required=True)
    parser.add_argument("--max-draft-steps", type=int, default=8)
    parser.add_argument("--require-c16-n8", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, required=True)
    parser.add_argument("--long-prompt-file", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    report = {"label": args.label, "settings": vars(args).copy(), "cases": [], "failures": [],
              "missing_coverage": [], "within_run_text_differences": []}
    report["settings"] = {key: str(value) if isinstance(value, Path) else value for key, value in report["settings"].items()}

    def http(method, path, body=None):
        return request(args.url, method, path, body, args.timeout)

    def get(path):
        result = http("GET", path)
        require(result["status"] == 200, f"GET {path} returned HTTP {result['status']}")
        return result["body"]

    def case(name, operation):
        entry = {"name": name}
        report["cases"].append(entry)
        try:
            entry["stats_before"] = get("/v1/stats")
            operation(entry)
            entry["stats_after"] = get("/v1/stats")
            entry["passed"] = True
        except Exception as error:
            entry["passed"] = False
            entry["error"] = f"{type(error).__name__}: {error}"
            report["failures"].append({"case": name, "error": entry["error"]})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"{name}: {'PASS' if entry['passed'] else entry['error']}", flush=True)

    def body(prompt=PROMPT, limit=32, group="independent-self-sd", **extra):
        return {"model": report["model"], "prompt": prompt, "max_tokens": limit,
                "temperature": 0, "ignore_eos": True, "cache_group": group, **extra}

    def generate(entry, name, payload):
        result = http("POST", COMPLETIONS, payload)
        item = {"name": name, "request": payload, "response": result}
        entry.setdefault("requests", []).append(item)
        completion(result, payload, cache_report=True)
        return result["body"]

    def compare_text(name, left, right):
        before, after = left["choices"][0]["text"], right["choices"][0]["text"]
        if before != after:
            report["within_run_text_differences"].append({"case": name, "before": before, "after": after})

    def rebuild(entry, payload, expected):
        result = http("POST", "/v1/cache/rebuild", payload)
        entry.setdefault("rebuilds", []).append({"request": payload, "response": result})
        require(result["status"] in expected, f"Rebuild expected HTTP {expected}, got {result}")
        if result["status"] == 200:
            require(result["body"].get("status") == "ok", f"Rebuild did not succeed: {result}")
        return result

    def idle_rebuild(entry):
        deadline = time.monotonic() + args.timeout
        payload = {"mode": "if_idle", **report["geometry"], "timeout": args.timeout}
        while True:
            result = rebuild(entry, payload, (200, 503))
            if result["status"] == 200:
                return
            require(result["body"].get("status") == "busy", f"Rebuild failed without being busy: {result}")
            require(time.monotonic() < deadline, "Service did not become idle before rebuild deadline")
            time.sleep(0.2)

    def ready(entry):
        require(get("/health").get("status") == "ok", "Service is not healthy")
        report["model"] = get("/v1/models")["data"][0]["id"]
        report["initial_cache_status"] = get("/v1/cache/status")
        report["geometry"] = geometry(report["initial_cache_status"])
        stats = get("/v1/stats")
        speculative = stats["speculative"]
        require(speculative["enabled"], "SD is not enabled")
        require(speculative["adaptive_cost_enabled"] == (args.adaptive == "yes"), "Adaptive mode differs from requested matrix row")
        require(speculative["max_draft_steps"] == args.max_draft_steps, "Draft limit differs from requested matrix row")
        require(stats["cuda_graph"]["enabled"] == (args.graph == "yes"), "Graph mode differs from requested matrix row")
        idle_rebuild(entry)

    case("ready_and_clear", ready)
    if not report["cases"][-1]["passed"]:
        return 1

    def lengths(entry):
        for limit in (1, 2, 3, 7, 8, 9, 16, 33):
            generate(entry, f"max_tokens_{limit}", body(limit=limit, group=f"length-{limit}"))

    case("output_boundaries", lengths)

    def sampled(entry):
        for index, settings in enumerate(({"temperature": 0.7, "top_k": 1, "top_p": 1.0},
                                           {"temperature": 0.8, "top_k": 20, "top_p": 0.9},
                                           {"temperature": 1.0, "top_k": 50, "top_p": 0.95})):
            generate(entry, f"sampling_{index}", body(limit=24, group=f"sampling-{index}", **settings))

    case("sampling_parameters", sampled)

    def streaming(entry):
        payload = body(limit=33, group="streaming", stream_options={"include_usage": True})
        result = stream(args.url, COMPLETIONS, payload, args.timeout)
        entry["stream"] = {"request": payload, "response": result}
        text_parts, reasons, usage = [], [], None
        for chunk in result["chunks"]:
            for choice in chunk.get("choices", []):
                require(isinstance(choice.get("text"), str), f"Invalid streamed text: {choice}")
                text_parts.append(choice["text"])
                if choice.get("finish_reason") is not None:
                    reasons.append(choice["finish_reason"])
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
        require(reasons == ["length"], f"Unexpected streaming termination: {reasons}")
        synthetic = {"status": 200, "body": {"choices": [{"text": "".join(text_parts), "finish_reason": reasons[0]}], "usage": usage}}
        completion(synthetic, payload, cache_report=True)
        entry["assembled_response"] = synthetic["body"]

    case("streaming_usage", streaming)

    def stopping(entry):
        reference = generate(entry, "stop_reference", body(limit=48, group="stop-reference"))
        text = reference["choices"][0]["text"]
        require(bool(text.strip()), "Cannot select a stop string from empty reference output")
        marker = text.strip()[:8]
        entry["stop_marker"] = marker
        for index, stop in enumerate((marker, [marker, "UNLIKELY_STOP_SENTINEL"])):
            response = generate(entry, f"stop_{index}", body(limit=48, group=f"stop-{index}", stop=stop))
            require(marker not in response["choices"][0]["text"], "Stop string leaked into output")
            if response["choices"][0]["finish_reason"] != "stop":
                report["missing_coverage"].append(f"stop_{index}: generated text did not reach selected stop marker")
        generate(entry, "after_stop", body(limit=16, group="stop-0"))

    case("stop_and_reuse", stopping)

    def eos(entry):
        response = generate(entry, "eos", body(CHAT_PROMPT, 128, "eos", ignore_eos=False))
        if response["choices"][0]["finish_reason"] != "stop":
            report["missing_coverage"].append("EOS: model did not emit EOS within 128 output tokens")
        generate(entry, "after_eos", body(limit=16, group="eos"))

    case("eos_and_reuse", eos)

    def concurrent(entry, count, mixed=False):
        requests = [body(f"{PROMPT}\nList variation {index}:",
                         (1, 7, 8, 9, 17, 33, 65)[index % 7] if mixed else 65,
                         f"concurrency-{count}-{int(mixed)}-{index}") for index in range(count)]
        results = concurrent_requests(args.url, COMPLETIONS, requests, args.timeout)
        entry["requests"] = [{"name": f"request_{index}", "request": payload, "response": result}
                             for index, (payload, result) in enumerate(zip(requests, results))]
        for payload, result in zip(requests, results):
            completion(result, payload, cache_report=True)

    for count in (1, 4, 16):
        case(f"concurrency_{count}", lambda entry, count=count: concurrent(entry, count))
    case("mixed_lengths_tail_17", lambda entry: concurrent(entry, 17, mixed=True))

    prefix = "The red boat crosses the calm lake. The green trees grow beside the water.\n" * 32

    def cached(entry):
        idle_rebuild(entry)
        cold = None
        for name, prompt, group, expected in (
            ("cold", prefix, "prefix-a", "cold"),
            ("second", prefix, "prefix-a", "warm"),
            ("third", prefix, "prefix-a", "warm"),
            ("extension", prefix + "Name the colors in the scene:\n", "prefix-a", "warm"),
            ("isolated", prefix, "prefix-b", "cold"),
        ):
            response = generate(entry, name, body(prompt, 16, group))
            cached = cached_tokens(response["usage"])
            require((cached == 0) if expected == "cold" else (cached > 0),
                    f"{name}: expected {expected} prefix, cached_tokens={cached}")
            if name == "cold":
                cold = response
            elif name != "extension":
                compare_text(f"prefix_reuse/{name}", cold, response)
        requests = [body(prefix + f"Describe scene detail {index}:\n", 24, "prefix-a") for index in range(4)]
        results = concurrent_requests(args.url, COMPLETIONS, requests, args.timeout)
        for index, (payload, result) in enumerate(zip(requests, results)):
            entry["requests"].append({"name": f"shared_{index}", "request": payload, "response": result})
            completion(result, payload, cache_report=True)
            require(cached_tokens(result["body"]["usage"]) > 0,
                    f"Concurrent shared prefix {index} did not hit cache")
        response = generate(entry, "after_shared", body(prefix, 16, "prefix-a"))
        compare_text("prefix_reuse/after_shared", cold, response)

    case("prefix_reuse_and_isolation", cached)

    def long_input(entry):
        prompt = (args.long_prompt_file.read_text() if args.long_prompt_file
                  else [0] * (args.prefill_chunk_size + 1))
        response = generate(entry, "chunked_prefill", body(prompt, 17, "long-input"))
        require(response["usage"]["prompt_tokens"] > args.prefill_chunk_size,
                "Long-input fixture did not exceed configured prefill chunk size")
        generate(entry, "after_long_input", body(limit=17, group="after-long-input"))

    case("chunked_long_input", long_input)

    def cancel_busy(entry):
        attempted = False

        def while_active(chunk):
            nonlocal attempted
            if attempted or not any(choice.get("text") for choice in chunk.get("choices", [])):
                return
            attempted = True
            result = rebuild(entry, {"mode": "if_idle", **report["geometry"], "timeout": args.timeout}, (200, 503))
            if result["status"] == 200:
                report["missing_coverage"].append("Busy rebuild: workload finished before rejection could be observed")

        payload = body(limit=512, group="cancelled")
        entry["stream"] = {"request": payload, "response": stream(
            args.url, COMPLETIONS, payload, args.timeout, cancel_after_chunks=2, on_chunk=while_active)}
        require(attempted, "Cancelled stream did not produce content before closure")
        reused = generate(entry, "after_cancel", body(limit=33, group="cancelled"))
        fresh = generate(entry, "fresh_after_cancel", body(limit=33, group="fresh-after-cancel"))
        compare_text("cancelled_versus_fresh_group", fresh, reused)
        idle_rebuild(entry)

    case("busy_rebuild_cancel_and_reuse", cancel_busy)

    def rebuild_lifecycle(entry):
        before = geometry(get("/v1/cache/status"))
        generate(entry, "before_invalid_rebuild", body(prefix, 17, "invalid-rebuild"))
        for name, payload, status in (
            ("malformed", {"mode": "not-a-supported-mode"}, 422),
            ("unfit", {"mode": "if_idle", "num_pages": before["num_pages"] + 10**9}, 503),
        ):
            rebuild(entry, payload, (status,))
            require(geometry(get("/v1/cache/status")) == before, f"{name} rebuild changed cache geometry")
            response = generate(entry, f"after_{name}_rebuild", body(prefix, 17, "invalid-rebuild"))
            require(cached_tokens(response["usage"]) > 0,
                    f"Repeated prefix after {name} rebuild did not report a positive cache hit")
        idle_rebuild(entry)
        require(geometry(get("/v1/cache/status")) == before, "Same-size rebuild changed cache geometry")
        response = generate(entry, "after_idle_rebuild", body(prefix, 17, "prefix-a"))
        require(cached_tokens(response["usage"]) == 0, "Same-size rebuild retained old prefix")

    case("rebuild_preserves_service", rebuild_lifecycle)

    def activity(entry):
        stats = get("/v1/stats")
        spec = stats["speculative"]
        snapshots = [(item["stats_before"], item["stats_after"]) for item in report["cases"]
                     if "stats_before" in item and "stats_after" in item]
        entry["speculative_delta"] = {key: sum(max(0, after["speculative"][key] - before["speculative"][key])
                                              for before, after in snapshots) for key in (
            "draft_tokens", "accepted_draft_tokens", "verify_steps", "draft_expert_loads")}
        require(entry["speculative_delta"]["draft_tokens"] > 0, "Workload did not draft any tokens")
        require(entry["speculative_delta"]["verify_steps"] > 0, "Workload did not verify any drafts")
        require(0 <= spec["accepted_draft_tokens"] <= spec["draft_tokens"], "Accepted draft count is invalid")
        require(any(spec["draft_length_histogram"]), "Draft length histogram is empty")
        observed_by_shape = {}
        for before, after in snapshots:
            previous = {(shape["phase"], shape["batch_size"], shape["query_tokens"], shape["physical_query_tokens"]): shape["replays"]
                        for shape in before["cuda_graph"]["replay_shapes"]}
            for shape in after["cuda_graph"]["replay_shapes"]:
                key = (shape["phase"], shape["batch_size"], shape["query_tokens"], shape["physical_query_tokens"])
                if shape["replays"] > previous.get(key, 0):
                    require(shape["physical_query_tokens"] >= shape["query_tokens"] > 0, f"Invalid Graph shape: {shape}")
                    observed_by_shape[key] = shape
        observed = list(observed_by_shape.values())
        entry["observed_graph_shapes"] = observed
        if args.graph == "yes":
            for count in (1, 4, 16):
                if not any(shape["batch_size"] == count for shape in observed):
                    report["missing_coverage"].append(f"Graph actual batch C{count} not observed")
            if not any(shape["batch_size"] not in (1, 4, 16) for shape in observed):
                report["missing_coverage"].append("Graph natural tail batch not observed")
            if args.require_c16_n8:
                if not any(shape["phase"] == "verify" and shape["batch_size"] == 16
                           and shape["query_tokens"] == 144 for shape in observed):
                    report["missing_coverage"].append("Fixed N8: Graph actual B16/query_tokens144 not observed")
        report["final_stats"] = stats
        report["final_cache_status"] = get("/v1/cache/status")

    case("observed_sd_and_graph_activity", activity)
    print(json.dumps({"failures": report["failures"], "missing_coverage": report["missing_coverage"],
                      "within_run_text_differences": len(report["within_run_text_differences"])}, ensure_ascii=False))
    return 1 if report["failures"] else (2 if report["missing_coverage"] else 0)


if __name__ == "__main__":
    sys.exit(main())

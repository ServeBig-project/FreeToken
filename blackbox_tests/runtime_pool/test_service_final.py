"""Independent HTTP acceptance from the published runtime-pool contract only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.error
import urllib.request


def request_json(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return json.load(response)


def measure(values):
    values = sorted(x for x in values if x is not None)
    return {"mean": statistics.mean(values),
            "p95": values[max(0, (95 * len(values) + 99) // 100 - 1)]} if values else None


class Run:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=False)
        self.results = []
        self.phases = []
        self.failures = []
        self.samples = []
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.model = request_json(args.url, "/v1/models")["data"][0]["id"]
        self.initial = request_json(args.url, "/v1/cache/status")
        self.experts = self.initial["geometry"]["moe_cache_size"]
        assert self.initial["state"] == "serving", self.initial
        self.write("initial.json", self.initial)
        self.write("stats-before.json", request_json(args.url, "/v1/stats"))

    def write(self, name, value):
        (self.out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    def observe(self):
        while not self.stop.is_set():
            try:
                state = request_json(self.args.url, "/v1/cache/status")
                self.samples.append({"at": time.monotonic() - self.started, "status": state})
                if state["geometry"]["moe_cache_size"] != self.experts:
                    self.failures.append("expert capacity changed")
                runtime = state.get("prefix_cache", {}).get("runtime")
                if runtime:
                    if not 0 <= runtime["used_bytes"] <= runtime["held_bytes"] <= runtime["budget_bytes"]:
                        self.failures.append("runtime used/held/budget relationship failed")
                    if runtime["waste_bytes"] < 0:
                        self.failures.append("runtime waste is negative")
                    for name, value in runtime.get("components", {}).items():
                        if not 0 <= value["used_bytes"] <= value["held_bytes"] or value["waste_bytes"] < 0:
                            self.failures.append("invalid component bytes: " + name)
            except Exception as error:
                self.failures.append("status observation: " + repr(error))
            self.stop.wait(1)

    def completion(self, name, body, first):
        start = time.monotonic()
        result = {"name": name, "body": body, "started": start - self.started,
                  "events": [], "text": "", "usages": [], "finish_reasons": []}
        try:
            payload = dict(body, stream=True, stream_options={"include_usage": True})
            request = urllib.request.Request(self.args.url + "/v1/completions",
                                             json.dumps(payload).encode(),
                                             {"Content-Type": "application/json"})
            ids = set()
            done = False
            with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    elapsed = time.monotonic() - start
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    result["events"].append({"at": elapsed, "data": event})
                    if "error" in event:
                        raise AssertionError(event["error"])
                    if event.get("id"):
                        ids.add(event["id"])
                    if event.get("usage"):
                        result["usages"].append(event["usage"])
                    for choice in event.get("choices", []):
                        text = choice.get("text", "")
                        if text:
                            result.setdefault("ttft", elapsed)
                            result["last_text_at"] = elapsed
                            result["text"] += text
                            first.set()
                        if choice.get("finish_reason"):
                            result["finish_reasons"].append(choice["finish_reason"])
            assert done, "stream ended without [DONE]"
            assert len(ids) <= 1, "request identity changed within stream"
            assert result["text"], "empty successful output"
            assert result["usages"], "no usage returned"
            usage = result["usages"][-1]
            assert usage["completion_tokens"] == body["max_tokens"], usage
            assert usage["prompt_tokens"] > 0, usage
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], usage
            assert result["finish_reasons"] == ["length"], result["finish_reasons"]
        except Exception as error:
            detail = error.read().decode() if isinstance(error, urllib.error.HTTPError) else ""
            result["error"] = repr(error) + detail
            self.failures.append(name + ": " + result["error"])
        result["elapsed"] = time.monotonic() - start
        result["ended"] = time.monotonic() - self.started
        first.set()
        self.write(name + ".json", result)
        return result

    def phase(self, name, bodies, first_token_gate=False):
        start = time.monotonic()
        bodies = [dict(item, model=item.get("model", self.model)) for item in bodies]
        record = {"name": name, "requests": bodies, "first_token_gate": first_token_gate}
        self.phases.append(record)
        self.write("requests.json", self.phases)
        before = request_json(self.args.url, "/v1/cache/status")
        stats_before = request_json(self.args.url, "/v1/stats")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for index, body in enumerate(bodies):
                first = threading.Event()
                futures.append(executor.submit(self.completion, f"{name}-{index}", body, first))
                if first_token_gate and index == 0:
                    assert first.wait(self.args.timeout), "leading request made no progress"
            results = [future.result() for future in futures]
        after = request_json(self.args.url, "/v1/cache/status")
        stats_after = request_json(self.args.url, "/v1/stats")
        elapsed = time.monotonic() - start
        self.results.extend(results)
        self.write(name + "-phase.json", {"elapsed": elapsed, "before": before,
                                          "stats_before": stats_before, "stats_after": stats_after,
                                          "after": after, "metrics": self.metrics(results, elapsed)})
        print(json.dumps({"phase": name, "elapsed": elapsed,
                          "errors": [r.get("error") for r in results if r.get("error")]}), flush=True)
        return results

    @staticmethod
    def metrics(results, elapsed):
        tokens = sum(r["usages"][-1]["completion_tokens"] for r in results if r["usages"])
        tpot, gaps, overlaps = [], [], []
        for result in results:
            if result["usages"] and "ttft" in result:
                count = result["usages"][-1]["completion_tokens"]
                if count > 1:
                    tpot.append((result["last_text_at"] - result["ttft"]) / (count - 1))
            arrivals = [e["at"] for e in result["events"]
                        if any(c.get("text") for c in e["data"].get("choices", []))]
            gaps.extend(b - a for a, b in zip(arrivals, arrivals[1:]))
        for i, left in enumerate(results):
            for right in results[i + 1:]:
                if "ttft" in left and "ttft" in right:
                    overlaps.append(max(left["started"] + left["ttft"], right["started"] + right["ttft"])
                                    < min(left["ended"], right["ended"]))
        return {"elapsed_seconds": elapsed, "requests": len(results), "output_tokens": tokens,
                "output_tokens_per_second_full_request": tokens / elapsed,
                "requests_per_second": len(results) / elapsed,
                "latency_seconds": measure([r["elapsed"] for r in results]),
                "ttft_seconds": measure([r.get("ttft") for r in results]),
                "tpot_seconds": measure(tpot), "max_stream_output_gap_seconds": max(gaps, default=None),
                "pairwise_output_overlap": overlaps,
                "cached_prompt_tokens": [r["usages"][-1].get("prompt_tokens_details", {}).get("cached_tokens")
                                         if r["usages"] else None for r in results]}

    def finish(self):
        self.stop.set()
        self.observer.join(timeout=35)
        self.write("status-samples.json", self.samples)
        self.write("final.json", request_json(self.args.url, "/v1/cache/status"))
        self.write("stats-after.json", request_json(self.args.url, "/v1/stats"))
        self.write("summary.json", {"failures": sorted(set(self.failures)),
                                    "metrics": self.metrics(self.results, time.monotonic() - self.started)})


def body(prompt, output, group):
    return {"prompt": prompt, "max_tokens": output, "ignore_eos": True,
            "temperature": 0, "cache_group": group}


def history(index, paragraphs):
    line = (f"Notebook {index}: In spring the gardeners planted beans beside the river. "
            "They recorded the weather, checked the soil, and carried water to each row.\n")
    return line * paragraphs + "Continue this notebook with detailed observations.\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=["sequence", "pressure", "peak"], default="sequence")
    parser.add_argument("--replay", help="Exact requests.json from another run, or the same public phase format")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--pressure-output", type=int, default=1200)
    parser.add_argument("--long-paragraphs", type=int, default=160)
    args = parser.parse_args()
    run = Run(args)
    run.observer = threading.Thread(target=run.observe, daemon=True)
    run.observer.start()
    try:
        if args.replay:
            for phase in json.loads(Path(args.replay).read_text()):
                run.phase(phase["name"], phase["requests"], phase.get("first_token_gate", False))
        elif args.case == "pressure":
            run.phase("reference", [body(history(0, 4), 64, "pressure-reference")])
            run.phase("pressure", [body(history(i, 4), args.pressure_output, f"pressure-{i}") for i in range(6)])
            run.phase("after", [body(history(0, 4), 64, "pressure-reference")])
        elif args.case == "sequence":
            prompts = [history(i, 4) for i in range(4)]
            short = run.phase("short", [body(p, args.output_tokens, "sequence") for p in prompts])
            run.phase("long", [body(prompts[i] + short[i]["text"] + history(i, args.long_paragraphs),
                                    args.output_tokens, "sequence") for i in range(2)])
            run.phase("short-again", [body(prompts[i] + short[i]["text"] + "\nContinue briefly.\n",
                                           args.output_tokens, "sequence") for i in range(4)])
            run.phase("repeat", [body(p, 32, "sequence") for p in prompts])
        else:
            prompt = history(0, args.long_paragraphs)
            warm = run.phase("history", [body(prompt, args.output_tokens, "peak")])[0]
            lead = body(prompt + warm["text"] + "\nContinue the detailed notebook.\n", 1200, "peak")
            new = [body(history(i + 1, args.long_paragraphs), args.output_tokens, f"peak-{i}")
                   for i in range(7)]
            run.phase("decode-and-prefill", [lead] + new, first_token_gate=True)
            run.phase("after", [body(history(0, 4), 32, "peak-after")])
    finally:
        run.finish()
    if run.failures:
        raise SystemExit("FAIL: " + "; ".join(sorted(set(run.failures))))


if __name__ == "__main__":
    main()

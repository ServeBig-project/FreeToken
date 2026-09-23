"""Public stop/cancel/prefix and one cache-rebuild cycle; no server launch."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import httpx

from benchmark_http import idle
from collect_http import PROMPTS, shape_counts

PREFIX = "Describe how a Python generator resumes after yielding a value, with a small example:"


def body(prompt, limit=17, stream=True, **extra):
    return {"prompt": prompt, "temperature": 0, "max_tokens": limit, "ignore_eos": True,
            "stream": stream, **extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("off", "router"), required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "lifecycle", "base_url": args.base_url, "mode": args.mode,
              "execution": args.execution, "stages": [], "rebuilds": [], "checks": {}}

    def save():
        (args.output / "lifecycle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        assert client.get("/health").json()["status"] == "ok"
        report["models"] = client.get("/v1/models").json()
        model = report["models"]["data"][0]["id"]
        report["initial_stats"] = idle(client)

        def call(request, cancel_before=None):
            payload = {"model": model, **request}
            parts, usage, finish, done, during = [], None, None, False, None
            if not request["stream"]:
                response = client.post("/v1/completions", json=payload)
                assert response.status_code == 200, response.text
                data = response.json()
                assert len(data["choices"]) == 1, data
                choice = data["choices"][0]
                parts, usage, finish, done = [choice["text"]], data["usage"], choice["finish_reason"], True
            else:
                payload["stream_options"] = {"include_usage": True}
                pieces = 0
                with client.stream("POST", "/v1/completions", json=payload) as response:
                    assert response.status_code == 200, response.read().decode()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        if line == "data: [DONE]":
                            done = True
                            break
                        data = json.loads(line[6:])
                        if data.get("usage"):
                            usage = data["usage"]
                        for choice in data["choices"]:
                            text = choice.get("text") or ""
                            parts.append(text)
                            pieces += bool(text)
                            if choice.get("finish_reason") is not None:
                                finish = choice["finish_reason"]
                        if cancel_before is not None and pieces >= 2:
                            snapshot = client.get("/v1/stats").json()
                            verified = snapshot["speculative"]["verify_steps"] > cancel_before["speculative"]["verify_steps"]
                            if args.execution == "graph":
                                verified = verified and snapshot["cuda_graph"]["verify"] > cancel_before["cuda_graph"]["verify"]
                            if snapshot["requests"]["active"] >= 2 and verified:
                                during = snapshot
                                break
            return {"request": payload, "text": "".join(parts), "usage": usage,
                    "finish_reason": finish, "done": done, "cancelled": during is not None,
                    "cancel_snapshot": during}

        def run(label, requests, capacity=1706, cancel=False):
            before = idle(client)
            stage = {"label": label, "inputs": requests, "stats_before": before}
            report["stages"].append(stage)
            (args.output / "phase.txt").write_text(f"lifecycle-{label}")
            save()
            with ThreadPoolExecutor(max_workers=len(requests)) as pool:
                rows = list(pool.map(lambda item: call(item[1], before if cancel and item[0] == 0 else None), enumerate(requests)))
            stage["responses"] = rows
            after = stage["stats_after"] = idle(client)
            old, new = shape_counts(before), shape_counts(after)
            stage["replay_delta"] = [{"phase": key[0], "batch_size": key[1], "query_tokens": key[2],
                                      "replays": value - old.get(key, 0)}
                                     for key, value in new.items() if value > old.get(key, 0)]
            checks = report["checks"]
            checks[f"{label}:geometry"] = (after["moe_residency"]["cache_slots"] == capacity
                                           and after["moe_residency"]["resident_experts"] == 0
                                           and after["kv"]["total_pages"] == 4096)
            checks[f"{label}:features"] = (after["speculative"]["enabled"]
                                           and after["speculative"]["draft_residency"] == args.mode
                                           and not after["speculative"]["adaptive_enabled"]
                                           and not after["speculative"]["reuse_enabled"])
            checks[f"{label}:execution"] = after["cuda_graph"]["enabled"] == (args.execution == "graph")
            checks[f"{label}:replay"] = bool(stage["replay_delta"]) if args.execution == "graph" else not stage["replay_delta"]
            for index, (request, row) in enumerate(zip(requests, rows)):
                if cancel and index == 0:
                    checks[f"{label}:cancel"] = row["cancelled"] and not row["done"]
                    continue
                usage = row["usage"]
                valid = row["done"] and usage is not None and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                if "stop" in request:
                    valid = valid and row["finish_reason"] == "stop" and request["stop"] not in row["text"]
                else:
                    valid = valid and row["finish_reason"] == "length" and usage["completion_tokens"] == request["max_tokens"]
                checks[f"{label}:response-{index}"] = bool(valid)
            save()
            print(json.dumps({"label": label, "capacity": capacity, "replays": stage["replay_delta"]}), flush=True)
            return rows

        def rebuild(size):
            assert idle(client)["requests"]["active"] == 0
            (args.output / "phase.txt").write_text(f"lifecycle-rebuild-{size}")
            response = client.post("/v1/cache/rebuild", json={"moe_cache_size": size, "mode": "if_idle", "timeout": 300.0})
            record = {"size": size, "http_status": response.status_code, "response": response.json()}
            report["rebuilds"].append(record)
            save()
            assert response.status_code == 200 and record["response"]["status"] == "ok", record

        shrunk = False
        try:
            cold = run("prefix-cold", [body(PREFIX, stream=False)])[0]
            warm = run("prefix-warm", [body(PREFIX)])[0]
            cached = lambda row: row["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
            report["checks"]["prefix_reused"] = cached(warm) > cached(cold) and cached(warm) > 0
            middle = len(warm["text"]) // 2
            stop = warm["text"][middle:middle + 8]
            assert stop, "Need visible output for the public stop probe"
            run("stop-stream", [body(PREFIX, stop=stop)])
            run("stop-nonstream", [body(PREFIX, stream=False, stop=stop)])
            run("churn", [body(prompt) for prompt in PROMPTS])
            run("prefix-after-churn", [body(PREFIX, stream=False)])
            run("cancel-with-survivors", [body(PROMPTS[0], 128)] + [body(prompt, 17) for prompt in PROMPTS[1:]], cancel=True)
            run("after-cancel", [body(PREFIX), body(PROMPTS[1], 7, stream=False)])
            rebuild(256)
            shrunk = True
            run("pool256-cold", [body(PROMPTS[2], stream=False)], capacity=256)
            run("pool256-warm", [body(PROMPTS[2])], capacity=256)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            save()
            raise
        finally:
            if shrunk:
                try:
                    rebuild(1706)
                    run("restored1706", [body(PREFIX, stream=False)])
                except Exception as error:
                    report["restoration_error"] = f"{type(error).__name__}: {error}"
                    save()
                    raise
        report["probe_completed"] = True
        report["missing_graph_coverage"] = [name for name, passed in report["checks"].items() if name.endswith(":replay") and not passed]
        report["passed"] = all(report["checks"].values())
        save()
        assert report["passed"], [name for name, passed in report["checks"].items() if not passed]
        print(f"Public lifecycle evidence: {args.output / 'lifecycle.json'}", flush=True)


if __name__ == "__main__":
    main()

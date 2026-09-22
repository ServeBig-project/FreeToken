"""Bounded public lifecycle/rebuild checks; run only on a dedicated idle server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import httpx
from tokenizers import Tokenizer

from evaluate_http import Server, request, stream
from test_serving import LONG, long_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--resident-count", type=int, default=0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"requests": [], "rebuilds": []}
    tokenizer = Tokenizer.from_file("/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B/tokenizer.json")
    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        assert client.get("/health").json()["status"] == "ok"
        model = client.get("/v1/models").json()["data"][0]["id"]
        observer = Server(args.output.name, client)

        def call(body):
            result = stream(client, model, body)
            report["requests"].append(result)
            return {"text": result["text"], "finish_reason": result["finish_reason"],
                    "usage": {key: result["usage"][key] for key in
                              ("prompt_tokens", "completion_tokens", "total_tokens")}}

        def batch(bodies):
            with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
                return list(pool.map(call, bodies))

        def geometry():
            stats = observer.idle()
            resident = stats["moe_residency"]
            assert resident["resident_experts"] == args.resident_count
            assert resident["temporary_slots"] == resident["cache_slots"] - args.resident_count
            assert stats["speculative"]["adaptive_enabled"] is args.adaptive
            assert stats["speculative"]["reuse_enabled"] is args.reuse
            return stats

        try:
            before = geometry()
            original_size = before["moe_residency"]["cache_slots"]
            assert original_size == 1536
            control = request(LONG, 17)
            expected = call(control)
            bodies = [request(LONG, limit) for limit in (1, 3, 7, 17)]
            ordinary_current_mode = [call(body) for body in bodies]
            assert batch(bodies) == ordinary_current_mode
            stop = expected["text"][3:9]
            assert stop
            stopped = call({**control, "stop": stop})
            assert stopped["finish_reason"] == "stop" and stop not in stopped["text"]
            eos = call(request("<|im_start|>user\nReply only with OK. /no_think<|im_end|>\n"
                               "<|im_start|>assistant\n<think>\n\n</think>\n\n", 64, False))
            assert eos["finish_reason"] == "stop", "EOS coverage missing"
            for limit in (1, 3, 17):
                result = call(request(long_prompt(tokenizer, "Near context: ", 1023), limit))
                assert result["usage"]["completion_tokens"] == 1 and result["finish_reason"] == "length"

            during, cancel_start = None, geometry()
            body = {"model": model, **request(LONG, 256), "stream": True}
            with client.stream("POST", "/v1/completions", json=body) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    if line.startswith("data: {") and any(c.get("text") for c in json.loads(line[6:])["choices"]):
                        snapshot = observer.stats()
                        verified = snapshot["speculative"]["verify_steps"] > cancel_start["speculative"]["verify_steps"]
                        if snapshot["requests"]["active"] > 0 and (not snapshot["speculative"]["enabled"] or verified):
                            during = snapshot
                            break
            assert during is not None, "Cancellation coverage missing"
            after_cancel = geometry()
            for key in ("draft_tokens", "accepted_draft_tokens", "verify_steps", "adaptive_stops", "reuse_changed_routes"):
                assert after_cancel["speculative"][key] >= during["speculative"][key]
            pressure = [request(long_prompt(tokenizer, f"Capacity request {index}: ", 1008), 8) for index in range(5)]
            for _ in range(2):
                assert all(row["usage"]["completion_tokens"] == 8 for row in batch(pressure))
                geometry()

            for size, expected_status in [(1408, 200), (args.resident_count + 255, 503), (1536, 200)]:
                prior = geometry()
                response = client.post("/v1/cache/rebuild", json={"moe_cache_size": size, "mode": "if_idle", "timeout": 300.0})
                report["rebuilds"].append({"size": size, "http_status": response.status_code,
                                           "response": response.json(), "cache_status": client.get("/v1/cache/status").json()})
                assert response.status_code == expected_status, response.text
                if expected_status == 200:
                    assert response.json()["status"] == "ok"
                else:
                    assert "status" in response.json() and "error" in response.json()
                current = geometry()
                assert current["moe_residency"]["cache_slots"] == (size if expected_status == 200 else prior["moe_residency"]["cache_slots"])
                assert call(control) == expected
            after = geometry()
            for counter in ("draft_tokens", "accepted_draft_tokens", "verify_steps", "adaptive_stops", "reuse_changed_routes"):
                values = [snapshot["speculative"][counter] for snapshot in observer.snapshots]
                assert all(first <= second for first, second in zip(values, values[1:])), counter
            for enabled, counter in [(args.adaptive, "adaptive_stops"), (args.reuse, "reuse_changed_routes")]:
                if not enabled:
                    assert after["speculative"][counter] == 0
            if args.reuse:
                assert after["speculative"]["reuse_changed_routes"] > before["speculative"]["reuse_changed_routes"], "Routing reuse not exercised"
            report["passed"] = True
        finally:
            report["stats"] = observer.snapshots
            (args.output / "lifecycle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

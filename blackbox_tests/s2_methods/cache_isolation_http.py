"""Public generated-KV isolation and normal prompt-cache reuse probe."""

import argparse
import json
from pathlib import Path

import httpx
from tokenizers import Tokenizer

from evaluate_http import Server, request, stream


PROMPT = "In plain text with no newline, the sequence after 1, 2, 3, 4, 5 is:"
CONTINUE = "\nNow continue the sequence with five more integers:"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--rebuild-first", action="store_true", help="Clear prior cache history through the public rebuild API")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file("/data2/servebig-envs/s2_sd_models/Qwen3-30B-A3B/tokenizer.json")
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False).ids
    report = {"original_prompt": PROMPT, "original_prompt_tokens": len(prompt_ids), "requests": []}
    with httpx.Client(base_url=args.base_url, timeout=900) as client:
        assert client.get("/health").json()["status"] == "ok"
        model = client.get("/v1/models").json()["data"][0]["id"]
        observer = Server(args.output.name, client)

        def send(label, prompt, limit):
            result = stream(client, model, request(prompt, limit))
            report["requests"].append({"label": label, **result})
            expected_length = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            assert result["usage"]["prompt_tokens"] == expected_length
            details = result["usage"].get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens", 0)
            assert isinstance(cached, int) and 0 <= cached <= expected_length
            observer.idle()
            return result, cached

        try:
            initial = observer.idle()
            assert initial["speculative"]["reuse_enabled"] is True, "Run this probe with verification reuse enabled"
            if args.rebuild_first:
                response = client.post("/v1/cache/rebuild", json={
                    "moe_cache_size": initial["moe_residency"]["cache_slots"], "mode": "if_idle", "timeout": 300.0})
                report["cache_reset"] = {"http_status": response.status_code, "response": response.json()}
                assert response.status_code == 200 and response.json()["status"] == "ok", response.text
                observer.idle()
            first, _ = send("original", PROMPT, 64)
            assert first["usage"]["completion_tokens"] == 64 and first["text"].strip()
            extended = PROMPT + first["text"] + CONTINUE
            extended_ids = tokenizer.encode(extended, add_special_tokens=False).ids
            report["extended_prompt"] = extended
            report["extended_prompt_tokens"] = len(extended_ids)
            report["token_prefix_preserved"] = extended_ids[:len(prompt_ids)] == prompt_ids
            assert report["token_prefix_preserved"], "Probe coverage missing: concatenation changed the original token prefix"
            _, first_cached = send("extended-first", extended, 8)
            _, repeated_cached = send("extended-repeat", extended, 8)
            report["checks"] = {
                "generated_kv_not_reused_as_prompt": first_cached <= len(prompt_ids),
                "normal_extended_prompt_cache_reused": len(prompt_ids) < repeated_cached <= len(extended_ids),
            }
            report["cached_tokens"] = {"extended_first": first_cached, "extended_repeat": repeated_cached}
            report["passed"] = all(report["checks"].values())
            assert report["passed"], report["checks"]
        finally:
            report["stats"] = observer.snapshots
            (args.output / "cache-isolation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({key: report[key] for key in ("original_prompt_tokens", "extended_prompt_tokens", "cached_tokens", "checks", "passed") if key in report}), flush=True)


if __name__ == "__main__":
    main()

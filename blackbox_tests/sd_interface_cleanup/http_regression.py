#!/usr/bin/env python3
"""Independent HTTP regression using the published DFlash service contract."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urlopen(req, timeout=600) as response:
            status, raw = response.status, response.read()
    except HTTPError as error:
        status, raw = error.code, error.read()
    return {"status": status, "body": json.loads(raw), "elapsed_seconds": time.monotonic() - started}


def save(directory, label, value):
    (directory / (label + ".json")).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def completion(base_url, model, prompt, max_tokens):
    return request(base_url, "/v1/completions", {
        "model": model, "prompt": prompt, "temperature": 0,
        "max_tokens": max_tokens, "stream": False,
    })


def check_completion(result, max_tokens):
    assert result["status"] == 200, f"completion HTTP {result['status']}: {result['body']}"
    body = result["body"]
    assert len(body["choices"]) == 1, "expected one completion choice"
    choice = body["choices"][0]
    assert isinstance(choice["text"], str), "completion text is not a string"
    assert choice["finish_reason"] in ("length", "stop"), f"finish_reason={choice['finish_reason']}"
    usage = body["usage"]
    assert 0 < usage["completion_tokens"] <= max_tokens, f"completion usage outside requested limit: {usage}"
    assert usage["prompt_tokens"] > 0, f"missing prompt usage: {usage}"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], f"inconsistent usage: {usage}"


def check_dflash(status):
    dflash = status["geometry"]["dflash"]
    assert dflash["active"] is True, "DFlash is inactive"
    for name in ("weight_bytes", "context_bytes", "metadata_bytes", "reserved_bytes"):
        assert isinstance(dflash[name], int) and dflash[name] >= 0, f"invalid {name}: {dflash[name]}"
    assert dflash["weight_bytes"] > 0 and dflash["context_bytes"] > 0, "DFlash has no weights or persistent context"
    assert dflash["reserved_bytes"] == sum(dflash[name] for name in ("weight_bytes", "context_bytes", "metadata_bytes")), "DFlash reserved bytes differ from component sum"
    return dflash


def check_geometry(status, pages):
    assert status["state"] == "serving", f"cache state={status['state']}"
    geometry = status["geometry"]
    assert geometry["num_pages"] == pages and geometry["page_size"] == 1, "target KV capacity changed unexpectedly"
    assert geometry["moe_cache_size"] == 1536, "expert pool capacity changed"
    dflash = check_dflash(status)
    replay = geometry["gdn_replayssm"]
    assert replay["active"] is True, "Replay is inactive"
    assert replay["reserved_bytes"] <= replay["state_budget_bytes"], "Replay exceeds its state budget"
    assert replay["reserved_bytes"] + dflash["reserved_bytes"] <= 6245744640, "Replay and DFlash exceed their original combined budget"
    total = (pages * geometry["unit_bytes"]["kv_per_token"]
             + 1536 * geometry["unit_bytes"]["moe_per_expert"]
             + replay["reserved_bytes"] + dflash["reserved_bytes"])
    assert total <= geometry["cache_budget_bytes"], "declared cache allocations exceed the total cache budget"


def check_statistics(before, after):
    old, new = before["speculative"], after["speculative"]
    assert new["enabled"] and new["adaptive_cost_enabled"], "adaptive speculative decoding is not enabled"
    assert new["max_draft_steps"] == 8, "unexpected maximum draft length"
    for name in ("draft_tokens", "verify_steps"):
        assert new[name] > old[name], f"{name} did not increase"
    assert new["accepted_draft_tokens"] >= old["accepted_draft_tokens"], "accepted draft token count decreased"
    assert new["accepted_draft_tokens"] <= new["draft_tokens"], "accepted more tokens than drafted"
    assert sum(new["draft_length_histogram"]) > sum(old["draft_length_histogram"]), "draft length histogram did not increase"
    assert new["dflash_block_gpu_ms"] > old.get("dflash_block_gpu_ms", 0), "DFlash cumulative block GPU time did not increase"
    samples = new["dflash_block_samples"]
    old_samples = old.get("dflash_block_samples", {})
    assert sum(samples.values()) > sum(old_samples.values()), "DFlash block samples did not increase"
    for key, count in samples.items():
        batch, length = map(int, key.split(":"))
        assert 1 <= batch <= 16 and 1 <= length <= 8 and count > 0, f"invalid DFlash block sample {key}: {count}"
    choices = new["dflash_block_choices"]
    old_choices = old.get("dflash_block_choices", [0] * len(choices))
    assert sum(choices) > sum(old_choices), "DFlash block choices did not increase"
    assert after["cuda_graph"]["enabled"], "CUDA Graph is disabled"
    assert after["cuda_graph"]["verify"] > before["cuda_graph"]["verify"], "verification did not use CUDA Graph"


def run_wave(base_url, model):
    cases = [
        ("Continue the counting sequence: one, two, three, four,", 1),
        ("Explain in plain language why the sky appears blue during the day.", 17),
        ("Write a numbered list of practical steps for organizing a small home library.", 33),
        ("Write a detailed explanation of how a seed grows into a mature tree, including the role of water and sunlight.", 65),
    ]
    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(completion, base_url, model, prompt, maximum) for prompt, maximum in cases]
        results = [{"prompt": prompt, "max_tokens": maximum, "response": future.result()}
                   for (prompt, maximum), future in zip(cases, futures)]
    for case in results:
        check_completion(case["response"], case["max_tokens"])
    return results


def snapshot(base_url, directory, label):
    result = {}
    for kind, path in (("status", "/v1/cache/status"), ("stats", "/v1/stats")):
        response = request(base_url, path)
        save(directory, label + "-" + kind, response)
        assert response["status"] == 200, f"{path}: HTTP {response['status']}"
        result[kind] = response["body"]
    return result


def observe_generation(base_url, model, directory, label, before):
    waves = []
    for wave in range(2):
        results = run_wave(base_url, model)
        save(directory, f"{label}-wave-{wave}", results)
        waves.append(results)
        after = snapshot(base_url, directory, f"{label}-after-{wave}")
        if after["stats"]["speculative"].get("dflash_block_gpu_ms", 0) > before["stats"]["speculative"].get("dflash_block_gpu_ms", 0):
            break
    check_statistics(before["stats"], after["stats"])
    return waves, after


def run(args, directory):
    initial = snapshot(args.base_url, directory, "initial")
    check_geometry(initial["status"], 4096)
    geometry = initial["status"]["geometry"]
    assert geometry["gdn_replayssm"]["state_budget_bytes"] + geometry["dflash"]["reserved_bytes"] == 6245744640, "initial combined budget differs from the coordinator's configuration"
    before_waves, before_rebuild = observe_generation(args.base_url, args.model, directory, "before-rebuild", initial)
    check_geometry(before_rebuild["status"], 4096)
    assert before_rebuild["stats"]["requests"]["active"] == 0, "requests remain active before idle rebuild"

    rebuilt = request(args.base_url, "/v1/cache/rebuild", {"num_pages": 3072, "mode": "if_idle", "timeout": 300})
    save(directory, "legal-rebuild", rebuilt)
    assert 200 <= rebuilt["status"] < 300 and rebuilt["body"]["status"] == "ok", f"legal rebuild failed: {rebuilt}"
    after_rebuild = snapshot(args.base_url, directory, "rebuilt")
    check_geometry(after_rebuild["status"], 3072)
    original = check_dflash(before_rebuild["status"])
    reduced = check_dflash(after_rebuild["status"])
    assert reduced["weight_bytes"] == original["weight_bytes"], "KV rebuild changed DFlash weight bytes"
    assert reduced["context_bytes"] < original["context_bytes"], "KV shrink did not shrink persistent DFlash context"
    assert reduced["reserved_bytes"] < original["reserved_bytes"], "KV shrink did not update DFlash reserved bytes"
    after_waves, after_generation = observe_generation(args.base_url, args.model, directory, "after-rebuild", after_rebuild)
    check_geometry(after_generation["status"], 3072)

    invalid = request(args.base_url, "/v1/cache/rebuild", {"num_pages": -1, "mode": "if_idle", "timeout": 300})
    save(directory, "invalid-rebuild", invalid)
    assert invalid["status"] == 503 and invalid["body"]["status"] == "rejected", f"invalid rebuild was not rejected: {invalid}"
    assert "num_pages must be positive" in invalid["body"]["error"], "invalid rebuild did not explain the parameter error"
    after_invalid = snapshot(args.base_url, directory, "after-invalid")
    check_geometry(after_invalid["status"], 3072)
    assert after_invalid["status"]["geometry"] == after_generation["status"]["geometry"], "invalid rebuild changed existing resources"
    last = completion(args.base_url, args.model, "Explain in plain language why the sky appears blue during the day.", 17)
    save(directory, "completion-after-invalid", last)
    check_completion(last, 17)
    final = snapshot(args.base_url, directory, "final")
    check_geometry(final["status"], 3072)

    differences = []
    for earlier, later in zip(before_waves[0], after_waves[0]):
        old_text = earlier["response"]["body"]["choices"][0]["text"]
        new_text = later["response"]["body"]["choices"][0]["text"]
        if old_text != new_text:
            differences.append({"prompt": earlier["prompt"], "max_tokens": earlier["max_tokens"], "before": old_text, "after": new_text})
    save(directory, "greedy-text-differences", differences)
    return {"status": "passed", "configuration": "DFlash adaptive + Replay + CUDA Graph", "initial_dflash": original,
            "rebuilt_dflash": reduced, "greedy_text_differences": len(differences),
            "stats_before_rebuild": before_rebuild["stats"]["speculative"],
            "stats_after_rebuild_generation": after_generation["stats"]["speculative"],
            "invalid_rebuild_http_status": invalid["status"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="sd-cleanup")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args, directory)
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    save(directory, "report", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()

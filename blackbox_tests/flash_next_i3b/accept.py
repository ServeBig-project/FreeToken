"""I3b HTTP acceptance against an externally started tiered-KV service."""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer
from histories import distributed_history

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flash_next_i3a"))
from cases import copy_history
from run import Suite, arguments


class TieredSuite(Suite):
    def __init__(self, *args):
        super().__init__(*args)
        self.host_scope = None
        self.host_snapshot = None
        self.host_output = {"frames": 0, "characters": 0}
        self.last_host_read_bytes = 0
        self.mixed_frames = {}
        self.mixed_last = None
        self.mixed_interleaved = False

    def record(self, event, **fields):
        super().record(event, **fields)
        if event != "sse" or fields["data"] == "[DONE]":
            return
        frame = json.loads(fields["data"])
        pieces = [choice.get("text") or choice.get("delta", {}).get("content", "") for choice in frame.get("choices", [])]
        count = sum(len(piece) for piece in pieces if piece)
        name = fields["name"]
        if count and name in self.mixed_frames:
            if self.mixed_last != name and self.mixed_frames[name] > 0:
                self.mixed_interleaved = True
            self.mixed_frames[name] += 1
            self.mixed_last = name
        if not self.host_scope or not self.host_snapshot or name != self.host_scope["name"]:
            return
        if not self.host_snapshot["eligible"]:
            return
        if count:
            if not self.host_output["frames"]:
                self.host_output["first_time"] = time.time()
                self.host_output["read_bytes_first"] = self.host_snapshot["kv_host_read_bytes"]
                super().record("active_host_output_started", name=fields["name"], residency=self.host_snapshot)
            self.host_output["frames"] += 1
            self.host_output["characters"] += count
            self.host_output["last_time"] = time.time()

    async def snapshot(self):
        status, stats = await super().snapshot()
        prefix = status["prefix_cache"]
        runtime = prefix["runtime"]
        assert 0 <= prefix["host_used_bytes"] <= prefix["host_budget_bytes"]
        assert prefix["host_budget_bytes"] == self.args.host_gib * 2**30
        assert stats["execution"]["effective"]["kv_placement"] == "tiered"
        for key in ("kv_gpu_payload_pages", "kv_host_payload_pages", "kv_host_payload_bytes"):
            assert runtime[key] >= 0, key
        assert runtime["kv_host_payload_bytes"] <= prefix["host_used_bytes"]
        assert runtime["kv_host_read_bytes"] >= self.last_host_read_bytes
        self.last_host_read_bytes = runtime["kv_host_read_bytes"]
        if self.host_scope:
            scope = self.host_scope
            self.host_snapshot = {key: runtime[key] for key in (
                "kv_gpu_payload_pages", "kv_host_payload_pages", "kv_host_payload_bytes", "paused",
            )}
            self.host_snapshot["host_used_bytes"] = prefix["host_used_bytes"]
            self.host_snapshot["kv_host_read_bytes"] = runtime["kv_host_read_bytes"]
            self.host_snapshot["eligible"] = (
                runtime["paused"] == scope["paused"]
                and runtime["kv_host_payload_pages"] > scope["old_payload_pages"]
                and runtime["kv_gpu_payload_pages"] < scope["input_pages"]
            )
        return status, stats

    def begin_host_scope(self, status, name, size):
        runtime = status["prefix_cache"]["runtime"]
        page_size = status["geometry"]["page_size"]
        self.host_scope = {
            "name": name, "paused": runtime["paused"],
            "old_payload_pages": runtime["kv_gpu_payload_pages"] + runtime["kv_host_payload_pages"],
            "input_pages": (size + page_size - 1) // page_size,
            "host_reads_before": runtime["kv_host_read_bytes"],
        }

    async def finish_host_scope(self):
        after, _ = await self.snapshot()
        before = self.host_scope["host_reads_before"]
        if after["prefix_cache"]["runtime"]["kv_host_read_bytes"] == before:
            after, _ = await self.snapshot()
        reads = after["prefix_cache"]["runtime"]["kv_host_read_bytes"]
        self.host_output["read_bytes_delta"] = reads - before
        if "read_bytes_first" in self.host_output:
            self.host_output["read_bytes_since_first_host_output"] = reads - self.host_output["read_bytes_first"]
        self.record("active_host_evidence", **self.host_output)
        assert self.host_output["frames"] >= 2, "no sustained output with attributable, unpaused host history"
        assert reads > before, "selected host KV reads were not observed, including one allowed delayed sample"
        self.host_scope = None
        self.checks.append("unique unpaused request preserved output while its history occupied host and selected host KV was read")

    async def reproduce(self):
        original = json.loads(self.args.original_request.read_text())
        body, resource = original["body"], original["resource_scope"]
        assert self.args.runtime_gib == resource["runtime_gib"] == 2
        assert self.args.host_gib == resource["host_gib"] == 4
        assert self.initial["geometry"]["moe_cache_size"] == resource["moe_cache_size"] == 2048
        assert self.first_stats["requests"]["active"] == 0
        effective = self.first_stats["execution"]["effective"]
        for key, value in (("dense_quant", "fp8"), ("kv_dtype", "int8"), ("batching_policy", "layered-pipeline")):
            assert effective[key] == value, (key, effective[key])
        assert body["model"] == self.model
        assert body["max_tokens"] == 4096
        size = len(self.tokenizer.encode(body["prompt"], add_special_tokens=False))
        assert size == original["expected"]["prompt_tokens"] == 157309
        expected = body["prompt"].split("BEGIN RECORDS\n", 1)[1].split("\nEND RECORDS", 1)[0]
        self.begin_host_scope(self.initial, "original-157309", size)
        self.record("exact_reproduction", original_request=str(self.args.original_request), prompt_tokens=size, scope=self.host_scope)
        await self.observe([asyncio.create_task(self.send(
            "original-157309", "/v1/completions", body, expected, size,
        ))])
        self.record("original_request_completed", prompt_tokens=size, exact_records_preserved=True)
        await self.finish_host_scope()
        self.record("result", status="original_failure_resolved", checked=self.checks,
                    pending=["The original single-request failure gate does not replace the complete phase matrix."])

    async def active_host(self):
        before, stats = await self.snapshot()
        assert stats["requests"]["active"] == 0
        runtime = before["prefix_cache"]["runtime"]
        size = (runtime["context_tokens"] * 3 // 5) // 64 * 64 + 61
        assert size + self.args.output_limit <= runtime["context_tokens"]
        prompt, expected = copy_history(self.tokenizer, size, "host_owner", self.args.copy_lines)
        self.begin_host_scope(before, "unique-active-host", size)
        self.record("active_host_trace", prompt_tokens=size, output_limit=self.args.output_limit, scope=self.host_scope)
        await self.observe([asyncio.create_task(self.complete(
            self.host_scope["name"], prompt, expected, group="host-history", max_tokens=self.args.output_limit, stream=True,
        ))])
        await self.finish_host_scope()
        before, _ = await self.snapshot()
        await self.complete("host-prefix-revisit", prompt, expected, group="host-history", max_tokens=self.args.output_limit, stream=True)
        after, _ = await self.snapshot()
        reused = sum(after["prefix_cache"][key] - before["prefix_cache"][key]
                     for key in ("gpu_reused_tokens", "host_reused_tokens"))
        assert reused > 0, "long host history was not reused"
        self.checks.append("long history prefix revisit preserved exact output")
        return size

    async def mixed_histories(self, solo_size):
        before, _ = await self.snapshot()
        runtime = before["prefix_cache"]["runtime"]
        size = solo_size
        first_output = asyncio.Event()
        cases = [distributed_history(self.tokenizer, size + i, label, 100007 + i * 100000, self.args.copy_lines)
                 for i, label in enumerate(("red", "blue", "gold"))]
        self.record("mixed_history_trace", prompt_tokens=[len(prompt) for prompt, _, _ in cases],
                    source_positions=[spans for _, _, spans in cases], output_limit=self.args.output_limit)
        self.mixed_frames = {f"tiered-history-{i}": 0 for i in range(3)}

        async def arrive(i):
            if i:
                await first_output.wait()
                await asyncio.sleep(i * 0.2)
            prompt, expected, _ = cases[i]
            assert len(self.tokenizer.encode(expected, add_special_tokens=False)) < self.args.output_limit
            return await self.complete(
                f"tiered-history-{i}", prompt, expected, max_tokens=self.args.output_limit,
                stream=True, started=first_output if i == 0 else None,
            )
        await self.observe([asyncio.create_task(arrive(i)) for i in range(3)])
        self.record("mixed_history_output", frames=self.mixed_frames, interleaved=self.mixed_interleaved)
        assert self.mixed_interleaved, "long histories completed without observable interleaved output"
        after, _ = await self.snapshot()
        self.record("mixed_history_deltas", values={key: after["prefix_cache"]["runtime"][key] - runtime[key]
                    for key in ("paused", "restored", "recompute", "recomputed_tokens")})
        self.checks.append("three arriving histories preserved exact output and usage within fixed GPU/host budgets")

    async def run(self):
        await self.prepare()
        if self.args.original_request:
            await self.reproduce()
            return
        if self.args.scenario == "full":
            await self.boundaries()
            await self.reuse()
        if self.args.scenario != "pause":
            await self.cancellation()
            await self.maintenance()
        if self.args.scenario != "maintenance":
            size = await self.active_host()
            await self.mixed_histories(size)
        _, stats = await self.snapshot()
        graph_delta = stats["cuda_graph"]["target_decode"] - self.first_stats["cuda_graph"]["target_decode"]
        replay_delta = stats["gdn_replayssm"]["ar_tokens"] - self.first_stats["gdn_replayssm"]["ar_tokens"]
        assert (graph_delta > 0) if self.args.graph == "on" else (graph_delta == 0)
        assert (replay_delta > 0) if self.args.replay == "on" else (replay_delta == 0)
        self.pending.extend([
            "Full format/configuration matrix, non-aligned pause positions and final 157K multi-user acceptance remain separate.",
            "A byte-level proof of shared host copies is not available from aggregate residency counts alone.",
        ])
        self.record("result", status="incomplete", checked=self.checks, pending=self.pending)
        print("I3b checks completed; full acceptance remains incomplete. See JSONL report.")


async def main():
    parser = arguments()
    parser.description = __doc__
    parser.add_argument("--host-gib", type=float, required=True)
    parser.add_argument("--original-request", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    with args.report.open("x", encoding="utf-8") as report:
        async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
            suite = TieredSuite(client, tokenizer, args, report)
            suite.record("run_parameters", values={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
            try:
                await asyncio.wait_for(suite.run(), args.timeout * 20)
            except BaseException as error:
                suite.record("result", status="failed", error=f"{type(error).__name__}: {error}")
                raise


if __name__ == "__main__":
    asyncio.run(main())

"""Run only against an externally started I3a service; never starts a model."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from cases import assert_answer, copy_history, history, shared_history, token_history


async def wait_for_output(task, event, timeout):
    waiting = asyncio.create_task(event.wait())
    try:
        done, _ = await asyncio.wait((waiting, task), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
        assert waiting in done, "no output arrived before the test deadline"
        assert not task.done(), "request completed before the cancellation case could run"
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


class Suite:
    def __init__(self, client, tokenizer, args, report):
        self.client, self.tokenizer = client, tokenizer
        self.args, self.report = args, report
        self.model = None
        self.initial = None
        self.first_stats = None
        self.group_prefix = f"i3a-{time.time_ns()}"
        self.reuse_round = 0
        self.checks = []
        self.pending = []

    def record(self, event, **fields):
        self.report.write(json.dumps({"time": time.time(), "event": event, **fields}) + "\n")
        self.report.flush()

    async def snapshot(self):
        status, stats = await asyncio.gather(
            self.client.get("/v1/cache/status"), self.client.get("/v1/stats")
        )
        status.raise_for_status()
        stats.raise_for_status()
        status, stats = status.json(), stats.json()
        self.record("snapshot", status=status, stats=stats)
        runtime = status["prefix_cache"]["runtime"]
        budget = runtime["budget_bytes"]
        assert budget > 0
        assert budget == runtime["held_bytes"] + runtime["idle_bytes"] + runtime["free_bytes"]
        assert runtime["held_bytes"] == runtime["used_bytes"] + runtime["waste_bytes"]
        assert 0 <= runtime["protected_bytes"] <= runtime["idle_bytes"]
        for field in ("held_bytes", "used_bytes", "waste_bytes", "idle_bytes", "protected_bytes"):
            assert all(component[field] >= 0 for component in runtime["components"].values())
            assert sum(component[field] for component in runtime["components"].values()) == runtime[field], field
        for name, component in runtime["components"].items():
            assert 0 <= component["held_bytes"] <= budget, name
        assert runtime["free_bytes"] >= 0
        prefix = status["prefix_cache"]
        assert 0 <= prefix["host_allocated_bytes"] <= prefix["host_budget_bytes"]
        if self.initial:
            assert budget == self.initial["prefix_cache"]["runtime"]["budget_bytes"]
            assert status["geometry"]["moe_cache_size"] == self.initial["geometry"]["moe_cache_size"]
        return status, stats

    async def complete(self, name, prompt, expected=None, group=None, max_tokens=64, stream=False, started=None):
        body = {
            "model": self.model, "max_tokens": max_tokens, "temperature": 0,
            "stream": stream, "cache_group": f"{self.group_prefix}:{group or name}",
        }
        if isinstance(prompt, list):
            path = "/v1/completions"
            body["prompt"] = self.tokenizer.decode(prompt, skip_special_tokens=False)
            assert self.tokenizer.encode(body["prompt"], add_special_tokens=False) == prompt
        else:
            path = "/v1/chat/completions"
            body["messages"] = [{"role": "user", "content": prompt}]
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if stream:
            body["stream_options"] = {"include_usage": True}
        self.record("request", name=name, path=path, body=body)
        if not stream:
            response = await self.client.post(path, json=body)
            self.record("response", name=name, code=response.status_code, body=response.text)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            text = choice["text"] if path == "/v1/completions" else choice["message"]["content"]
            usage, finish = data["usage"], choice["finish_reason"]
        else:
            text, usage, finish, done = "", None, None, False
            async with self.client.stream("POST", path, json=body) as response:
                if response.is_error:
                    await response.aread()
                    self.record("response", name=name, code=response.status_code, body=response.text)
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    value = line[5:].strip()
                    self.record("sse", name=name, data=value)
                    if value == "[DONE]":
                        done = True
                        break
                    frame = json.loads(value)
                    assert "error" not in frame, frame
                    if frame.get("usage"):
                        usage = frame["usage"]
                    for choice in frame.get("choices", []):
                        piece = choice.get("text") if path == "/v1/completions" else choice.get("delta", {}).get("content")
                        if piece:
                            text += piece
                            if started:
                                started.set()
                        if choice.get("finish_reason") is not None:
                            assert finish is None, "duplicate final choice"
                            finish = choice["finish_reason"]
            assert done, "SSE ended without [DONE]"
        assert finish in ("stop", "length"), finish
        assert usage is not None, "missing usage"
        assert 0 <= usage["completion_tokens"] <= max_tokens
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        if isinstance(prompt, list):
            assert usage["prompt_tokens"] == len(prompt), (usage, len(prompt))
        if expected is not None:
            assert_answer(text, expected)
        self.record("validated", name=name, text=text, usage=usage, finish_reason=finish)
        return text, usage

    async def observe(self, tasks):
        async def poll():
            while True:
                await self.snapshot()
                await asyncio.sleep(0.25)
        monitor = asyncio.create_task(poll())
        work = asyncio.gather(*tasks)
        try:
            completed, _ = await asyncio.wait((work, monitor), timeout=self.args.timeout, return_when=asyncio.FIRST_COMPLETED)
            if not completed:
                raise TimeoutError("finite request set made insufficient progress before the test deadline")
            if monitor in completed:
                await monitor
            return await work
        finally:
            monitor.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()
            work.cancel()
            await asyncio.gather(monitor, work, *tasks, return_exceptions=True)

    async def boundaries(self):
        source = self.tokenizer.encode("plain input " * 65, add_special_tokens=False)
        for size in (3, 4, 5, 63, 64, 65):
            await self.complete(f"boundary-{size}", source[:size], max_tokens=4, stream=size % 2 == 1)
        for size in (255, 256, 257, 1023, 1024, 1025):
            prompt = token_history(self.tokenizer, size, "amber", "581204")
            await self.complete(f"history-{size}", prompt, "ANSWER=581204")
        self.checks.append("exact input boundaries and semantic history")

    async def reuse(self):
        self.reuse_round += 1
        cycle = self.reuse_round
        cases = [shared_history(label) for label in ("amber", "birch", "coral", "denim")]
        for i, (prompt, answer) in enumerate(cases):
            await self.complete(f"isolated-{i}", prompt, answer)
        for wave in range(3):
            tasks = []
            for i, (prompt, answer) in enumerate(cases):
                tasks.append(asyncio.create_task(self.complete(
                    f"reuse-{wave}-{i}", prompt, answer,
                    group="shared-records" if wave < 2 else f"separate-{i}", stream=True,
                )))
            await self.observe(tasks)
        prompt = token_history(self.tokenizer, self.args.pressure_tokens, "cache", "673201")
        await self.complete(f"warm-first-{cycle}", prompt, "ANSWER=673201", group=f"warm-{cycle}")
        before, _ = await self.snapshot()
        await self.complete(f"warm-again-{cycle}", prompt, "ANSWER=673201", group=f"warm-{cycle}")
        after, _ = await self.snapshot()
        reused = lambda status: sum(status["prefix_cache"][key] for key in ("gpu_reused_tokens", "host_reused_tokens"))
        if before["prefix_cache"]["enabled"]:
            assert reused(after) > reused(before), "immediate warm prefix reuse was not observed"
        else:
            assert reused(after) == reused(before), "naive mode reused a public prefix"
        before = after
        await self.complete(f"isolated-cache-{cycle}", prompt, "ANSWER=673201", group=f"isolated-cache-{cycle}")
        after, _ = await self.snapshot()
        assert reused(after) == reused(before), "cache_group isolation was violated"
        self.checks.append("shared and isolated groups preserve semantic answers")

    async def pressure(self):
        size = self.args.pressure_tokens
        prompts = [token_history(self.tokenizer, size + i, f"user{i}", str(510123 + i * 817)) for i in range(5)]
        for i, prompt in enumerate(prompts):
            await self.complete(f"pressure-isolated-{i}", prompt, f"ANSWER={510123 + i * 817}")
        async def arrive(wave, i, prompt):
            await asyncio.sleep((0, 0.5 + i * 0.1, 1.2)[wave])
            return await self.complete(
                f"pressure-{wave}-{i}", prompt, f"ANSWER={510123 + i * 817}",
                group=f"pressure-user-{i}", max_tokens=self.args.output_limit, stream=True,
            )
        tasks = []
        for wave in range(3):
            for i, prompt in enumerate(prompts):
                tasks.append(asyncio.create_task(arrive(wave, i, prompt)))
        await self.observe(tasks)
        copies = [copy_history(self.tokenizer, size + i, f"owner{i}", self.args.copy_lines) for i in range(3)]
        for i, (prompt, answer) in enumerate(copies):
            assert len(self.tokenizer.encode(answer, add_special_tokens=False)) < self.args.output_limit
            await self.complete(f"copy-isolated-{i}", prompt, answer, max_tokens=self.args.output_limit)
        await self.observe([asyncio.create_task(self.complete(
            f"growing-{i}", prompt, answer, max_tokens=self.args.output_limit, stream=True,
        )) for i, (prompt, answer) in enumerate(copies)])
        await self.reuse()
        self.checks.append("finite burst and staggered histories finish under fixed budget")

    async def cancellation(self):
        prompt, answer = copy_history(self.tokenizer, self.args.pressure_tokens, "survivor", self.args.copy_lines)
        first_output = asyncio.Event()
        cancelled = asyncio.create_task(self.complete(
            "cancel-stream", prompt, group="cancel-shared", max_tokens=self.args.output_limit, stream=True, started=first_output,
        ))
        survivor = None
        try:
            await wait_for_output(cancelled, first_output, self.args.timeout)
            survivor_output = asyncio.Event()
            survivor = asyncio.create_task(self.complete(
                "surviving-stream", prompt, answer, group="cancel-shared", max_tokens=self.args.output_limit,
                stream=True, started=survivor_output,
            ))
            await wait_for_output(survivor, survivor_output, self.args.timeout)
            assert not cancelled.done(), "first stream ended before the two-stream cancellation case"
            response = await self.client.post("/v1/cache/rebuild", json={
                "runtime_cache_gib": self.args.runtime_gib, "mode": "if_idle", "timeout": 1,
            })
            self.record("busy_rebuild", code=response.status_code, body=response.text)
            assert response.status_code in (409, 503), "busy rebuild was not refused"
            body = response.json()
            assert body.get("status") == "busy", body
        except BaseException:
            if survivor:
                survivor.cancel()
                await asyncio.gather(survivor, return_exceptions=True)
            raise
        finally:
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)
        self.record("cancelled", name="cancel-stream", method="SSE disconnect")
        await self.observe([survivor])
        prompt, answer = history("after_cancel", "926173")
        await self.complete("after-cancel", prompt, answer)
        self.checks.append("SSE cancellation and busy maintenance refusal")

    async def maintenance(self):
        for field, value in (("runtime_cache_gib", -1), ("runtime_cache_gib", 0), ("num_pages", 1),
                             ("num_mamba_slots", 1), ("num_swa_pages", 1), ("swa_full_tokens_ratio", 0.5)):
            body = {"mode": "if_idle", "timeout": 1, field: value}
            response = await self.client.post("/v1/cache/rebuild", json=body)
            self.record("rejected_rebuild", request=body, code=response.status_code, body=response.text)
            assert response.status_code in (422, 503), response.text
            if response.status_code == 503:
                assert response.json().get("status") == "rejected", response.text
            prompt, answer = history("after_rejection", "732061")
            await self.complete(f"after-rejection-{field}", prompt, answer)
            await self.snapshot()
        response = await self.client.post("/v1/cache/rebuild", json={"mode": "force"})
        self.record("unsupported_maintenance_mode", code=response.status_code, body=response.text)
        assert response.status_code == 422, response.text
        response = await self.client.post("/v1/cache/rebuild", json={
            "runtime_cache_gib": self.args.runtime_gib, "mode": "if_idle", "timeout": self.args.timeout,
        })
        self.record("idle_rebuild", code=response.status_code, body=response.text)
        response.raise_for_status()
        assert response.json()["status"] == "ok", response.text
        prompt, answer = history("after_rebuild", "417852")
        await self.complete("after-rebuild", prompt, answer, stream=True)
        await self.snapshot()
        self.checks.append("invalid rebuild keeps service usable; idle rebuild preserves expert capacity")

    async def pause(self):
        before, _ = await self.snapshot()
        runtime = before["prefix_cache"]["runtime"]
        size = (runtime["context_tokens"] * 3 // 5) // 64 * 64 + 61
        assert size + 2 + self.args.output_limit <= runtime["context_tokens"]
        first_output = asyncio.Event()
        cases = [copy_history(self.tokenizer, size + i, f"paused{i}", self.args.copy_lines) for i in range(3)]
        self.record("pause_trace", prompt_tokens=[len(prompt) for prompt, _ in cases], output_limit=self.args.output_limit)

        async def arrive(i):
            if i:
                await first_output.wait()
                await asyncio.sleep(i * 0.2)
            prompt, expected = cases[i]
            return await self.complete(
                f"pause-history-{i}", prompt, expected, max_tokens=self.args.output_limit,
                stream=True, started=first_output if i == 0 else None,
            )
        await self.observe([asyncio.create_task(arrive(i)) for i in range(3)])
        after, _ = await self.snapshot()
        deltas = {key: after["prefix_cache"]["runtime"][key] - runtime[key]
                  for key in ("paused", "restored", "recompute", "recomputed_tokens")}
        observed = deltas["paused"] > 0 and deltas["restored"] + deltas["recompute"] > 0
        self.record("result", status="incomplete", checked=["three long histories preserve exact output and usage"],
                    pause_resume_observed=observed, execution_deltas=deltas,
                    pending=["Exact non-aligned pause positions require public evidence."] if observed else ["No actual pause and resume observed."])
        print(f"Long-history trace completed; pause/resume observed: {observed}.")

    async def run(self):
        health = await self.client.get("/health")
        health.raise_for_status()
        assert health.json()["status"] == "ok", health.text
        models = await self.client.get("/v1/models")
        models.raise_for_status()
        self.model = models.json()["data"][0]["id"]
        self.initial, self.first_stats = await self.snapshot()
        runtime = self.initial["prefix_cache"]["runtime"]
        assert abs(runtime["budget_bytes"] - self.args.runtime_gib * 2**30) <= runtime["granularity_bytes"]
        assert self.args.pressure_tokens + 4 + self.args.output_limit <= runtime["context_tokens"]
        assert not self.first_stats["speculative"]["enabled"]
        assert bool(self.first_stats["cuda_graph"]["enabled"]) == (self.args.graph == "on")
        if self.args.scenario == "pause":
            await self.pause()
            return
        if self.args.scenario == "full":
            await self.boundaries()
            await self.reuse()
            await self.pressure()
        await self.cancellation()
        status, stats = await self.snapshot()
        deltas = {
            name: status["prefix_cache"]["runtime"][name] - runtime[name]
            for name in ("paused", "restored", "recompute", "recomputed_tokens")
        }
        deltas["graph_decode"] = stats["cuda_graph"]["target_decode"] - self.first_stats["cuda_graph"]["target_decode"]
        for name in ("ar_tokens", "flushes"):
            deltas[f"replay_{name}"] = stats["gdn_replayssm"][name] - self.first_stats["gdn_replayssm"][name]
        self.record("execution_deltas", values=deltas)
        if self.args.graph == "on":
            assert deltas["graph_decode"] > 0, "Graph enabled but no decode execution observed"
        else:
            assert deltas["graph_decode"] == 0
        if self.args.replay == "on":
            assert deltas["replay_ar_tokens"] > 0, "Replay enabled but no AR tokens observed"
            if not deltas["replay_flushes"]:
                self.pending.append("Replay wrap/flush was not observed.")
        else:
            assert deltas["replay_ar_tokens"] == 0
        if not deltas["paused"] or not (deltas["restored"] or deltas["recompute"]):
            self.pending.append("A real pause followed by restore/recompute was not observed.")
        await self.maintenance()
        self.pending.extend([
            "Reconcile all model KV/index/request state bytes against physical allocation.",
            "Prove pause/restore/recompute at non-4/64 positions, plus paused/queued cancellation.",
            "Verify resource release after cancellation against idle active-state semantics.",
            "Exercise prefill cancellation and max legal single-request input.",
        ])
        self.record("result", status="incomplete", checked=self.checks, pending=self.pending)
        print("I3a preparation cases completed; full acceptance remains incomplete. See JSONL report.")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--tokenizer", default="/data1/yuchen/models/Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--runtime-gib", type=float, required=True)
    parser.add_argument("--pressure-tokens", type=int, required=True)
    parser.add_argument("--output-limit", type=int, required=True)
    parser.add_argument("--copy-lines", type=int, required=True)
    parser.add_argument("--graph", choices=("on", "off"), required=True)
    parser.add_argument("--replay", choices=("on", "off"), required=True)
    parser.add_argument("--scenario", choices=("full", "maintenance", "pause"), default="full")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    with args.report.open("x", encoding="utf-8") as report:
        async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
            suite = Suite(client, tokenizer, args, report)
            suite.record("run_parameters", values={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
            try:
                await asyncio.wait_for(suite.run(), args.timeout * 20)
            except BaseException as error:
                suite.record("result", status="failed", error=f"{type(error).__name__}: {error}")
                raise


if __name__ == "__main__":
    asyncio.run(main())

"""Independent early/middle/tail facts; run only after the frozen long-output trace."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flash_next_i3b"))
from accept import TieredSuite
from histories import positioned_records


RECORDS = (
    ("red", ["red_early=KITE_4817", "red_middle=AMBER_9026", "red_tail=CEDAR_1359"]),
    ("blue", ["blue_early=LYNX_6402", "blue_middle=MAPLE_3178", "blue_tail=HERON_8254"]),
    ("gold", ["gold_early=ORCHID_7591", "gold_middle=RAVEN_2064", "gold_tail=QUARTZ_5830"]),
)


def build_fixtures(tokenizer):
    cases = []
    for i, (name, records) in enumerate(RECORDS):
        tokens, expected, spans = positioned_records(tokenizer, 157309 + i, records)
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        assert tokenizer.encode(text, add_special_tokens=False) == tokens
        cases.append(dict(name=name, prompt=text, prompt_tokens=len(tokens), expected=expected, source_positions=spans))
    return cases


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    args.runtime_gib, args.host_gib = 2, 8
    args.pressure_tokens, args.output_limit = 157311, 128
    args.graph, args.replay = "on", "off"
    cases = json.loads(args.fixtures.read_text())
    tokenizer = AutoTokenizer.from_pretrained("/data1/yuchen/models/Qwen3.8-Flash-Next-NVFP4", local_files_only=True)
    with args.report.open("x", encoding="utf-8") as report:
        async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
            suite = TieredSuite(client, tokenizer, args, report)
            passed = []
            try:
                await suite.prepare()
                assert suite.initial["geometry"]["moe_cache_size"] == 2048
                for case in cases:
                    assert len(tokenizer.encode(case["prompt"], add_special_tokens=False)) == case["prompt_tokens"]
                    suite.record("position_case", name=case["name"], prompt_tokens=case["prompt_tokens"],
                                 expected=case["expected"], source_positions=case["source_positions"])
                    body = dict(model=suite.model, prompt=case["prompt"], temperature=0, max_tokens=128,
                                stream=True, stream_options={"include_usage": True},
                                cache_group=f"{suite.group_prefix}:independent-{case['name']}")
                    await suite.observe([asyncio.create_task(suite.send(
                        case["name"], "/v1/completions", body, case["expected"], case["prompt_tokens"],
                    ))])
                    passed.append(case["name"])
                suite.record("result", status="independent_position_facts_passed", cases=passed,
                             scope="Position semantics only; long-output interleaving has a separate frozen trace.")
            except BaseException as error:
                suite.record("result", status="failed", cases_completed=passed, error=f"{type(error).__name__}: {error}")
                raise


if __name__ == "__main__":
    asyncio.run(main())

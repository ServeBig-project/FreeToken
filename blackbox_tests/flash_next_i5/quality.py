"""Reuse the independent 74888a0 quality cases against an existing HTTP service."""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_next import scenarios, tasks
from flash_next.client import Client


class RecordedClient(Client):
    def __init__(self, url, record):
        self.record = record
        super().__init__(url)

    def post(self, path, body, timeout=None):
        self.record("request", path=path, body=body)
        response = super().post(path, body, timeout)
        self.record("response", code=response.status_code, body=response.text)
        return response

    def stream(self, prompt, max_tokens, group=None, **kwargs):
        self.record("stream_request", prompt=prompt, max_tokens=max_tokens, cache_group=group, options=kwargs)
        result = super().stream(prompt, max_tokens, group, **kwargs)
        self.record("stream_result", result=result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cases = (scenarios.test_01_single_tasks_alone, scenarios.test_02_conversations,
             scenarios.test_03_chat_api, scenarios.test_07_gsm8k, scenarios.test_08_stream_stop_eos)
    with args.report.open("x", encoding="utf-8") as report:
        report_lock = threading.Lock()
        def record(event, **fields):
            with report_lock:
                report.write(json.dumps({"time": time.time(), "event": event, **fields}, ensure_ascii=False) + "\n")
                report.flush()

        client = RecordedClient(args.url, record)
        before, cache = client.get("/v1/stats"), client.get("/v1/cache/status")
        session = SimpleNamespace(c=client, tok=tasks.Tok(), cfg={"radix": cache["prefix_cache"]["enabled"]}, rec={})
        session.rec.update(name=args.name, source_cases_commit="74888a0", stats_ready=before, cache_ready=cache)
        record("public_configuration", name=args.name, stats=before, cache=cache, source_cases_commit="74888a0")
        records = args.report.with_suffix(".records.json")
        try:
            for case in cases:
                record("case_started", name=case.__name__)
                case(session)
                record("case_passed", name=case.__name__)
                records.write_text(json.dumps(session.rec, ensure_ascii=False, indent=2))
                print(f"Passed {case.__name__}", flush=True)
            session.rec.update(stats_end=client.get("/v1/stats"), cache_end=client.get("/v1/cache/status"))
            record("result", status="selected_quality_cases_passed", cases=[case.__name__ for case in cases],
                   gsm8k=session.rec["gsm8k"]["correct"], pending="Cross-precision/reference comparison remains separate.")
        except BaseException as error:
            record("result", status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            records.write_text(json.dumps(session.rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

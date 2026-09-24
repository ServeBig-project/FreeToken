"""Record public text/history controls; the coordinator runs this on a fresh server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from check_http import api, connect, require, validate


def streamed(url, body, record):
    record["events"] = []
    pieces, finish, usage, done = [], None, None, False
    with connect(url, "/v1/completions", dict(body, stream=True)) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            record["events"].append(event)
            require("error" not in event, f"stream error: {event}")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                pieces.append(choice.get("text", ""))
                finish = choice.get("finish_reason") or finish
    record["summary"] = {"text": "".join(pieces), "finish_reason": finish, "usage": usage}
    require(done and finish is not None, "stream did not terminate cleanly")


def main(args):
    url = args.url.rstrip("/")
    model = api(url, "/v1/models")["data"][0]["id"]
    common = {"model": model, "temperature": 0, "top_k": -1, "top_p": 1.0}
    primary = dict(common, prompt="The sequence is: 2, 4, 6, 8,", max_tokens=8)
    history = [
        dict(common, prompt="A short sentence about the ocean:\n", max_tokens=1),
        dict(common, prompt="Translate 'good morning' into French:\n", max_tokens=2),
        dict(common, prompt="The capital of Japan is", max_tokens=4),
    ]
    plan = [("cold", args.first, primary)]
    plan += [(f"history-{index}", "plain", body) for index, body in enumerate(history, 1)]
    sampled = [dict(body, temperature=0.7 + index * 0.1, top_k=20, top_p=0.9,
                    max_tokens=limit)
               for index, (body, limit) in enumerate(zip([primary, *history], (1, 2, 3, 5)))]
    plan.append(("sampled-history", "sampled", sampled))
    plan += [(f"repeat-{index}", kind, primary)
             for index, kind in enumerate(("plain", "plain", "stream", "stream", "plain"), 1)]
    report = {"url": url, "first": args.first, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, kind, body in plan:
        record = {"label": label, "kind": kind, "request": body,
                  "before": api(url, "/v1/stats")}
        report["records"].append(record)
        try:
            if kind == "stream":
                streamed(url, body, record)
            elif kind == "sampled":
                with ThreadPoolExecutor(max_workers=4) as pool:
                    record["responses"] = list(pool.map(
                        lambda item: api(url, "/v1/completions", item), body))
                record["summary"] = [validate(response, item["max_tokens"])
                                     for response, item in zip(record["responses"], body)]
            else:
                record["response"] = api(url, "/v1/completions", body)
                record["summary"] = validate(record["response"], body["max_tokens"])
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        record["after"] = api(url, "/v1/stats")
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"label": label, "kind": kind, "summary": record["summary"],
                          "speculative": record["after"]["speculative"]},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first", choices=("plain", "stream"), default="plain")
    main(parser.parse_args())

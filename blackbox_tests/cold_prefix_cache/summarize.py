"""Print pass/fail counts per launch and every failed check from results.jsonl."""

import collections
import json
import sys

path = sys.argv[1]
counts = collections.defaultdict(lambda: [0, 0, 0])
failures = []
for line in open(path):
    row = json.loads(line)
    if "ok" not in row:
        continue
    label = row["check"].split("/")[0]
    slot = {True: 0, False: 1, None: 2}[row["ok"]]
    counts[label][slot] += 1
    if row["ok"] is False:
        failures.append(row)
for label, (ok, bad, notes) in counts.items():
    print(f"{label:40s} pass={ok:4d} fail={bad:3d} notes={notes:3d}")
for row in failures:
    print(json.dumps(row)[:700])

"""Public-contract test helpers; exact comparisons have no tolerance."""

import argparse
import json
import struct
from pathlib import Path

import torch


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def exact(actual, expected, label):
    actual = actual.detach().cpu()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: expected {expected.shape}/{expected.dtype}, "
                             f"got {actual.shape}/{actual.dtype}")
    mismatch = actual != expected
    if mismatch.any():
        index = tuple(mismatch.nonzero()[0].tolist())
        raise AssertionError(f"{label}: {mismatch.sum().item()} mismatches; "
                             f"at {index}: expected {expected[index].item()}, "
                             f"got {actual[index].item()}")


def storage_bytes(tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def run_cases(phase, cases, default_device="cpu"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default=default_device)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = []
    for name, run in cases:
        try:
            details = run(args.device)
            result = {"name": name, "status": "passed", **(details or {})}
        except Exception as error:
            message = str(error).strip().splitlines()
            result = {"name": name, "status": "failed", "error_type": type(error).__name__,
                      "error": message[0] if message else ""}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {"phase": phase, "device": args.device, "results": results}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return int(any(result["status"] == "failed" for result in results))

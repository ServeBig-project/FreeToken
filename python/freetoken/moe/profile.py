"""Offline expert-frequency records and explicit GPU-resident expert lists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_resident_experts(path: str, num_layers: int, num_experts: int) -> tuple[tuple[int, int], ...]:
    pairs = json.loads(Path(path).read_text())["gpu_experts"]
    if not isinstance(pairs, list) or any(
        not isinstance(pair, list) or len(pair) != 2
        or any(type(value) is not int for value in pair)
        or not 0 <= pair[0] < num_layers or not 0 <= pair[1] < num_experts
        for pair in pairs
    ):
        raise ValueError("gpu_experts must contain valid [layer, expert] integer pairs")
    result = tuple(map(tuple, pairs))
    if len(set(result)) != len(result):
        raise ValueError("gpu_experts contains duplicate pairs")
    return result


def write_profile(path: str, counts) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {"num_layers": counts.shape[0], "num_experts": counts.shape[1],
            "counts": counts.cpu().tolist()}
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(data) + "\n")
    temporary.replace(destination)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.profile).read_text())
    layers, experts, counts = data["num_layers"], data["num_experts"], data["counts"]
    if (type(layers) is not int or type(experts) is not int or layers < 1 or experts < 1
            or not isinstance(counts, list) or len(counts) != layers
            or any(not isinstance(row, list) or len(row) != experts
                   or any(type(n) is not int or n < 0 for n in row) for row in counts)):
        parser.error("profile counts must be a nonnegative integer matrix matching model dimensions")
    if not 0 <= args.count <= layers * experts:
        parser.error("count must be between zero and the number of model experts")
    ranked = sorted(((-n, layer, expert) for layer, row in enumerate(counts)
                     for expert, n in enumerate(row)))
    selected = [[layer, expert] for _, layer, expert in ranked[:args.count]]
    Path(args.output).write_text(json.dumps({"gpu_experts": selected}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

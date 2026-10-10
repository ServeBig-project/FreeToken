"""Reproduce the numbers behind tolerances.BOUNDS (reference only, no candidate code).

python blackbox_tests/nowag_runtime/calibrate.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import reference as R  # noqa: E402
from cases import FAMILIES  # noqa: E402


def main():
    torch.set_num_threads(8)
    for name, (hidden, inter, top_k, math_) in FAMILIES.items():
        for d in (4, 6):
            g = torch.Generator().manual_seed(d)
            cb = R.random_codebook(d, g)
            bank = [R.random_expert(hidden, inter, d, g, bias=math_.get("bias", False))
                    for _ in range(4)]
            x = torch.randn(16, hidden, generator=g).bfloat16()
            rows = torch.randint(0, 4, (16, top_k), generator=g).int()
            rw = torch.rand(16, top_k, generator=g)
            ref = R.moe(x, rows, rw, bank, cb, math_)
            legal = R.moe(x, rows, rw, bank, cb, math_, gate_up_bf16=True).bfloat16().float()
            err = legal - ref
            line = (f"{name} D{d} legal: rel_fro {float(err.norm() / ref.norm()):.2e} "
                    f"max/peak {float(err.abs().max() / ref.abs().max()):.2e}")
            if math_.get("dsv4_round"):
                for label, wrong in (("no-round", dict(math_, dsv4_round=False)),
                                     ("route-output", dict(math_, route="output"))):
                    w = R.moe(x, rows, rw, bank, cb, wrong).bfloat16().float()
                    line += f" | {label} {float((w - ref).norm() / ref.norm()):.2e}"
            print(line)


if __name__ == "__main__":
    main()

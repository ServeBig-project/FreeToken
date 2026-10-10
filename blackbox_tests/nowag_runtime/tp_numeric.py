"""Two-process public bind + NCCL reduction, compared with independent full expert math.

Launched by test_tp_numeric.py only after two devices are explicitly assigned. The per-rank
recipe was supplied as public contract by the coordinator on 2026-10-09.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import reference as R
import sidecar as S
import tiny_model as tiny
import tolerances as TOL
from cases import FAMILIES, need_tp2
from access import expert_math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--side", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--math", choices=["silu", "dsv4"], default="silu")
    args = parser.parse_args()
    need_tp2()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    from freetoken.distributed import set_tp_info
    from freetoken.distributed.info import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.moe.expert_banks import load_expert_banks
    import freetoken.moe.expert_format as F

    set_tp_info(rank=rank, size=2)
    device = torch.device("cuda", local_rank)
    cfg = EngineConfig(model_path=args.base, nowag_expert_path=args.side,
                       tp_info=DistributedInfo(rank, 2), dtype=torch.bfloat16)
    loaded = load_expert_banks(args.base, cfg.model_config, device=device, dtype=torch.bfloat16)
    banks = {name: layers[0].to(device).contiguous() for name, layers in loaded.sources.items()}
    shared = {name: value.to(device) for name, value in loaded.shared.items()}
    config = json.loads((Path(args.base) / "config.json").read_text())
    hidden, inter = config["hidden_size"], config["moe_intermediate_size"]
    top_k = config["num_experts_per_tok"]
    math = FAMILIES["dsv4"][3] if args.math == "dsv4" else {"family": "silu", "route": "output"}
    method = F.bind_expert_method(expert_math(F, math),
                                  F.ExpertLayout("nowag", hidden, inter, tiny.EXPERTS),
                                  loaded.format_state, device=device, backend="offload")
    codebook = S.codebook(args.side)
    weights = S.read_experts(args.side, 0, range(tiny.EXPERTS))
    results = []
    for rows_count in (1, 4, 16, 7):
        gen = torch.Generator().manual_seed(2000 + rows_count)
        x = torch.randn(rows_count, hidden, generator=gen).bfloat16()
        rows = torch.stack([torch.randperm(tiny.EXPERTS, generator=gen)[:top_k]
                            for _ in range(rows_count)]).int()
        route = torch.softmax(torch.randn(rows_count, top_k, generator=gen), -1)
        if rows_count == 7:
            rows[-2:] = -1
        workspace = {name: torch.full(shape, float("nan") if dtype.is_floating_point else -7,
                                     dtype=dtype, device=device)
                     for name, (shape, dtype) in method.workspace_spec(
                         rows_count, top_k, bank_rows=tiny.EXPERTS).items()}
        out = torch.full((rows_count, hidden), float("nan"), dtype=torch.bfloat16, device=device)
        returned = method.run(x.to(device), rows.to(device), route.to(device), banks, shared,
                              workspace=workspace, out=out)
        assert returned.data_ptr() == out.data_ptr()
        contribution_norm = float(out.float().norm())
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        expected = R.moe(x, rows, route, weights, codebook, math)
        metrics = TOL.assert_close(out.cpu(), expected, math, f"TP2 rank {rank} rows {rows_count}")
        results.append({"rows": rows_count, "local_norm": contribution_norm, "metrics": metrics})
    record = {"rank": rank, "local_rank": local_rank, "device": str(device),
              "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "world_size": dist.get_world_size(),
              "backend": dist.get_backend(), "all_reduce_calls": len(results), "results": results}
    Path(f"{args.result}.rank{rank}.json").write_text(json.dumps(record, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

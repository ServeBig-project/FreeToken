"""Public call contract of bind_expert_method(...).run / .workspace_spec (contract §5, §2.2).

impl "reference" exercises every test body against the independent reference (always runs);
"cpu"/"cuda" bind the candidate (see access.candidate; CUDA additionally needs NOWAG_GPU_OK=1
and CUDA_VISIBLE_DEVICES set to the approved GPU).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import access as A  # noqa: E402
import tolerances as TOL  # noqa: E402
from cases import need_gpu  # noqa: E402

IMPLS = ["reference", "cpu", "cuda"]
NUMERIC = ["qwen36-d6-real", "dsv4-d6-real", "qwen36-d4-random", "qwen36-d6-wordmajor",
           "gptoss-d6-random", "comp-dsv4math-qwen36", "comp-swigluoai-qwen36",
           "comp-gelutanh-qwen36"]
EXACT = ["qwen36-d6-exact"]
BUFFER = ["qwen36-d6-real", "comp-dsv4math-qwen36"]   # buffer/purity checks: one plain, one DSV4 math
_BOUND = {}


def bind(impl, name, layer_pos="first"):
    key = (impl, name, layer_pos)
    if key not in _BOUND:
        if impl == "reference":
            _BOUND[key] = A.reference(name, layer_pos)
        else:
            if impl == "cuda":
                need_gpu()
            _BOUND[key] = A.candidate(name, layer_pos, impl)
    return _BOUND[key]


def garbage(spec, device):
    """Workspace per spec, deliberately non-zero (contract: must not rely on zeroed buffers)."""
    ws = {}
    for name, (shape, dtype) in spec.items():
        if dtype.is_floating_point:
            ws[name] = torch.full(shape, float("nan"), dtype=dtype, device=device)
        else:
            ws[name] = torch.full(shape, -7, dtype=dtype, device=device)
    return ws


def inputs(b, t, seed, valid_slots=None):
    g = torch.Generator().manual_seed(seed)
    n_bank = len(b.bank_experts)
    pool = torch.randperm(n_bank, generator=g)[:max(b.top_k + 4, 12)]
    rows = torch.stack([pool[torch.randperm(len(pool), generator=g)[:b.top_k]] for _ in range(t)]) \
        if t else torch.zeros(0, b.top_k, dtype=torch.long)
    x = torch.randn(t, b.hidden, generator=g).bfloat16()
    rw = torch.softmax(torch.randn(t, b.top_k, generator=g), -1).float()
    return x, rows.int(), rw


def call(b, x, rows, rw, rows_capacity=None, out=None, workspace=None):
    dev = b.device
    x, rows, rw = x.to(dev).contiguous(), rows.to(dev).contiguous(), rw.to(dev).contiguous()
    spec = b.method.workspace_spec(rows_capacity or x.shape[0], b.top_k,
                                   bank_rows=len(b.bank_experts))
    ws = workspace if workspace is not None else garbage(spec, dev)
    if out is None:
        out = torch.full((x.shape[0], b.hidden), float("nan"), dtype=torch.bfloat16, device=dev)
    ret = b.method.run(x, rows, rw, b.banks, b.shared, workspace=ws, out=out)
    return ret, out


def snapshot(b, *tensors):
    every = list(tensors) + list(b.banks.values()) + list(b.shared.values())
    return [t.detach().clone() for t in every if torch.is_tensor(t)]


# ------------------------------------------------------------------ workspace_spec

@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", BUFFER)
def test_workspace_spec_is_pure_and_well_formed(impl, name):
    b = bind(impl, name)
    before_io = Path("/proc/self/io").read_text()
    alloc = torch.cuda.memory_allocated() if b.device.type == "cuda" else 0
    reserved = torch.cuda.memory_reserved() if b.device.type == "cuda" else 0
    specs = [b.method.workspace_spec(rows, b.top_k, bank_rows=len(b.bank_experts))
             for rows in (1, 16, 1, 128)]
    after_io = Path("/proc/self/io").read_text()
    if b.device.type == "cuda":
        assert torch.cuda.memory_allocated() == alloc and torch.cuda.memory_reserved() == reserved
    rchar = lambda s: int(s.split("rchar:")[1].split()[0])
    # a layer file is >=200 MB; 1 MiB leaves room for interpreter noise but not a weight read
    assert rchar(after_io) - rchar(before_io) < 1 << 20
    assert specs[0] == specs[2], "same query must give the same answer"
    for spec in specs:
        assert isinstance(spec, dict)
        for key, value in spec.items():
            shape, dtype = value
            assert isinstance(key, str) and isinstance(dtype, torch.dtype)
            assert isinstance(shape, tuple) and all(isinstance(v, int) and v >= 0 for v in shape)


# ------------------------------------------------------------------ numerics

@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", NUMERIC)
@pytest.mark.parametrize("t", [1, 4, 16, 7])
def test_run_matches_reference(impl, name, t):
    b = bind(impl, name)
    x, rows, rw = inputs(b, t, seed=t)
    ret, out = call(b, x, rows, rw)
    assert ret.data_ptr() == out.data_ptr() and ret.shape == (t, b.hidden)
    assert ret.dtype == torch.bfloat16 and ret.device == out.device
    ref = b.expected(x, rows, rw)
    TOL.assert_close(ret.cpu(), ref, b.math, f"{impl}/{name}/T={t}")


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", NUMERIC)
def test_rounding_and_route_position_are_the_models(impl, name):
    """DSV4 keeps both E4M3 roundings and the route weight before the down rounding; other
    models must not pick up the E4M3 rounding."""
    b = bind(impl, name)
    x, rows, rw = inputs(b, 16, seed=99)
    got = call(b, x, rows, rw)[0].cpu()
    ref = b.expected(x, rows, rw)
    if b.math.get("dsv4_round"):
        TOL.assert_discriminates(got, ref, b.expected(x, rows, rw, dict(b.math, dsv4_round=False)),
                                 "E4M3 rounding")
        TOL.assert_discriminates(got, ref, b.expected(x, rows, rw, dict(b.math, route="output")),
                                 "route weight position")
    elif b.hidden % 128 == 0 and b.inter % 128 == 0:
        TOL.assert_discriminates(got, ref, b.expected(x, rows, rw, dict(b.math, dsv4_round=True)),
                                 "no extra rounding")


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("layer_pos", ["first", "last"])
def test_layer_mapping_uses_that_layers_weights(impl, layer_pos):
    b = bind(impl, "qwen36-d6-real", layer_pos)
    x, rows, rw = inputs(b, 8, seed=5)
    TOL.assert_close(call(b, x, rows, rw)[0].cpu(), b.expected(x, rows, rw), b.math, layer_pos)


def exact_inputs(b, t, seed):
    """Four lanes of x equal 1 (one of them the H tail lane), two non-zero route weights
    {1, 0.5}, the remaining slots weight 0. See sidecar.synth_expert("exact")."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(t, b.hidden)
    for r in range(t):
        lanes = torch.randperm(b.hidden - 1, generator=g)[:3].tolist() + [b.hidden - 1]
        x[r, lanes] = 1.0
    _, rows, _ = inputs(b, t, seed)
    rw = torch.zeros(t, b.top_k)
    rw[:, 0], rw[:, 1] = 1.0, 0.5
    return x.bfloat16(), rows, rw


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", EXACT)
@pytest.mark.parametrize("t", [1, 5, 16])
def test_exact_sample_bitwise(impl, name, t):
    b = bind(impl, name)
    x, rows, rw = exact_inputs(b, t, seed=t)
    ref = b.expected(x, rows, rw)
    assert torch.equal(ref.bfloat16().float(), ref), "exact design broken (reference)"
    assert ref.abs().max() > 0
    got = call(b, x, rows, rw)[0].cpu()
    bad = (got.float() != ref).nonzero()
    assert bad.numel() == 0, f"{len(bad)} values differ, first {bad[:3].tolist()}: " \
                             f"{got.float()[tuple(bad[0])]} vs {ref[tuple(bad[0])]}"


# ------------------------------------------------------------------ buffer contract

@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", BUFFER)
def test_dirty_buffers_and_repeat_give_identical_output(impl, name):
    b = bind(impl, name)
    x, rows, rw = inputs(b, 9, seed=3)
    first = call(b, x, rows, rw)[0].clone()
    spec = b.method.workspace_spec(9, b.top_k, bank_rows=len(b.bank_experts))
    ws = {k: torch.randn(s, device=b.device).to(dt) if dt.is_floating_point
          else torch.randint(-5, 5, s, device=b.device, dtype=dt) for k, (s, dt) in spec.items()}
    out = torch.randn(9, b.hidden, device=b.device).bfloat16() * 1e4
    second = call(b, x, rows, rw, out=out, workspace=ws)[0]
    assert torch.equal(first, second)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", BUFFER)
def test_inputs_and_weights_not_modified(impl, name):
    b = bind(impl, name)
    x, rows, rw = (t.to(b.device) for t in inputs(b, 6, seed=4))
    before = snapshot(b, x, rows, rw)
    call(b, x, rows, rw)
    after = snapshot(b, x, rows, rw)
    assert all(torch.equal(p, q) for p, q in zip(before, after))


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", BUFFER)
def test_empty_batch(impl, name):
    b = bind(impl, name)
    x, rows, rw = inputs(b, 0, seed=0)
    ret, out = call(b, x, rows, rw)
    assert ret.shape == (0, b.hidden) and ret.dtype == torch.bfloat16


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", BUFFER)
def test_invalid_routes_and_padding_rows(impl, name):
    """expert_rows == -1 contributes nothing; padded rows (all -1) come out exactly zero."""
    b = bind(impl, name)
    x, rows, rw = inputs(b, 16, seed=8)
    rows[:5, -2:] = -1                 # partially routed rows
    rows[10:] = -1                     # graph padding / empty tail of the batch
    ret = call(b, x, rows, rw, rows_capacity=16)[0].cpu()
    assert torch.equal(ret[10:], torch.zeros_like(ret[10:]))
    TOL.assert_close(ret[:10], b.expected(x[:10], rows[:10], rw[:10]), b.math, "partial rows")
    rows[:] = -1                       # empty valid sub-batch
    assert torch.equal(call(b, x, rows, rw)[0].cpu(), torch.zeros(16, b.hidden).bfloat16())


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", NUMERIC[:1])
def test_out_aliasing_x_is_correct_or_rejected_before_running(impl, name):
    b = bind(impl, name)
    x, rows, rw = (t.to(b.device) for t in inputs(b, 4, seed=6))
    keep = x.clone()
    ref = b.expected(keep, rows, rw)
    spec = b.method.workspace_spec(4, b.top_k, bank_rows=len(b.bank_experts))
    try:
        ret = b.method.run(x, rows, rw, b.banks, b.shared, workspace=garbage(spec, b.device), out=x)
    except Exception:
        assert torch.equal(x, keep), "rejected, but only after writing into x"
        return
    TOL.assert_close(ret.cpu(), ref, b.math, "aliased out")


# ------------------------------------------------------------------ unsupported math (§9)

UNSUPPORTED = {
    "route_on_gate_up_input": dict(activation="silu", router_weight_on_input=True),
    "erf_gelu": dict(activation="gelu"),
    "unknown_rounding": dict(activation="silu", down_input_rounding="int8_per_tensor"),
}


@pytest.mark.parametrize("impl", ["cpu", "cuda"])
@pytest.mark.parametrize("variant", sorted(UNSUPPORTED))
def test_unsupported_math_rejected_at_bind(impl, variant):
    """Contract §9: NoWAG does not support these; refuse at bind, before any run."""
    if impl == "cuda":
        need_gpu()
    import freetoken.moe.expert_format as F
    b = bind(impl, "qwen36-d6-real")
    banks = A.loaded_banks(str(b.base), str(b.side))
    with pytest.raises(Exception):
        F.bind_expert_method(F.ExpertMath(**UNSUPPORTED[variant]),
                             F.ExpertLayout("nowag", b.hidden, b.inter, len(b.bank_experts)),
                             banks.format_state, device=b.device, backend=impl if impl == "cpu" else "offload")


# ------------------------------------------------------------------ CUDA graph replay

@pytest.mark.parametrize("name", BUFFER + EXACT)
def test_graph_replay_changes_tokens_routes_and_tail(name):
    need_gpu()
    b = bind("cuda", name)
    cap = 16
    spec = b.method.workspace_spec(cap, b.top_k, bank_rows=len(b.bank_experts))
    ws = garbage(spec, b.device)
    x0, rows0, rw0 = inputs(b, cap, seed=20)
    sx, srows, srw = x0.cuda(), rows0.cuda(), rw0.cuda()
    out = torch.empty(cap, b.hidden, dtype=torch.bfloat16, device="cuda")
    b.method.run(sx, srows, srw, b.banks, b.shared, workspace=ws, out=out)     # warm-up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b.method.run(sx, srows, srw, b.banks, b.shared, workspace=ws, out=out)
    for step, valid in enumerate([16, 3, 16, 1, 9]):
        x, rows, rw = (exact_inputs(b, cap, 30 + step) if "exact" in name
                       else inputs(b, cap, 30 + step))
        rows[valid:] = -1                         # natural tail: padded rows
        sx.copy_(x), srows.copy_(rows), srw.copy_(rw)
        graph.replay()
        torch.cuda.synchronize()
        got = out.cpu()
        assert torch.equal(got[valid:], torch.zeros_like(got[valid:])), f"replay {step} kept residue"
        ref = b.expected(x[:valid], rows[:valid], rw[:valid])
        if "exact" in name:
            assert torch.equal(got[:valid].float(), ref), f"replay {step}"
        else:
            TOL.assert_close(got[:valid], ref, b.math, f"replay {step}")

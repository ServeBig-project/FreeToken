"""Independent NoWAG reference, written from public contract §2 only.

Nothing here imports production code. Every function is plain torch on CPU so the
expected values do not share a code path with the implementation under test.
"""

import math

import torch

CODEBOOK_SIZE = 4096
BITS = 12
E4M3_MAX = 448.0
DSV4_GROUP = 128


# ---------------------------------------------------------------- assignment packing

def ids_per_row(k, d):
    return math.ceil(k / d)


def words_per_row(k, d):
    return math.ceil(ids_per_row(k, d) * BITS / 32)


def pack(ids):
    """[N, n_ids] ids in [0,4096) -> row_major int32 [N, words]; 12-bit LSB-first stream per row."""
    n, count = ids.shape
    assert int(ids.min()) >= 0 and int(ids.max()) < CODEBOOK_SIZE
    words = math.ceil(count * BITS / 32)
    bits = (ids.long().unsqueeze(-1) >> torch.arange(BITS)) & 1           # [N, n, 12], bit0 first
    bits = bits.reshape(n, count * BITS)
    bits = torch.nn.functional.pad(bits, (0, words * 32 - count * BITS))
    bits = bits.reshape(n, words, 32)
    value = (bits << torch.arange(32)).sum(-1)                            # uint32 value in int64
    value = torch.where(value >= 2 ** 31, value - 2 ** 32, value)
    return value.to(torch.int32)


def unpack(packed, count):
    """row_major int32 [N, words] -> [N, count] int64 ids."""
    n, words = packed.shape
    value = packed.long() & 0xFFFFFFFF
    bits = (value.unsqueeze(-1) >> torch.arange(32)) & 1                  # [N, words, 32]
    bits = bits.reshape(n, words * 32)[:, :count * BITS].reshape(n, count, BITS)
    return (bits << torch.arange(BITS)).sum(-1)


def to_layout(packed_row_major, layout):
    if layout == "row_major":
        return packed_row_major
    if layout == "word_major":
        return packed_row_major.t().contiguous()
    raise ValueError(layout)


def to_row_major(packed, layout):
    return packed if layout == "row_major" else packed.t().contiguous()


# ---------------------------------------------------------------- dequantised weight

def codeword_matrix(packed, codebook, k, layout="row_major"):
    """C[A] as fp32 [N, K]; tail lanes past K are dropped (they never meet an input)."""
    d = codebook.shape[1]
    rows = to_row_major(packed, layout)
    ids = unpack(rows, ids_per_row(k, d))
    return codebook.float()[ids].reshape(rows.shape[0], -1)[:, :k]


def projection(x, proj, codebook, layout="row_major"):
    """Unrounded single projection ((x*in_norm) @ C[A].T) * out_norm + bias, fp32."""
    k = proj["input_norm"].shape[0]
    w = codeword_matrix(proj["assignments"], codebook, k, layout)
    y = (x.float() * proj["input_norm"].float()) @ w.t() * proj["output_norm"].float()
    if proj.get("bias") is not None:
        y = y + proj["bias"].float()
    return y


def effective_weight(proj, codebook, layout="row_major"):
    """Dense fp32 weight W such that projection(x) == x @ W.T (+bias)."""
    k = proj["input_norm"].shape[0]
    w = codeword_matrix(proj["assignments"], codebook, k, layout)
    return w * proj["input_norm"].float()[None, :] * proj["output_norm"].float()[:, None]


# ---------------------------------------------------------------- DSV4 E4M3 group rounding

def e4m3_group_round(x, group=DSV4_GROUP):
    """Fake-quant as in DeepSeek-V4 public inference act_quant: per `group` lanes,
    scale = 2**ceil(log2(max(amax,1e-4)/448)), q = e4m3(clamp(x/scale)), return q*scale (fp32)."""
    x = x.float()
    n = x.shape[-1]
    assert n % group == 0, "DSV4 rounding groups must tile the lane count"
    g = x.reshape(*x.shape[:-1], n // group, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)))
    q = (g / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(x.shape)


# ---------------------------------------------------------------- expert math families

def activation(gate, up, family, alpha=None, limit=None):
    """Combine gate/up outputs (fp32). Families are distinct references (contract §2.2)."""
    if family == "silu":
        return torch.nn.functional.silu(gate) * up
    if family == "gelu":
        return torch.nn.functional.gelu(gate) * up
    if family == "gelu_tanh":
        return torch.nn.functional.gelu(gate, approximate="tanh") * up
    if family == "swiglu_limit":                        # DeepSeek-V4: clamp then SiLU
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
        return torch.nn.functional.silu(gate) * up
    if family == "gptoss":                              # GPT-OSS: clamp, alpha-sigmoid, (up+1)
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
        return (up + 1) * (gate * torch.sigmoid(gate * alpha))
    raise ValueError(family)


def expert(x, w, codebook, math_, route_weight=None, layout="row_major", gate_up_bf16=False):
    """One expert on rows x [T,H]. route_weight [T] applied where the model puts it.

    math_: dict(family, alpha, limit, route="output"|"down_input"|"input", dsv4_round=bool)
    gate_up_bf16: round gate/up outputs to bf16 (a legal kernel choice; used only to size
    tolerances, never as the expected value).
    """
    x = x.float()
    route = math_.get("route", "output")
    if route == "input" and route_weight is not None:
        x = x * route_weight[:, None]
    if math_.get("dsv4_round"):
        x = e4m3_group_round(x)
    gate = projection(x, w["w1"], codebook, layout)
    up = projection(x, w["w3"], codebook, layout)
    if gate_up_bf16:
        gate, up = gate.bfloat16().float(), up.bfloat16().float()
    h = activation(gate, up, math_["family"], math_.get("alpha"), math_.get("limit"))
    if route == "down_input" and route_weight is not None:
        h = h * route_weight[:, None]
    if math_.get("dsv4_round"):
        h = e4m3_group_round(h.bfloat16())                # public code casts to bf16 first
    y = projection(h, w["w2"], codebook, layout)
    if route == "output" and route_weight is not None:
        y = y * route_weight[:, None]
    return y


def moe(x, expert_rows, route_weights, bank, codebook, math_, layout="row_major",
        gate_up_bf16=False):
    """Routed-expert sum. bank: list of expert weight dicts indexed by expert_rows.
    expert_rows == -1 contributes nothing. Returns fp32 [T,H]."""
    out = torch.zeros(x.shape, dtype=torch.float32)
    for e in sorted(set(expert_rows.flatten().tolist()) - {-1}):
        row, slot = torch.where(expert_rows == e)
        y = expert(x[row], bank[e], codebook, math_, route_weights[row, slot].float(), layout,
                   gate_up_bf16)
        out.index_add_(0, row, y)
    return out


# ---------------------------------------------------------------- tensor parallel split

def tp_shard(w, rank, tp):
    """Rank's logical slice of one expert: gate/up output rows, down input columns.

    The returned w2 keeps the full packed rows plus the lane window [lo,hi); codewords that
    straddle the boundary contribute only their in-window lanes (contract §4)."""
    inter = w["w1"]["output_norm"].shape[0]
    assert inter % tp == 0
    lo, hi = rank * inter // tp, (rank + 1) * inter // tp
    shard = {}
    for p in ("w1", "w3"):
        q = w[p]
        shard[p] = {"assignments": q["assignments"][lo:hi], "input_norm": q["input_norm"],
                    "output_norm": q["output_norm"][lo:hi],
                    "bias": None if q.get("bias") is None else q["bias"][lo:hi]}
    shard["w2"] = dict(w["w2"], window=(lo, hi), bias=w["w2"].get("bias") if rank == 0 else None)
    return shard


def expert_tp(x, shard, codebook, math_, route_weight=None):
    """Per-rank contribution of one expert (row_major only); sum over ranks == expert()."""
    x = x.float()
    route = math_.get("route", "output")
    if route == "input" and route_weight is not None:
        x = x * route_weight[:, None]
    if math_.get("dsv4_round"):
        x = e4m3_group_round(x)
    gate = projection(x, shard["w1"], codebook)
    up = projection(x, shard["w3"], codebook)
    h = activation(gate, up, math_["family"], math_.get("alpha"), math_.get("limit"))
    if route == "down_input" and route_weight is not None:
        h = h * route_weight[:, None]
    if math_.get("dsv4_round"):
        h = e4m3_group_round(h.bfloat16())
    w2 = shard["w2"]
    lo, hi = w2["window"]
    k = w2["input_norm"].shape[0]
    w = codeword_matrix(w2["assignments"], codebook, k)[:, lo:hi]
    y = (h * w2["input_norm"].float()[lo:hi]) @ w.t() * w2["output_norm"].float()
    if w2.get("bias") is not None:
        y = y + w2["bias"].float()
    if route == "output" and route_weight is not None:
        y = y * route_weight[:, None]
    return y


# ---------------------------------------------------------------- random legal weights

def random_codebook(d, gen, scale=0.05):
    return (torch.randn(CODEBOOK_SIZE, d, generator=gen) * scale).bfloat16()


def random_projection(n, k, d, gen, bias=False, layout="row_major"):
    ids = torch.randint(0, CODEBOOK_SIZE, (n, ids_per_row(k, d)), generator=gen)
    return {
        "assignments": to_layout(pack(ids), layout),
        "input_norm": (torch.rand(k, generator=gen) * 1.5 + 0.25).bfloat16(),
        "output_norm": (torch.rand(n, generator=gen) * 1.5 + 0.25).bfloat16(),
        "bias": (torch.randn(n, generator=gen) * 0.1).bfloat16() if bias else None,
    }


def random_expert(hidden, inter, d, gen, bias=False, layout="row_major"):
    """w1=gate [I,H], w3=up [I,H], w2=down [H,I] (contract §2.1)."""
    return {"w1": random_projection(inter, hidden, d, gen, bias, layout),
            "w3": random_projection(inter, hidden, d, gen, bias, layout),
            "w2": random_projection(hidden, inter, d, gen, bias, layout)}

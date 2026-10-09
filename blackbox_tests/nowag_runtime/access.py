"""How a test obtains a bound expert method: the candidate's public entry, or the reference.

Candidate: the recipe of contract §9 -- EngineConfig(model_path=BASE, nowag_expert_path=SIDE,
tp_info=DistributedInfo(0, 1)), load_expert_banks(...), banks.sources[name][l] for MoE layer
l = decoder layer - first_k_dense_replace, banks.shared, banks.format_state,
ExpertLayout("nowag", H, I, E), backend = the --moe-backend value; math is an ExpertMath (§5).

The "reference" implementation follows the same call contract on CPU so every test body is
exercised against the independent reference too; it validates the tests, not the product.
"""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import reference as R
import sidecar as S
from cases import (FAMILIES, QWEN36_BASE, QWEN36_SIDE, DSV4_BASE, DSV4_SIDE, GPTOSS_BASE,
                   need_path, need_scratch)

# name: (geometry family, math family, D, kind, assignment layout, base, real sidecar/geometry)
# "comp-*" cases run another published math family on the Qwen3.6 weights: a component-level
# check of that math (contract §3), not a claim about a served model.
CASES = {
    "qwen36-d6-real": ("qwen36_silu", "qwen36_silu", 6, "real", "row_major", QWEN36_BASE, QWEN36_SIDE),
    "dsv4-d6-real": ("dsv4", "dsv4", 6, "real", "row_major", DSV4_BASE, DSV4_SIDE),
    "qwen36-d4-random": ("qwen36_silu", "qwen36_silu", 4, "random", "row_major", QWEN36_BASE, QWEN36_SIDE),
    "qwen36-d6-wordmajor": ("qwen36_silu", "qwen36_silu", 6, "random", "word_major", QWEN36_BASE,
                            QWEN36_SIDE),
    "gptoss-d6-random": ("gptoss", "gptoss", 6, "random", "row_major", GPTOSS_BASE, None),
    "qwen36-d6-exact": ("qwen36_silu", "qwen36_silu", 6, "exact", "row_major", QWEN36_BASE, QWEN36_SIDE),
    "comp-dsv4math-qwen36": ("qwen36_silu", "dsv4", 6, "real", "row_major", QWEN36_BASE, QWEN36_SIDE),
    "comp-swigluoai-qwen36": ("qwen36_silu", "gptoss", 6, "real", "row_major", QWEN36_BASE,
                              QWEN36_SIDE),
    "comp-gelutanh-qwen36": ("qwen36_silu", "gelu_tanh", 6, "real", "row_major", QWEN36_BASE,
                             QWEN36_SIDE),
}


class Bound:
    """method + the tensors to pass + a way to rebuild bank row weights for the reference."""

    def __init__(self, name, method, banks, shared, bank_experts, layer, device, weights_of):
        self.name, self.method, self.banks, self.shared = name, method, banks, shared
        self.bank_experts, self.layer, self.device = bank_experts, layer, device
        geom, mathf, self.d, self.kind, self.layout, self.base, self.side = CASES[name]
        self.hidden, self.inter, self.top_k, _ = FAMILIES[geom]
        self.math = FAMILIES[mathf][3]
        self._weights_of = weights_of
        self._cache = {}

    def weights(self, row):
        e = self.bank_experts[row]
        if e not in self._cache:   # decode once; "dense" is C[A] cropped to K (reference.projection)
            w, cb = self._weights_of(e), self.codebook()
            self._cache[e] = {p: dict(q, dense=R.codeword_matrix(q["assignments"], cb,
                                                                 q["input_norm"].shape[0]))
                              for p, q in w.items()}
        return self._cache[e]

    def codebook(self):
        if "codebook" not in self._cache:
            self._cache["codebook"] = self._weights_of("codebook")
        return self._cache["codebook"]

    def expected(self, x, rows, rw, math_=None):
        """fp32 reference; math_ overrides the family math (wrong-variant comparisons)."""
        used = sorted(set(rows.flatten().tolist()) - {-1})
        bank = {r: self.weights(r) for r in used}
        return R.moe(x.cpu(), rows.cpu(), rw.cpu(), bank, self.codebook(), math_ or self.math)


# ------------------------------------------------------------------ public weight sources

@lru_cache(maxsize=None)
def gptoss_bias(base, layer):
    index = json.loads((Path(base) / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.layers.{layer}.mlp.experts."
    out = {}
    for name in ("gate_up_proj_bias", "down_proj_bias"):
        with safe_open(str(Path(base) / index[pre + name]), "pt") as f:
            out[name] = f.get_tensor(pre + name)
    return out


def with_base_bias(w, base, layer, e):
    b = gptoss_bias(str(base), layer)
    gate_up = b["gate_up_proj_bias"][e]                      # HF interleave: gate ::2, up 1::2
    return {"w1": dict(w["w1"], bias=gate_up[::2]), "w3": dict(w["w3"], bias=gate_up[1::2]),
            "w2": dict(w["w2"], bias=b["down_proj_bias"][e])}


def sidecar_dir(name):
    """On-disk NoWAG directory for a case (synthetic ones are generated under NOWAG_SCRATCH)."""
    geom, _, d, kind, layout, base, src = CASES[name]
    if kind == "real":
        return need_path(src, f"{name} sidecar")
    g = (S.gptoss_geometry(need_path(base, f"{name} base")) if geom == "gptoss"
         else S.geometry(need_path(src, f"{name} geometry source")))
    return S.synth_dir(g, need_scratch() / f"synth-{geom}-d{d}-{kind}-{layout}", d, kind, layout)


def file_weights(name, side, layer):
    geom, base = CASES[name][0], CASES[name][5]

    def weights_of(e):
        if e == "codebook":
            return S.codebook(side)
        w = S.read_experts(side, layer, [e])[e]
        return with_base_bias(w, base, layer, e) if geom == "gptoss" else w
    return weights_of


def layer_numbers(side, pos):
    layers = [e["layer"] for e in S.manifest(side)["layers"]]
    return layers[0] if pos == "first" else layers[-1]


# ------------------------------------------------------------------ candidate (contract §9)

def expert_math(F, family):
    """ExpertMath for a math family (field names from the public ExpertMath signature, §5/§9)."""
    m = FAMILIES[family][3]
    if m["family"] == "silu":
        return F.ExpertMath(activation="silu")
    if m["family"] == "gptoss":
        return F.ExpertMath(activation="swigluoai", activation_alpha=m["alpha"],
                            activation_limit=m["limit"])
    if m["family"] == "swiglu_limit":   # DSV4: clamped SiLU, route on down input, E4M3 twice
        return F.ExpertMath(activation="silu", activation_limit=m["limit"],
                            router_weight_on_down_input=True,
                            gate_up_input_rounding=F.E4M3_GROUP128_UE8M0,
                            down_input_rounding=F.E4M3_GROUP128_UE8M0)
    pytest.skip(f"contract publishes no ExpertMath activation string for {m['family']}")


@lru_cache(maxsize=None)
def loaded_banks(base, side):
    from freetoken.distributed.info import DistributedInfo, set_tp_info, try_get_tp_info
    from freetoken.engine.config import EngineConfig
    from freetoken.moe.expert_banks import load_expert_banks
    # The §9 recipe alone raises "TP info has not been set" in load_expert_banks; set_tp_info
    # is the public setter next to DistributedInfo. Reported to the coordinator.
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    cfg = EngineConfig(model_path=str(base), nowag_expert_path=str(side),
                       tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16)
    return load_expert_banks(str(base), cfg.model_config, device=torch.device("cpu"),
                             dtype=torch.bfloat16)


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def candidate(name, layer_pos, impl):
    """Bind the candidate through its public entry; impl "cpu" -> backend cpu on CPU,
    "cuda" -> backend offload on cuda:0."""
    import freetoken.moe.expert_format as F
    geom, mathf, d, kind, layout, base, src = CASES[name]
    base = need_path(base, f"{name} base")
    side = sidecar_dir(name)
    math = expert_math(F, mathf)
    device = torch.device("cpu") if impl == "cpu" else torch.device("cuda", 0)
    backend = "cpu" if impl == "cpu" else "offload"
    hidden, inter, _, _ = FAMILIES[geom]
    config = json.loads((base / "config.json").read_text())
    first_dense = config.get("text_config", config).get("first_k_dense_replace", 0)
    layer = layer_numbers(side, layer_pos)
    banks = loaded_banks(str(base), str(side))
    experts = next(iter(banks.sources.values()))[layer - first_dense].shape[0]
    layer_banks = {k: v[layer - first_dense].to(device).contiguous() for k, v in banks.sources.items()}
    method = F.bind_expert_method(math, F.ExpertLayout("nowag", hidden, inter, experts),
                                  banks.format_state, device=device, backend=backend)
    return Bound(name, method, layer_banks, to_device(banks.shared, device), list(range(experts)),
                 layer, device, file_weights(name, side, layer))


# ------------------------------------------------------------------ reference implementation

class ReferenceMethod:
    def __init__(self, bound_ref):
        self.ref = bound_ref

    def workspace_spec(self, rows, top_k, bank_rows=None):
        return {"scratch": ((rows, self.ref.hidden), torch.float32),
                "slots": ((rows, top_k), torch.int32)}

    def run(self, x, expert_rows, route_weights, banks, shared, workspace=None, out=None):
        y = self.ref.expected(x, expert_rows, route_weights)
        out.copy_(y.to(out.dtype))
        return out


def reference(name, layer_pos, bank_size=12):
    geom, mathf, d, kind, layout, base, src = CASES[name]
    hidden, inter, _, geom_math = FAMILIES[geom]
    if kind == "real":
        side = need_path(src, f"{name} sidecar")
        layer = layer_numbers(side, layer_pos)
        weights_of = file_weights(name, side, layer)
        total = S.geometry(side)["experts"]
    else:
        layer, total = 0, 32

        def weights_of(e):
            if e == "codebook":
                return S.synth_codebook(kind, d, inter)
            w = S.synth_expert(kind, d, hidden, inter, layer, e)
            if geom_math.get("bias"):
                g = torch.Generator().manual_seed(e)
                w = {p: dict(q, bias=(torch.randn(q["output_norm"].shape[0], generator=g) * 0.1)
                             .bfloat16()) for p, q in w.items()}
            return w
    # a non-identity bank -> expert mapping, as a cache would have
    g = torch.Generator().manual_seed(7)
    bank_experts = torch.randperm(total, generator=g)[:bank_size].tolist()
    banks = {"rows": torch.tensor(bank_experts, dtype=torch.int32)}
    bound = Bound(name, None, banks, {}, bank_experts, layer, torch.device("cpu"), weights_of)
    bound.shared = {"codebook": weights_of("codebook")}
    bound.method = ReferenceMethod(bound)
    return bound

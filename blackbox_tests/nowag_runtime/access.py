"""How a test obtains a bound expert method: the candidate's public entry, or the reference.

Contract §5 fixes the call `freetoken.moe.expert_format.bind_expert_method(math, layout,
format_state, device=..., backend=...)` and the shape of `run`/`workspace_spec`, but not how a
black-box caller gets `math`, `layout`, `format_state`, the `banks` dict and the `shared` dict
from a model directory plus a NoWAG directory. `candidate()` therefore skips with that reason
until the coordinator publishes the recipe; only that function needs filling in.

The "reference" implementation below follows the same call contract on CPU so every test body
is exercised now; it validates the tests, not the product.
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

# name: (family, D, kind, base, real sidecar or geometry source)
CASES = {
    "qwen36-d6-real": ("qwen36_silu", 6, "real", QWEN36_BASE, QWEN36_SIDE),
    "dsv4-d6-real": ("dsv4", 6, "real", DSV4_BASE, DSV4_SIDE),
    "qwen36-d4-random": ("qwen36_silu", 4, "random", QWEN36_BASE, QWEN36_SIDE),
    "dsv4-d4-random": ("dsv4", 4, "random", DSV4_BASE, DSV4_SIDE),
    "gptoss-d6-random": ("gptoss", 6, "random", GPTOSS_BASE, None),
    "gptoss-d4-random": ("gptoss", 4, "random", GPTOSS_BASE, None),
    "qwen36-d6-exact": ("qwen36_silu", 6, "exact", QWEN36_BASE, QWEN36_SIDE),
    "qwen36-d4-exact": ("qwen36_silu", 4, "exact", QWEN36_BASE, QWEN36_SIDE),
}


class Bound:
    """method + the tensors to pass + a way to rebuild bank row weights for the reference."""

    def __init__(self, name, method, banks, shared, bank_experts, layer, device, weights_of):
        self.name, self.method, self.banks, self.shared = name, method, banks, shared
        self.bank_experts, self.layer, self.device = bank_experts, layer, device
        family, self.d, self.kind, self.base, self.side = CASES[name]
        self.hidden, self.inter, self.top_k, self.math = FAMILIES[family]
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
    family, d, kind, base, src = CASES[name]
    if kind == "real":
        return need_path(src, f"{name} sidecar")
    geom = (S.gptoss_geometry(need_path(base, f"{name} base")) if family == "gptoss"
            else S.geometry(need_path(src, f"{name} geometry source")))
    return S.synth_dir(geom, need_scratch() / f"synth-{name}", d, kind)


def file_weights(name, side, layer):
    family, d, kind, base, _ = CASES[name]

    def weights_of(e):
        if e == "codebook":
            return S.codebook(side)
        w = S.read_experts(side, layer, [e])[e]
        return with_base_bias(w, base, layer, e) if family == "gptoss" else w
    return weights_of


# ------------------------------------------------------------------ the two implementations

def candidate(name, layer_pos, device, backend):
    """Bind the candidate through its public entry. Recipe not yet published (see module doc)."""
    pytest.skip("contract §5 does not say how a black-box caller obtains math/layout/"
                "format_state/banks/shared for a model + NoWAG directory; ask coordinator")


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
    family, d, kind, base, src = CASES[name]
    hidden, inter, top_k, math_ = FAMILIES[family]
    if kind == "real":
        side = need_path(src, f"{name} sidecar")
        layers = [e["layer"] for e in S.manifest(side)["layers"]]
        layer = layers[0] if layer_pos == "first" else layers[-1]
        weights_of = file_weights(name, side, layer)
        total = S.geometry(side)["experts"]
    else:
        layer, total = 0, 32

        def weights_of(e):
            if e == "codebook":
                return S.synth_codebook(kind, d, inter)
            w = S.synth_expert(kind, d, hidden, inter, layer, e)
            if math_.get("bias"):
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

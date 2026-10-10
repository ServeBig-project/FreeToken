"""Read and write native NoWAG v1 directories (contract §2.1), independent of production code."""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import reference as R

PROJ = ("w1", "w2", "w3")


def manifest(side):
    return json.loads((Path(side) / "manifest.json").read_text())


def codebook(side):
    m = manifest(side)
    with safe_open(str(Path(side) / m["codebook"]["file"]), "pt") as f:
        return f.get_tensor(m["codebook"]["tensor"])


def key(layer, e, p, part):
    base = f"layers.{layer}.ffn.experts.{e}.{p}"
    return {"assignments": f"{base}.assignments", "input_norm": f"{base}.normalizer.norms.0",
            "output_norm": f"{base}.normalizer.norms.1"}[part]


def layer_entry(side, layer):
    for entry in manifest(side)["layers"]:
        if entry["layer"] == layer:
            return entry
    raise KeyError(layer)


def read_experts(side, layer, experts):
    """{expert: {w1/w2/w3: {assignments,input_norm,output_norm,bias=None}}}; assignments are
    returned row_major whatever the manifest's assignment_layout (contract §9)."""
    entry = layer_entry(side, layer)
    layout = manifest(side).get("assignment_layout", "row_major")
    out = {}
    with safe_open(str(Path(side) / entry["file"]), "pt") as f:
        for e in experts:
            out[e] = {p: {part: f.get_tensor(key(layer, e, p, part))
                          for part in ("assignments", "input_norm", "output_norm")} | {"bias": None}
                      for p in PROJ}
            for w in out[e].values():
                w["assignments"] = R.to_row_major(w["assignments"], layout)
    return out


def tensor_shapes(side, layer):
    entry = layer_entry(side, layer)
    with safe_open(str(Path(side) / entry["file"]), "pt") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def write_layer(out, layer, experts, layout="row_major"):
    """Write layer-LLL.safetensors/.json for {expert: weights} (row_major in memory, stored in
    `layout`); return the manifest entry."""
    tensors, matrices = {}, {}
    name = f"layer-{layer:03d}"
    for e, w in experts.items():
        for p in PROJ:
            meta = {}
            for field, part in (("packed_assignments", "assignments"),
                                ("input_norm", "input_norm"), ("output_norm", "output_norm")):
                t = w[p][part]
                t = (R.to_layout(t, layout) if part == "assignments" else t).contiguous()
                tensors[key(layer, e, p, part)] = t
                meta[field] = {"dtype": str(t.dtype).replace("torch.", ""),
                               "name": key(layer, e, p, part), "shape": list(t.shape)}
            matrices[f"layers.{layer}.ffn.experts.{e}.{p}.weight"] = {
                "expert": e, "file": f"{name}.safetensors", "projection": p, "tensors": meta,
                "logical_shape": [w[p]["output_norm"].shape[0], w[p]["input_norm"].shape[0]]}
    save_file(tensors, str(Path(out) / f"{name}.safetensors"))
    (Path(out) / f"{name}.json").write_text(json.dumps(
        {"file": f"{name}.safetensors", "layer": layer, "matrices": matrices}))
    return {"file": f"{name}.safetensors", "index": f"{name}.json", "layer": layer,
            "matrix_count": len(matrices)}


def write_head(out, template, d, cb, entries, layout="row_major"):
    """Codebook file + manifest. template: a real manifest whose model-geometry fields
    (format, model_type, hidden_size, num_experts, ...) are kept; calibration bookkeeping dropped."""
    save_file({"global_all.codebook": cb.contiguous()}, str(Path(out) / "global_codebook.safetensors"))
    keep = {k: v for k, v in template.items()
            if not k.startswith(("lloyd", "calibration", "initialize", "kmeans", "pool"))}
    keep.update({"d": d, "assignment_bits": 12, "assignments_packed": True, "scope": "expert_only",
                 "codebook_sharing": "global_all", "layers": entries, "assignment_layout": layout,
                 "matrix_count": sum(e["matrix_count"] for e in entries),
                 "n_bits": 12 / d, "normalizer_order": [0, 1], "normalizer_zero": [False, False],
                 "codebook": {"dtype": "bfloat16", "file": "global_codebook.safetensors",
                              "shape": list(cb.shape), "tensor": "global_all.codebook"}})
    (Path(out) / "manifest.json").write_text(json.dumps(keep, indent=1))


def geometry(real_side):
    """Model geometry and layer map of a real sidecar (used as the template for synthetic ones)."""
    m = manifest(real_side)
    layer = m["layers"][0]["layer"]
    shapes = tensor_shapes(real_side, layer)
    return {"template": m, "layers": [e["layer"] for e in m["layers"]],
            "experts": len({int(k.split(".")[4]) for k in shapes}),
            "hidden": shapes[key(layer, 0, "w1", "input_norm")][0],
            "inter": shapes[key(layer, 0, "w1", "output_norm")][0]}


def gptoss_geometry(base):
    """GPT-OSS has no published NoWAG artifact; the manifest uses the generic v1 geometry keys."""
    c = json.loads((Path(base) / "config.json").read_text())
    template = {"format": "nowag_expert_sidecar_v1", "model_type": c["model_type"],
                "hidden_size": c["hidden_size"], "moe_intermediate_size": c["intermediate_size"],
                "num_experts": c["num_local_experts"], "num_moe_layers": c["num_hidden_layers"]}
    return {"template": template, "layers": list(range(c["num_hidden_layers"])),
            "experts": c["num_local_experts"], "hidden": c["hidden_size"],
            "inter": c["intermediate_size"]}


def synth_codebook(kind, d, inter):
    if kind == "random":
        return R.random_codebook(d, torch.Generator().manual_seed(d))
    # "exact": a few dyadic codewords; everything else zero (see synth_expert)
    cb = torch.zeros(R.CODEBOOK_SIZE, d)
    cb[1] = 1.0
    cb[2] = torch.tensor([0.5, -0.5] * d)[:d]
    cb[3] = 0.5
    cb[4, 0] = 1.0
    cb[5, :2] = torch.tensor([0.5, 0.25])
    if inter % d:                          # lanes past I in the last down codeword: never used
        cb[5, inter % d:] = 64.0
    return cb.bfloat16()


def synth_expert(kind, d, hidden, inter, layer, e):
    """Deterministic per (layer, expert), so files and in-memory banks agree.

    kind "exact" (SiLU family, see test_bind_contract.exact_inputs): gate = 32 for every row
    when x has four lanes equal to 1, so SiLU(gate) == gate exactly; up in [-2,2]; the down row
    n reads lane D*j (j depends on n and e) plus the two valid lanes of the last codeword.
    All intermediate and final values are small dyadic numbers, exactly representable in BF16.
    """
    if kind == "random":
        return R.random_expert(hidden, inter, d, torch.Generator().manual_seed(layer * 4096 + e))
    nh, ni = R.ids_per_row(hidden, d), R.ids_per_row(inter, d)
    rows = torch.arange(inter)
    up_ids = torch.where(((rows + e) % 2 == 0)[:, None], 2, 3).expand(inter, nh)
    down_ids = torch.zeros(hidden, ni, dtype=torch.long)
    down_ids[torch.arange(hidden), (torch.arange(hidden) + e) % (ni - 1)] = 4
    down_ids[:, -1] = 5
    one = lambda n: torch.ones(n, dtype=torch.bfloat16)
    return {
        "w1": {"assignments": R.pack(torch.ones(inter, nh, dtype=torch.long)),
               "input_norm": one(hidden) * 8, "output_norm": one(inter), "bias": None},
        "w3": {"assignments": R.pack(up_ids), "input_norm": one(hidden),
               "output_norm": torch.where(rows % 2 == 0, 1.0, 0.5).bfloat16(), "bias": None},
        "w2": {"assignments": R.pack(down_ids), "input_norm": one(inter),
               "output_norm": one(hidden), "bias": None},
    }


def synth_dir(geom, out, d, kind, layout="row_major"):
    """Full-geometry synthetic sidecar (cached by path)."""
    out = Path(out)
    if (out / "manifest.json").exists():
        return out
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)          # thousands of small tensors: threading only adds overhead
    entries = [write_layer(tmp, layer, {e: synth_expert(kind, d, geom["hidden"], geom["inter"], layer, e)
                                        for e in range(geom["experts"])}, layout)
               for layer in geom["layers"]]
    torch.set_num_threads(threads)
    write_head(tmp, geom["template"], d, synth_codebook(kind, d, geom["inter"]), entries, layout)
    tmp.rename(out)
    return out


def link_copy(src, dst, mutable=("manifest.json",)):
    """dst/ with symlinks to every file of src/ except `mutable` ones (copied)."""
    src, dst = Path(src), Path(dst)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.name in mutable:
            shutil.copy2(item, dst / item.name)
        else:
            os.symlink(item.resolve(), dst / item.name)
    return dst


def edit_manifest(side, fn):
    path = Path(side) / "manifest.json"
    m = json.loads(path.read_text())
    fn(m)
    path.write_text(json.dumps(m, indent=1))

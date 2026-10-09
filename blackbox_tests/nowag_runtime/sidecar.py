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
    """{expert: {w1/w2/w3: {assignments,input_norm,output_norm,bias=None}}} (row_major)."""
    entry = layer_entry(side, layer)
    out = {}
    with safe_open(str(Path(side) / entry["file"]), "pt") as f:
        for e in experts:
            out[e] = {p: {part: f.get_tensor(key(layer, e, p, part))
                          for part in ("assignments", "input_norm", "output_norm")} | {"bias": None}
                      for p in PROJ}
    return out


def tensor_shapes(side, layer):
    entry = layer_entry(side, layer)
    with safe_open(str(Path(side) / entry["file"]), "pt") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def write_layer(out, layer, experts):
    """Write layer-LLL.safetensors/.json for {expert: weights}; return the manifest entry."""
    tensors, matrices = {}, {}
    name = f"layer-{layer:03d}"
    for e, w in experts.items():
        for p in PROJ:
            meta = {}
            for field, part in (("packed_assignments", "assignments"),
                                ("input_norm", "input_norm"), ("output_norm", "output_norm")):
                t = w[p][part].contiguous()
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


def write_head(out, template, d, cb, entries):
    """Codebook file + manifest. template: a real manifest whose model-geometry fields
    (format, model_type, hidden_size, num_experts, ...) are kept; calibration bookkeeping dropped."""
    save_file({"global_all.codebook": cb.contiguous()}, str(Path(out) / "global_codebook.safetensors"))
    keep = {k: v for k, v in template.items()
            if not k.startswith(("lloyd", "calibration", "initialize", "kmeans", "pool"))}
    keep.update({"d": d, "assignment_bits": 12, "assignments_packed": True, "scope": "expert_only",
                 "codebook_sharing": "global_all", "layers": entries,
                 "matrix_count": sum(e["matrix_count"] for e in entries),
                 "n_bits": 12 / d, "normalizer_order": [0, 1], "normalizer_zero": [False, False],
                 "codebook": {"dtype": "bfloat16", "file": "global_codebook.safetensors",
                              "shape": list(cb.shape), "tensor": "global_all.codebook"}})
    (Path(out) / "manifest.json").write_text(json.dumps(keep, indent=1))


def write(out, template, d, cb, layers):
    """Write a native sidecar; layers: {layer: {expert: weights}} with row_major assignments."""
    Path(out).mkdir(parents=True, exist_ok=True)
    entries = [write_layer(out, layer, experts) for layer, experts in sorted(layers.items())]
    write_head(out, template, d, cb, entries)
    return Path(out)


def synth_like(real_side, out, d, seed=0):
    """Random legal sidecar with the real artifact's geometry and layer map, at D=d (cached)."""
    out = Path(out)
    if (out / "manifest.json").exists():
        return out
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    template = manifest(real_side)
    g = torch.Generator().manual_seed(seed)
    cb = R.random_codebook(d, g)
    entries = []
    for entry in template["layers"]:
        layer = entry["layer"]
        shapes = tensor_shapes(real_side, layer)
        experts = {}
        for e in sorted({int(k.split(".")[4]) for k in shapes}):
            inter = shapes[key(layer, e, "w1", "output_norm")][0]
            hidden = shapes[key(layer, e, "w1", "input_norm")][0]
            experts[e] = R.random_expert(hidden, inter, d, g)
        entries.append(write_layer(tmp, layer, experts))
    write_head(tmp, template, d, cb, entries)
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

"""Self-checks of the independent reference (no candidate code involved). CPU only."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import reference as R  # noqa: E402
import sidecar as S  # noqa: E402
from cases import FAMILIES, QWEN36_BF16, QWEN36_SIDE, DSV4_SIDE  # noqa: E402

# K values that occur in the supported models: Qwen3.6 H/I, DSV4 H/I, GPT-OSS H=I, Flash-Next
# H/I, plus the TP2 halves that carry a boundary codeword.
REAL_K = [2048, 512, 4096, 2880, 2560, 640, 256, 1024, 320, 1440]


def test_pack_hand_vector():
    # 0xABC | 0x123<<12 | low 8 bits of 0xFFF <<24 ; then the remaining 4 bits start word 1
    assert torch.equal(R.pack(torch.tensor([[0xABC, 0x123, 0xFFF]])),
                       R.pack_bits(torch.tensor([[0xABC, 0x123, 0xFFF]])))
    words = R.pack(torch.tensor([[0xABC, 0x123, 0xFFF]]))
    assert words.dtype == torch.int32 and words.shape == (1, 2)
    assert (int(words[0, 0]) & 0xFFFFFFFF) == 0xFF123ABC
    assert int(words[0, 1]) == 0xF


def test_pack_rows_restart_at_bit0():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 4096, (3, 86), generator=g)
    whole = R.pack(ids)
    for r in range(3):
        assert torch.equal(whole[r], R.pack(ids[r:r + 1])[0])


@pytest.mark.parametrize("k", REAL_K)
@pytest.mark.parametrize("d", [4, 6])
def test_pack_roundtrip_and_width(k, d):
    g = torch.Generator().manual_seed(k * d)
    count = R.ids_per_row(k, d)
    ids = torch.randint(0, 4096, (5, count), generator=g)
    ids[0] = 4095
    ids[1] = 0
    packed = R.pack(ids)
    assert packed.shape == (5, -(-count * 12 // 32))
    assert torch.equal(packed, R.pack_bits(ids))
    assert torch.equal(R.unpack(packed, count), ids)


@pytest.mark.parametrize("d", [4, 6])
def test_word_major_is_transpose_with_same_values(d):
    g = torch.Generator().manual_seed(d)
    cb = R.random_codebook(d, g)
    proj = R.random_projection(64, 512, d, g)
    wm = R.to_layout(proj["assignments"], "word_major")
    assert wm.shape == proj["assignments"].shape[::-1]
    assert torch.equal(R.codeword_matrix(wm, cb, 512, "word_major"),
                       R.codeword_matrix(proj["assignments"], cb, 512))


def slow_projection(x, proj, cb):
    """Second, loop-based formulation straight from the contract wording."""
    d = cb.shape[1]
    k = proj["input_norm"].shape[0]
    n = proj["output_norm"].shape[0]
    ids = R.unpack(proj["assignments"], R.ids_per_row(k, d))
    xin = x.double() * proj["input_norm"].double()
    out = torch.zeros(x.shape[0], n, dtype=torch.float64)
    for row in range(n):
        for j in range(ids.shape[1]):
            word = cb[int(ids[row, j])].double()
            for lane in range(d):
                col = j * d + lane
                if col < k:                        # tail lanes past K never take part
                    out[:, row] += xin[:, col] * word[lane]
    out = out * proj["output_norm"].double()
    if proj.get("bias") is not None:
        out = out + proj["bias"].double()
    return out


@pytest.mark.parametrize("d,k", [(4, 20), (6, 20), (6, 26), (6, 30), (4, 32)])
@pytest.mark.parametrize("bias", [False, True])
def test_projection_matches_loop_formulation(d, k, bias):
    g = torch.Generator().manual_seed(d * 100 + k)
    cb = R.random_codebook(d, g)
    proj = R.random_projection(7, k, d, g, bias=bias)
    x = torch.randn(3, k, generator=g).bfloat16()
    ref = R.projection(x, proj, cb).double()
    assert torch.allclose(ref, slow_projection(x, proj, cb), rtol=1e-5, atol=1e-6)
    eff = R.effective_weight(proj, cb).double()
    alt = x.double() @ eff.t() + (proj["bias"].double() if bias else 0)
    assert torch.allclose(ref, alt, rtol=1e-5, atol=1e-6)


def test_activation_families_match_public_definitions():
    from transformers.activations import ACT2FN
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    g = torch.Generator().manual_seed(3)
    gate, up = torch.randn(4, 64, generator=g) * 6, torch.randn(4, 64, generator=g) * 6
    assert torch.allclose(R.activation(gate, up, "silu"), ACT2FN["silu"](gate) * up)
    assert torch.allclose(R.activation(gate, up, "gelu_tanh"),
                          ACT2FN["gelu_pytorch_tanh"](gate) * up, atol=1e-6)
    assert torch.allclose(R.activation(gate, up, "gelu"), ACT2FN["gelu"](gate) * up)
    interleaved = torch.stack([gate, up], -1).reshape(4, 128)
    hf = GptOssExperts._apply_gate(type("C", (), {"limit": 7.0, "alpha": 1.702})(), interleaved)
    assert torch.allclose(R.activation(gate, up, "gptoss", alpha=1.702, limit=7.0), hf)
    # DSV4 inference/model.py Expert.forward
    lim = 10.0
    want = torch.nn.functional.silu(gate.clamp(max=lim)) * up.clamp(-lim, lim)
    assert torch.allclose(R.activation(gate, up, "swiglu_limit", limit=lim), want)
    # the families are genuinely different references
    fams = [R.activation(gate, up, f, alpha=1.702, limit=7.0)
            for f in ("silu", "gelu", "gelu_tanh", "gptoss", "swiglu_limit")]
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            assert not torch.allclose(fams[i], fams[j], atol=1e-3)


def test_e4m3_group_round_properties():
    g = torch.Generator().manual_seed(4)
    x = (torch.randn(3, 512, generator=g) * torch.logspace(-3, 2, 512)).bfloat16()
    y = R.e4m3_group_round(x)
    groups = y.reshape(3, 4, 128)
    amax = x.float().reshape(3, 4, 128).abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    assert torch.equal(torch.log2(scale), torch.log2(scale).round())          # power of two
    q = groups / scale
    assert float(q.abs().max()) <= 448
    assert torch.equal(q.to(torch.float8_e4m3fn).float(), q)                   # e4m3 exact
    assert torch.equal(R.e4m3_group_round(y), y)                               # idempotent
    # a single group transcribed by hand: amax 3.0 -> scale 2^ceil(log2(3/448)) = 2^-7
    one = torch.zeros(128)
    one[0], one[1] = 3.0, 0.0101
    r = R.e4m3_group_round(one)
    assert float(r[0]) == 3.0
    assert float(r[1]) == float((torch.tensor(0.0101 * 128).to(torch.float8_e4m3fn).float()) / 128)


def test_dsv4_expert_rounding_positions():
    hidden, inter, _, m = FAMILIES["dsv4"]
    hidden, inter = 256, 256
    g = torch.Generator().manual_seed(5)
    cb = R.random_codebook(6, g)
    w = R.random_expert(hidden, inter, 6, g)
    x = torch.randn(3, hidden, generator=g).bfloat16()
    rw = torch.rand(3, generator=g)
    xq = R.e4m3_group_round(x)
    gate, up = R.projection(xq, w["w1"], cb), R.projection(xq, w["w3"], cb)
    h = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)) * rw[:, None]
    hq = R.e4m3_group_round(h.bfloat16())
    want = R.projection(hq, w["w2"], cb)          # down normalizer applied after rounding
    assert torch.allclose(R.expert(x, w, cb, m, rw), want)


TP_SHAPES = [  # (hidden, inter, d, family) -- boundary codeword exists when (inter/2) % d != 0
    (2048, 512, 6, "qwen36_silu"),
    (4096, 2048, 6, "dsv4"),
    (2560, 640, 6, "qwen36_silu"),   # Flash-Next shape, component-level math
    (2880, 2880, 6, "gptoss"),
    (2048, 512, 4, "qwen36_silu"),
]


@pytest.mark.parametrize("hidden,inter,d,family", TP_SHAPES)
def test_tp2_rank_sum_equals_full(hidden, inter, d, family):
    m = FAMILIES[family][3]
    g = torch.Generator().manual_seed(hidden + inter + d)
    cb = R.random_codebook(d, g)
    w = R.random_expert(hidden, inter, d, g, bias=m.get("bias", False))
    x = torch.randn(4, hidden, generator=g).bfloat16()
    rw = torch.rand(4, generator=g)
    full = R.expert(x, w, cb, m, rw)
    parts = [R.expert_tp(x, R.tp_shard(w, r, 2), cb, m, rw) for r in range(2)]
    assert torch.allclose(parts[0] + parts[1], full, rtol=1e-4, atol=1e-4 * float(full.abs().max()))
    if m.get("bias"):   # down bias exactly once: only rank 0 carries it
        assert R.tp_shard(w, 0, 2)["w2"]["bias"] is not None
        assert R.tp_shard(w, 1, 2)["w2"]["bias"] is None


def test_tp_boundary_codeword_present_in_required_shapes():
    # contract §4: I=512 D6 TP2 (boundary 256) and I=2048 DSV4; 640/2=320 for Flash-Next
    assert 256 % 6 != 0 and 1024 % 6 != 0 and 320 % 6 != 0


# ------------------------------------------------------------------ real public artifacts

def need(path):
    if path is None or not Path(path).exists():
        pytest.skip(f"{path} not available")
    return Path(path)


@pytest.mark.parametrize("which", ["qwen36", "dsv4"])
def test_real_artifact_shapes_follow_contract(which):
    side = need(QWEN36_SIDE if which == "qwen36" else DSV4_SIDE)
    m = S.manifest(side)
    assert m["scope"] == "expert_only" and m["codebook_sharing"] == "global_all"
    assert m["assignment_bits"] == 12 and m["assignments_packed"] is True
    d = m["d"]
    assert tuple(S.codebook(side).shape) == (4096, d)
    first = m["layers"][0]["layer"]
    shapes = S.tensor_shapes(side, first)
    experts = {int(k.split(".")[4]) for k in shapes}
    assert len(shapes) == len(experts) * 9
    for e in experts:
        for p in S.PROJ:
            n = shapes[S.key(first, e, p, "output_norm")][0]
            k = shapes[S.key(first, e, p, "input_norm")][0]
            assert shapes[S.key(first, e, p, "assignments")] == (n, R.words_per_row(k, d))


def dict_weight(cb, ids, proj, k):
    w = cb.float()[ids].reshape(ids.shape[0], -1)[:, :k]
    return w * proj["input_norm"].float()[None, :] * proj["output_norm"].float()[:, None]


def test_real_qwen_decode_tracks_original_bf16():
    """Bit order / norm order sanity: the decoded expert must resemble the BF16 original."""
    side, base = need(QWEN36_SIDE), need(QWEN36_BF16)
    from safetensors import safe_open
    w = S.read_experts(side, 0, [0])[0]
    cb = S.codebook(side)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    pre = "model.language_model.layers.0.mlp.experts."
    with safe_open(str(base / index[pre + "gate_up_proj"]), "pt") as f:
        gate_up = f.get_slice(pre + "gate_up_proj")[0].float()          # [2I, H]
    with safe_open(str(base / index[pre + "down_proj"]), "pt") as f:
        down = f.get_slice(pre + "down_proj")[0].float()                # [H, I]
    inter = w["w1"]["output_norm"].shape[0]
    originals = {"w1": gate_up[:inter], "w3": gate_up[inter:], "w2": down}
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))
    for p, orig in originals.items():
        assert orig.shape == R.effective_weight(w[p], cb).shape
        right = cos(R.effective_weight(w[p], cb), orig)
        d, k = cb.shape[1], w[p]["input_norm"].shape[0]
        ids = R.unpack(w[p]["assignments"], R.ids_per_row(k, d))
        msb_first = (((ids.unsqueeze(-1) >> torch.arange(12)) & 1) << torch.arange(11, -1, -1)).sum(-1)
        wrong = cos(dict_weight(cb, msb_first, w[p], k), orig)
        print(p, "cos right", round(right, 3), "msb-first", round(wrong, 3))
        assert right > 0.8 and right > wrong + 0.2, (p, right, wrong)


@pytest.mark.parametrize("family", ["qwen36_silu", "gptoss", "dsv4"])
def test_frozen_bounds_accept_legal_and_reject_wrong(family):
    import tolerances as TOL
    hidden, inter, top_k, m = FAMILIES[family]
    g = torch.Generator().manual_seed(11)
    cb = R.random_codebook(6, g)
    bank = [R.random_expert(hidden, inter, 6, g, bias=m.get("bias", False)) for _ in range(3)]
    x = torch.randn(8, hidden, generator=g).bfloat16()
    rows = torch.randint(0, 3, (8, top_k), generator=g).int()
    rw = torch.rand(8, top_k, generator=g)
    ref = R.moe(x, rows, rw, bank, cb, m)
    legal = R.moe(x, rows, rw, bank, cb, m, gate_up_bf16=True).bfloat16()
    TOL.assert_close(legal, ref, m)
    if hidden % 128 == 0 and inter % 128 == 0:   # E4M3 block-128 variant is defined
        rounded = dict(m, dsv4_round=not m.get("dsv4_round"))
        wrong = R.moe(x, rows, rw, bank, cb, rounded)
        TOL.assert_discriminates(legal, ref, wrong)
    if m.get("dsv4_round"):
        with pytest.raises(AssertionError):
            TOL.assert_close(wrong.bfloat16(), ref, m)
        moved = R.moe(x, rows, rw, bank, cb, dict(m, route="output"))
        TOL.assert_discriminates(legal, ref, moved)

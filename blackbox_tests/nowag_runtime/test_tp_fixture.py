"""CPU pre-registration of the tiny-model TP2 service rule (reference only, no candidate code).

HF transformers runs the tiny Qwen3MoE BASE with (a) its BF16 experts, (b) the BASE-fitted
NoWAG experts (dequantised with the independent reference) and (c) the rank-1-shuffled
control. TP2's only legal difference from TP1 is an extra bf16 rounding of the cross-rank
sum; it is emulated by splitting every MoE output into two random parts, rounding each to
bf16 and adding them. The measured agreements are the basis of harness.TP2_MARGIN.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import harness as H  # noqa: E402
import reference as R  # noqa: E402
import sidecar as S  # noqa: E402
import tiny_model as tiny  # noqa: E402


def model_with(base, side=None):
    from transformers import Qwen3MoeForCausalLM
    model = Qwen3MoeForCausalLM.from_pretrained(base, dtype=torch.bfloat16).eval()
    if side is None:
        return model
    cb = S.codebook(side)
    for layer in range(tiny.LAYERS):
        experts = model.model.layers[layer].mlp.experts
        for e, w in S.read_experts(side, layer, range(tiny.EXPERTS)).items():
            dense = {p: R.effective_weight(w[p], cb).bfloat16() for p in w}
            with torch.no_grad():
                experts.gate_up_proj[e] = torch.cat([dense["w1"], dense["w3"]])
                experts.down_proj[e] = dense["w2"]
    return model


def split_round(output, gen):
    y = output[0] if isinstance(output, tuple) else output
    a = torch.rand(y.shape, generator=gen) * 0.5 + 0.25
    z = ((y.float() * a).bfloat16().float() + (y.float() * (1 - a)).bfloat16().float()).bfloat16()
    return (z, *output[1:]) if isinstance(output, tuple) else z


def greedy(model, tok, noise=False):
    gen = torch.Generator().manual_seed(7)
    hooks = [layer.mlp.register_forward_hook(lambda m, i, o: split_round(o, gen))
             for layer in model.model.layers] if noise else []
    out = []
    with torch.no_grad():
        for prompt in tiny.PROMPTS:
            ids = tok(prompt, return_tensors="pt").input_ids
            gen_ids = model.generate(ids, max_new_tokens=tiny.GREEDY_TOKENS, do_sample=False)
            out.append(tok.decode(gen_ids[0, ids.shape[1]:], skip_special_tokens=True))
    for h in hooks:
        h.remove()
    return out


@pytest.mark.parametrize("d", [4, 6])
def test_tp2_rule_separates_legal_rounding_from_a_wrong_shard(d):
    from transformers import PreTrainedTokenizerFast
    base, _ = tiny.paths(d)
    tok = PreTrainedTokenizerFast.from_pretrained(base)
    ref = model_with(base)
    gate_up = ref.model.layers[0].mlp.experts.gate_up_proj[0].float()
    assert torch.equal(gate_up[:tiny.INTER], tiny.base_experts(base)[0, 0]["w1"])   # gate first
    bf16 = greedy(ref, tok), greedy(ref, tok, noise=True)
    fit = model_with(base, tiny.fitted_side(d))
    fitted = greedy(fit, tok), greedy(fit, tok, noise=True)
    wrong = greedy(model_with(base, tiny.fitted_side(d, "shuffled")), tok)
    a_bf16, a_fit = H.agreement(*bf16), H.agreement(*fitted)
    a_wrong = H.agreement(fitted[0], wrong)
    print(f"D{d}: bf16 split-round {a_bf16:.3f}  fitted split-round {a_fit:.3f}  "
          f"shuffled rank-1 half {a_wrong:.3f}")
    threshold = a_bf16 - H.TP2_MARGIN
    assert a_fit >= threshold and a_wrong < threshold, (a_bf16, a_fit, a_wrong)

"""Draft routing restricted to experts already present in the shared GPU cache."""
import torch

from .fused import fused_topk


def build_expert_affinity(cache) -> torch.Tensor:
    if cache.quant_format != "bf16":
        raise ValueError("affinity draft residency requires unquantized floating-point expert weights")
    distances = torch.empty(cache.num_layers, cache.num_experts, cache.num_experts,
                            dtype=torch.float32, device="cpu")
    for layer in range(cache.num_layers):
        gram = torch.zeros(cache.num_experts, cache.num_experts, dtype=torch.float32, device="cpu")
        for sources in cache.bank_sources.values():
            weights = sources[layer].view(cache.num_experts, -1)
            # Full weights, in bounded chunks; no sampled dimensions or normalization.
            for start in range(0, weights.shape[1], 65536):
                block = weights[:, start : start + 65536].float()
                gram.addmm_(block, block.T)
        norms = gram.diagonal()
        distances[layer] = (norms[:, None] + norms[None, :] - 2 * gram).clamp_min_(0)
    return distances.to(cache.device)


def cached_draft_routing(hidden, logits, top_k, renormalize, available, affinity):
    if affinity is None:
        weights, ids = fused_topk(hidden, logits.masked_fill(~available, -float("inf")),
                                  top_k, renormalize)
        if not renormalize:
            weights = logits.float().softmax(dim=-1).gather(1, ids.long())
        return weights, ids, None

    weights, ids = fused_topk(hidden, logits, top_k, renormalize)
    original = ids.long()
    present = available[original]
    missing = ~present
    occupied = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, original, present)
    selected = ids.clone()
    # The fused router orders output by expert ID, not by routing score.
    order = logits.gather(1, original).argsort(dim=-1, descending=True, stable=True)
    for rank in range(top_k):
        column = order[:, rank : rank + 1]
        source = original.gather(1, column).flatten()
        candidates = affinity[source].masked_fill(~available | occupied, float("inf"))
        nearest = candidates.argmin(dim=-1, keepdim=True)
        replacement = torch.where(present.gather(1, column), source[:, None], nearest)
        selected.scatter_(1, column, replacement.to(torch.int32))
        occupied.scatter_(1, replacement, True)
    return weights, selected, missing

"""Draft routing restricted to experts already present in the shared GPU cache."""
from .fused import fused_topk


def draft_routing(ctx, layer_id, experts, hidden, logits):
    """(weights, ids) for a speculative draft forward, or None when this forward is not a draft."""
    draft_experts = ctx.batch.draft_experts
    if draft_experts is None:
        return None
    if ctx.speculative_cost is not None:
        _, predicted = fused_topk(hidden, logits, experts.top_k, experts.renormalize)
        ctx.speculative_cost.record_prediction(layer_id, predicted, logits)
    available = (ctx.batch.draft_available_experts[layer_id]
                 if ctx.batch.draft_available_experts is not None else None)
    if ctx.draft_load_missing:
        available = ctx.moe_offload_cache.slot_for_id[layer_id] >= 0
    if available is not None:
        return cached_draft_routing(hidden, logits, draft_experts, experts.renormalize,
                                    available, load_missing=ctx.draft_load_missing)
    return fused_topk(hidden, logits, draft_experts, experts.renormalize)


def cached_draft_routing(hidden, logits, top_k, renormalize, available, load_missing=False):
    if load_missing:
        scores = logits.float()
        span = scores.amax(dim=-1, keepdim=True) - scores.amin(dim=-1, keepdim=True) + 1
        _, ids = fused_topk(hidden, scores + available * span, top_k, renormalize)
        weights = (scores.gather(1, ids.long()).softmax(dim=-1) if renormalize
                   else scores.softmax(dim=-1).gather(1, ids.long()))
        return weights, ids
    weights, ids = fused_topk(hidden, logits.masked_fill(~available, -float("inf")), top_k, renormalize)
    if not renormalize:
        weights = logits.float().softmax(dim=-1).gather(1, ids.long())
    return weights, ids

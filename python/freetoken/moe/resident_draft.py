"""Draft routing restricted to experts already present in the shared GPU cache."""
from .fused import fused_topk


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

"""Router components shared by target, draft and verification prediction."""


class SoftmaxRouter:
    def __init__(self, top_k, renormalize):
        self.top_k, self.renormalize = top_k, renormalize

    def route(self, hidden, logits, top_k=None, available=None, load_missing=False,
              num_token_non_padded=None):
        from .fused import fused_topk

        top_k = self.top_k if top_k is None else top_k
        selection = logits
        if available is not None:
            if load_missing:
                scores = logits.float()
                span = scores.amax(dim=-1, keepdim=True) - scores.amin(dim=-1, keepdim=True) + 1
                selection = scores + available * span
            else:
                selection = logits.masked_fill(~available, -float("inf"))
        weights, ids = fused_topk(hidden, selection, top_k, self.renormalize,
                                 num_token_non_padded=num_token_non_padded)
        if available is not None and (load_missing or not self.renormalize):
            scores = logits.float()
            weights = (scores.gather(1, ids.long()).softmax(dim=-1) if self.renormalize
                       else scores.softmax(dim=-1).gather(1, ids.long()))
        return weights, ids

    def predict(self, hidden, logits, scores):
        _, ids = self.route(hidden, logits)
        return ids, logits.float().softmax(dim=-1) if scores else None


ROUTERS = {"softmax": SoftmaxRouter}


def route_experts(layer, hidden, logits):
    from freetoken.core import get_global_ctx

    ctx = get_global_ctx()
    batch, router = ctx.batch, layer.router
    if batch.draft_experts is None:
        padding = (batch.num_token_non_padded
                   if not batch.uses_extend_path or batch.is_speculative_verify else None)
        return router.route(hidden, logits, num_token_non_padded=padding)
    if ctx.speculative_cost is not None:
        ids, scores = router.predict(hidden, logits, ctx.speculative_cost.prefetch is not None)
        ctx.speculative_cost.record_prediction(layer.layer_id, ids, scores)
    available = (batch.draft_available_experts[layer.layer_id]
                 if batch.draft_available_experts is not None else None)
    if ctx.draft_load_missing:
        available = ctx.moe_offload_cache.slot_for_id[layer.layer_id] >= 0
    return router.route(hidden, logits, batch.draft_experts, available, ctx.draft_load_missing)

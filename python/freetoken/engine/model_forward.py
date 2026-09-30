"""Complete one model forward, including asynchronous expert work."""
from freetoken.core import get_global_ctx


def forward_model(model):
    logits = model.forward()
    ctx = get_global_ctx()
    cost = ctx.speculative_cost
    if cost is not None and cost.prefetch is not None:
        # Join before returning, including the warmup immediately preceding capture.
        cost.prefetch.finish_forward(cost.phase(ctx.batch))
    return logits

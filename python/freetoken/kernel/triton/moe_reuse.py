"""One-program-per-request S2-MoE verification routing."""
import torch
import triton
import triton.language as tl


@triton.jit
def _reuse_routing_kernel(
    logits_ptr, weights_ptr, ids_ptr, offsets_ptr, out_weights_ptr, out_ids_ptr, changed_ptr,
    E: tl.constexpr, K: tl.constexpr, CAP: tl.constexpr, RENORMALIZE: tl.constexpr,
    BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr,
):
    request = tl.program_id(0)
    start = tl.load(offsets_ptr + request)
    end = tl.load(offsets_ptr + request + 1)
    experts = tl.arange(0, BLOCK_E)
    ranks = tl.arange(0, BLOCK_K)
    expert_mask = experts < E
    rank_mask = ranks < K
    preferred = tl.full((BLOCK_E,), False, tl.int1)
    maximum_weight = 0.0
    strength = 0.0

    if end - start > 1:
        importance = tl.zeros((BLOCK_E,), tl.float32)
        margin = 0.0
        for token in tl.range(start, end):
            raw = tl.load(logits_ptr + token * E + experts, expert_mask, other=0).to(tl.float32)
            maximum = tl.max(tl.where(expert_mask, raw, -float("inf")), axis=0)
            weight = tl.maximum(maximum - tl.sum(raw, axis=0) / E, 0.0)
            maximum_weight = tl.maximum(maximum_weight, weight)
            importance += weight * raw
            ordered = tl.sort(tl.where(expert_mask, raw, -float("inf")), descending=True)
            next_score = tl.sum(tl.where(experts == K, ordered, 0.0), axis=0)
            margin += maximum - next_score
        strength = margin / (end - start)
        if maximum_weight > 0 and strength > 0:
            importance /= maximum_weight
            importance = tl.where(expert_mask, importance, -float("inf"))
            for _ in range(CAP):
                winner = tl.argmax(importance, axis=0, tie_break_left=True)
                preferred |= experts == winner
                importance = tl.where(experts == winner, -float("inf"), importance)

    changed_total = 0
    for token in tl.range(start, end):
        old_ids = tl.load(ids_ptr + token * K + ranks, rank_mask, other=0)
        old_weights = tl.load(weights_ptr + token * K + ranks, rank_mask, other=0)
        output_ids = old_ids
        output_weights = old_weights
        if maximum_weight > 0 and strength > 0:
            raw = tl.load(logits_ptr + token * E + experts, expert_mask, other=0).to(tl.float32)
            selection = tl.where(expert_mask, raw + tl.where(preferred, strength, 0.0), -float("inf"))
            chosen = tl.full((BLOCK_E,), False, tl.int1)
            new_ids = tl.zeros((BLOCK_K,), tl.int32)
            selected_logits = tl.zeros((BLOCK_K,), tl.float32)
            for rank in range(K):
                winner = tl.argmax(selection, axis=0, tie_break_left=True)
                original_score = tl.load(logits_ptr + token * E + winner).to(tl.float32)
                new_ids = tl.where(ranks == rank, winner, new_ids)
                selected_logits = tl.where(ranks == rank, original_score, selected_logits)
                chosen |= experts == winner
                selection = tl.where(experts == winner, -float("inf"), selection)
            retained = tl.gather(chosen.to(tl.int32), old_ids, axis=0)
            changed = tl.sum(tl.where(rank_mask, retained, 0), axis=0) != K
            maximum = tl.max(tl.where(expert_mask, raw, -float("inf")), axis=0)
            numerator = tl.where(rank_mask, tl.exp(selected_logits - maximum), 0.0)
            if RENORMALIZE:
                denominator = tl.sum(numerator, axis=0)
            else:
                denominator = tl.sum(tl.where(expert_mask, tl.exp(raw - maximum), 0.0), axis=0)
            output_ids = tl.where(changed, new_ids, old_ids)
            output_weights = tl.where(changed, numerator / denominator, old_weights)
            changed_total += changed.to(tl.int32)
        tl.store(out_ids_ptr + token * K + ranks, output_ids, rank_mask)
        tl.store(out_weights_ptr + token * K + ranks, output_weights, rank_mask)
    tl.atomic_add(changed_ptr, changed_total.to(tl.int64))


def reuse_routing(logits, weights, ids, offsets, cap: int, renormalize: bool, changed):
    experts, top_k = logits.shape[1], ids.shape[1]
    if cap == experts or top_k == experts:
        return weights, ids
    out_weights, out_ids = torch.empty_like(weights), torch.empty_like(ids)
    _reuse_routing_kernel[(offsets.numel() - 1,)](
        logits, weights, ids, offsets, out_weights, out_ids, changed,
        experts, top_k, cap, renormalize,
        triton.next_power_of_2(experts), triton.next_power_of_2(top_k), num_warps=4,
    )
    return out_weights, out_ids

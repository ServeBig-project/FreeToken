"""Bounded same-pool verification prefetch, published only after its copy completes."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .offload_kernels import _select_layer_victim


@triton.jit(do_not_specialize=["layer", "used_count", "batch_size"])
def _plan(slots, owners, usage, predicted, scores, target_use, used_slots,
          allowed, marks, evicted, stats, unit_ms, window_ms, probe_left,
          dst, src, count, layer, used_count, batch_size,
          LAYERS: tl.constexpr, EXPERTS: tl.constexpr, CAPACITY: tl.constexpr,
          BLOCK_C: tl.constexpr, BLOCK_E: tl.constexpr, PROTECT_ALLOWED: tl.constexpr):
    pinned: tl.constexpr = 9223372036854775807
    e = tl.arange(0, BLOCK_E)
    c = tl.arange(0, BLOCK_C)
    base = layer * EXPERTS
    candidate = (e < EXPERTS) & tl.load(predicted + base + e, mask=e < EXPERTS, other=0)
    candidate = candidate & (tl.load(slots + base + e, mask=e < EXPERTS, other=0) < 0)
    priority = tl.load(scores + base + e, mask=e < EXPERTS, other=-float("inf"))
    owner = tl.load(owners + c, mask=c < CAPACITY, other=-1)
    age = tl.load(usage + c, mask=c < CAPACITY, other=pinned)
    eligible = (c < CAPACITY) & (age != pinned)
    for i in tl.range(used_count):
        eligible = eligible & (c != tl.load(used_slots + i))
    if PROTECT_ALLOWED:
        eligible = eligible & ~tl.load(allowed + owner, mask=owner >= 0, other=0)
    unit = tl.load(unit_ms)
    window = tl.load(window_ms)
    probe = tl.load(probe_left)
    hit_rate = (tl.load(stats + 2).to(tl.float32) + 1) / (
        tl.load(stats + 2).to(tl.float32) + tl.load(stats + 3).to(tl.float32) + 2)
    copies = 0
    copied_ms = 0.0
    running = True
    while running & (tl.sum(candidate.to(tl.int32)) > 0) & (tl.sum(eligible.to(tl.int32)) > 0):
        expert = tl.argmax(tl.where(candidate, priority, -float("inf")), axis=0)
        victim, old_id = _select_layer_victim(owner, age, eligible, c, layer,
                                             LAYERS, EXPERTS, True)
        victim_p = tl.load(target_use + old_id, mask=old_id >= 0, other=0.0)
        reload_cost = (1 - tl.exp(tl.log(1 - victim_p) * batch_size)) * unit
        exposed = tl.maximum(0.0, copied_ms + unit - window) - tl.maximum(0.0, copied_ms - window)
        profitable = (exposed + reload_cost <= hit_rate * unit) & (unit > 0)
        allowed_copy = (probe >= 0) & (profitable | ((unit <= 0) & (probe > 0)))
        if allowed_copy:
            if old_id >= 0:
                tl.store(slots + old_id, -1)
                tl.store(evicted + old_id, True)
                unused = tl.load(marks + old_id)
                tl.atomic_add(stats + 3, unused.to(tl.int64))
                tl.store(marks + old_id, False)
            tl.store(owners + victim, base + expert)
            tl.store(usage + victim, pinned)
            # A negative mapping cannot be mistaken for a completed cache hit.
            tl.store(slots + base + expert, -2)
            tl.store(dst + copies, victim)
            tl.store(src + copies, expert)
            copies += 1
            copied_ms += unit
            if unit <= 0:
                probe -= 1
            candidate = candidate & (e != expert)
            eligible = eligible & (c != victim)
        running = allowed_copy
    tl.store(count, copies.to(tl.int64))
    tl.store(probe_left, probe)


@triton.jit(do_not_specialize=["layer"])
def _publish(slots, usage, step, marks, stats, dst, src, count,
             layer, EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    n = tl.load(count)
    slot = tl.load(dst + i, mask=i < n, other=0)
    expert = tl.load(src + i, mask=i < n, other=0)
    tl.store(slots + layer * EXPERTS + expert, slot, mask=i < n)
    tl.store(usage + slot, tl.load(step), mask=i < n)
    tl.store(marks + layer * EXPERTS + expert, True, mask=i < n)
    tl.atomic_add(stats + 1, n)


class VerifyPrefetch:
    def __init__(self, cost):
        self.cost, self.cache = cost, cost.cache
        device = cost.engine.device
        self.stream = torch.cuda.Stream(device=device)
        self.ready, self.done = torch.cuda.Event(), torch.cuda.Event()
        self.copy_events = [cost._events(True) for _ in range(cost.layers)]
        self.wait_events = [cost._events(True) for _ in range(cost.layers + 1)]
        self.window_ms = torch.zeros_like(cost.gemm_ms)
        self.scores = torch.zeros(cost.layers, cost.experts, device=device)
        self.marks = torch.zeros_like(cost.predicted)
        self.evicted = torch.zeros_like(cost.predicted)
        self.stats = torch.zeros(4, dtype=torch.int64, device=device)
        self.rows = torch.zeros(cost.layers, dtype=torch.int32, device=device)
        self.dst = torch.empty(cost.experts, dtype=torch.int32, device=device)
        self.src = torch.empty_like(self.dst)
        self.count = torch.zeros(1, dtype=torch.int64, device=device)
        self.probe_left = torch.full((), -1, dtype=torch.int32, device=device)
        self.pending_layer = None

    def predict(self, layer, ids, scores):
        predicted = torch.zeros_like(self.marks[layer]).scatter_(0, ids.flatten().long(), True)
        self.cost.predicted[layer] |= predicted
        self.stats[0] += self.cost.predicted[layer].sum()
        self.scores[layer] += scores.sum(dim=0)

    def start_step(self):
        self.probe_left.fill_(1 if not self.cost.expert_sample_rows else 0)

    def reconcile(self):
        present = self.cache.slot_for_id >= 0
        self.stats[3] += (self.marks & ~present).sum()
        self.marks &= present

    def finish(self, index):
        self.wait_events[index][0].record()
        if self.pending_layer is not None:
            torch.cuda.current_stream().wait_event(self.done)
            cache = self.cache
            _publish[(1,)](cache.slot_for_id, cache.usage, cache.step, self.marks, self.stats,
                            self.dst, self.src, self.count, self.pending_layer,
                            cache.num_experts, triton.next_power_of_2(cache.num_experts))
            self.pending_layer = None
        self.wait_events[index][1].record()

    def before_layer(self, phase, layer):
        if phase == 1:
            self.finish(layer)
        self.reconcile()
        if phase != 1:
            used = self.marks[layer] & (self.cost.routes[phase, layer] > 0)
            self.stats[2] += used.sum()
            self.marks[layer] &= ~used

    def launch(self, layer, slots, batch):
        cache = self.cache
        allowed = batch.draft_available_experts
        _plan[(1,)](
            cache.slot_for_id, cache.id_of_slot, cache.usage,
            self.cost.predicted, self.scores, self.cost.target_use, slots,
            allowed if allowed is not None else self.marks,
            self.marks, self.evicted, self.stats, self.cost.expert_ms,
            self.window_ms[batch.size, layer], self.probe_left,
            self.dst, self.src, self.count, layer, slots.numel(), batch.size,
            LAYERS=cache.num_layers, EXPERTS=cache.num_experts, CAPACITY=cache.decode_cache_size,
            BLOCK_C=triton.next_power_of_2(cache.decode_cache_size),
            BLOCK_E=triton.next_power_of_2(cache.num_experts), PROTECT_ALLOWED=allowed is not None,
        )
        self.rows[layer].copy_(self.count[0])
        self.ready.record()
        self.stream.wait_event(self.ready)
        with torch.cuda.stream(self.stream):
            self.copy_events[layer][0].record()
            cache._copy_missing_plan(layer, self.dst, self.src, self.count)
            self.copy_events[layer][1].record()
            self.done.record()
        self.pending_layer = layer

    def finish_forward(self, phase):
        if phase == 1:
            self.finish(self.cost.layers)
        self.reconcile()

    def collect_times(self, batch_size, rows):
        copied = sum(start.elapsed_time(end) for count, (start, end) in
                     zip(rows, self.copy_events, strict=True) if count)
        wait = sum(start.elapsed_time(end) for start, end in self.wait_events)
        windows = [self.cost.gemm_events[1][layer][0].elapsed_time(self.wait_events[layer + 1][0])
                   for layer in range(self.cost.layers)]
        self.window_ms[batch_size] = torch.tensor(windows, device=self.window_ms.device)
        return copied, sum(rows), wait

    def before_capture(self):
        self.reconcile()
        self.stats[3] += self.marks.sum()
        self.marks.zero_()
        self.probe_left.fill_(-1)
        return self.stats.clone()

    def after_capture(self, stats):
        self.stats.copy_(stats)
        self.marks.zero_()
        self.rows.zero_()
        self.scores.zero_()
        self.evicted.zero_()

    def snapshot(self):
        values = self.stats.cpu().tolist()
        names = ("predicted", "loaded", "used", "evicted_unused")
        result = {f"prefetch_{name}_experts": value for name, value in zip(names, values, strict=True)}
        for name, value in zip(names[1:], values[1:], strict=True):
            result[f"prefetch_{name}_bytes"] = value * self.cache._expert_row_bytes
        return result

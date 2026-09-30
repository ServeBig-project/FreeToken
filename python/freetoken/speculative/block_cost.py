"""Choose a whole parallel draft block using measured draft and target costs."""
from __future__ import annotations

import time

import torch


class BlockDraftCost:
    def __init__(self, engine):
        self.engine = engine
        self.cost = engine.speculative_cost
        self.limit = engine.config.speculative_num_steps
        self.adaptive = engine.config.speculative_adaptive_cost
        self.events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.pending = None
        self.means = {}
        self.samples = {}
        self.accepted = {}
        self.trials = {}
        self.steps = (self.cost.steps if self.cost is not None else
                      torch.arange(1, self.limit + 1, device=engine.device))
        self.prior = self.cost.prior if self.cost is not None else 0.5 ** self.steps
        self.rounds = {}
        self.choices = [0] * (self.limit + 1)
        self.gpu_ms = 0.0

    def _collect(self):
        if self.pending is None or not self.events[1].query():
            return
        batch_size, width = self.pending
        ms = self.events[0].elapsed_time(self.events[1])
        key = (batch_size, width)
        self.means[key] = 0.8 * self.means.get(key, ms) + 0.2 * ms
        self.samples[key] = self.samples.get(key, 0) + 1
        self.gpu_ms += ms
        if self.cost is not None:
            self.cost.gpu_ms["draft"] += ms
            self.cost.samples[1][batch_size] += 1
        self.pending = None

    def begin(self):
        self._collect()
        self.pending = None
        self.events[0].record()

    def end(self, batch_size, width):
        self.events[1].record()
        self.pending = (batch_size, width)

    def _draft_ms(self, batch_size, width):
        # Scale only across batch sizes at the same block width. A new width
        # must be measured; a parallel block need not scale linearly in width.
        reference = min((b for b, n in self.means if n == width), key=lambda b: abs(b - batch_size))
        return self.means[reference, width] * batch_size / reference

    def observe_acceptance(self, lengths, accepted):
        key = (len(lengths), max(lengths))
        if key not in self.trials:
            self.accepted[key] = torch.zeros_like(self.prior)
            self.trials[key] = torch.zeros_like(self.prior)
        drafted = torch.tensor(lengths, device=accepted.device)
        self.trials[key] += (drafted[:, None] >= self.steps).sum(dim=0)
        self.accepted[key] += (accepted[:, None] >= self.steps).sum(dim=0)
        if self.cost is not None:
            self.cost.observe_acceptance(lengths, accepted)

    def _survival(self, batch_size, width):
        # Changing the number of mask tokens changes every draft distribution.
        key = (batch_size, width)
        if key not in self.trials:
            return self.prior
        empirical = (self.accepted[key] + 2 * self.prior) / (self.trials[key] + 2)
        return empirical.cummin(dim=0).values

    def plan(self, lengths):
        started = time.perf_counter()
        self._collect()
        cost = self.cost
        if cost is not None:
            cost.collect_ready()
            for part in range(2):
                cost._collect_state(part)
            cost.predicted.zero_()
            cost.predicted_positions.zero_()
            cost.rounds += 1
        if not self.adaptive or not any(lengths):
            return self._choose(lengths, max(lengths, default=0), False, started)

        batch_size = len(lengths)
        round_no = self.rounds.get(batch_size, 0) + 1
        self.rounds[batch_size] = round_no
        candidates = sorted({min(n, self.limit, max(lengths)) for n in (2, 4, 8)})
        if not cost.samples[0][batch_size]:
            return self._choose(lengths, 0, False, started)

        unseen = [n for n in candidates if not any(width == n for _, width in self.means)]
        missing = [n for n in candidates if (batch_size, n) not in self.means]
        if unseen:
            return self._choose(lengths, unseen[0], True, started)
        if len(missing) == len(candidates):
            return self._choose(lengths, candidates[0], True, started)
        if round_no % 16 == 0:
            width = missing[0] if missing else candidates[(round_no // 16 - 1) % len(candidates)]
            return self._choose(lengths, width, True, started)

        survival = torch.stack([self._survival(batch_size, width) for width in candidates])
        active = torch.tensor([
            [sum(min(length, width) > step for length in lengths) / batch_size
             for step in range(self.limit)]
            for width in candidates
        ], device=self.engine.device)
        expected = 1 + (active * survival).sum(dim=1)
        state = cost.state_ms[:, batch_size].sum()
        scores = [cost._estimate(0, batch_size, batch_size)]
        for i, width in enumerate(candidates):
            logical = batch_size + sum(min(length, width) for length in lengths)
            physical = cost._physical_verify(batch_size, logical)
            verify = cost._estimate(2, batch_size, physical, logical)
            scores.append((self._draft_ms(batch_size, width) + verify + state) / expected[i])
        selected = int(torch.stack(scores).argmin().item())
        width = candidates[selected - 1] if selected else 0
        return self._choose(lengths, width, False, started)

    def _choose(self, lengths, width, probe, started):
        self.choices[width] += 1
        if self.cost is not None:
            eligible = sum(length > 0 for length in lengths)
            if not width:
                self.cost.ar_requests += eligible
            elif probe:
                self.cost.probe_requests += eligible
            self.cost.control_ms += (time.perf_counter() - started) * 1000
        return [min(length, width) for length in lengths]

    def snapshot(self):
        self._collect()
        return {
            "dflash_block_gpu_ms": self.gpu_ms,
            "dflash_block_samples": {f"{b}:{n}": count for (b, n), count in self.samples.items()},
            "dflash_block_choices": list(self.choices),
        }

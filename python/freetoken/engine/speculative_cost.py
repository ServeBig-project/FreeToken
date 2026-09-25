"""Online batch costs and empirical accepted-prefix lengths for stepwise SD."""
from __future__ import annotations

import time

import torch


class SpeculativeCost:
    phases = ("ar", "draft", "verify")

    def __init__(self, engine):
        self.engine, self.cache = engine, engine.moe_offload_cache
        self.adaptive = engine.config.speculative_adaptive_cost
        self.limit = engine.config.speculative_num_steps
        model = engine.config.model_config
        self.layers, self.experts = model.num_moe_layers, model.num_experts
        self.target_k = model.num_experts_per_tok
        device, batches = engine.device, engine.config.max_running_req + 1
        self.routes = torch.zeros(3, self.layers, self.experts, dtype=torch.int32, device=device)
        self.predicted = torch.zeros(self.layers, self.experts, dtype=torch.bool, device=device)
        self.predicted_positions = torch.zeros((), dtype=torch.int32, device=device)
        self.use = torch.empty(2, self.layers, self.experts, device=device)
        self.use[0].fill_(engine.config.speculative_draft_experts / self.experts)
        self.use[1].fill_(self.target_k / self.experts)
        self.target_use = self.use[1]
        self.copy_rows = torch.zeros(3, self.layers, dtype=torch.int32, device=device)
        self.predicted_misses = torch.zeros(3, device=device)
        self.means = torch.zeros(3, batches, 3, device=device)  # other/physical, MoE/logical, wait
        self.gemm_ms = torch.zeros(batches, self.layers, device=device)
        self.expert_ms = torch.zeros((), device=device)
        self.accepted = torch.zeros(batches, self.limit, device=device)
        self.trials = torch.zeros_like(self.accepted)
        self.steps = torch.arange(1, self.limit + 1, device=device)
        self.prior = 0.5 ** self.steps
        self.samples = [[0] * batches for _ in self.phases]
        self.depth_seen = [0] * batches
        self.mean_cpu = {}
        self.expert_sample_rows = 0
        self.expert_ms_cpu = 0.0
        self.pending = {}
        self.forward_events = [self._events() for _ in self.phases]
        # External record nodes publish fresh timestamps on every graph replay.
        self.copy_events = [[self._events(True) for _ in range(self.layers)] for _ in self.phases]
        self.gemm_events = [[self._events(True) for _ in range(self.layers)] for _ in self.phases]
        self.gpu_ms = dict.fromkeys((*self.phases, "moe_compute", "demand_copy", "prefetch_copy", "prefetch_wait"), 0.0)
        self.transfer_predictions = {p: dict(predicted_experts=0.0, actual_experts=0,
                                              abs_error_experts=0.0) for p in self.phases}
        self.control_ms = 0.0
        self.ar_requests = self.stopped_requests = self.probe_requests = 0
        self.rounds = 0
        self.probe_depth = 0
        self.prefetch = None

    @staticmethod
    def _events(external=False):
        return (torch.cuda.Event(enable_timing=True, external=external),
                torch.cuda.Event(enable_timing=True, external=external))

    @staticmethod
    def phase(batch):
        if batch.draft_experts is not None:
            return 1
        if batch.is_speculative_verify:
            return 2
        return 0 if batch.is_decode_only else None

    def record_routes(self, phase, layer, ids):
        flat = ids.flatten()
        row = self.routes[phase, layer]
        row.zero_().scatter_add_(0, flat.clamp_min(0).long(), (flat >= 0).to(torch.int32))
        valid_rows = (ids[:, 0] >= 0).sum().clamp_min(1)
        self.use[0 if phase == 1 else 1, layer].lerp_(row.float() / valid_rows, 0.1)

    def record_prediction(self, layer, ids, scores):
        if self.prefetch is None:
            self.predicted[layer].scatter_(0, ids.flatten().long(), True)
        else:
            self.prefetch.predict(layer, ids, scores)
        if layer == 0:
            self.predicted_positions.add_(ids.shape[0])

    def begin_model(self, batch):
        phase = self.phase(batch)
        if phase is None:
            return
        self.forward_events[phase][0].record()
        if phase == 1 and self.prefetch is not None:
            self.prefetch.start_step()
        self.predicted_misses[phase].copy_(self._expected_misses(phase, batch.size, batch.positions.numel()))

    def end_model(self, batch):
        phase = self.phase(batch)
        if phase is None:
            return
        self.forward_events[phase][1].record()
        physical = batch.positions.numel()
        graphs = self.engine.graph_runner
        if graphs.can_use_cuda_graph(batch):
            physical = (graphs.speculative._key(batch)[2] if phase else batch.padded_size)
        self.pending[phase] = (batch.size, physical, batch.positions.numel())

    def _collect(self, phase, rows, predicted, prefetch_rows=None):
        pending = self.pending.pop(phase, None)
        if pending is None:
            return
        batch_size, physical, logical = pending
        total = self.forward_events[phase][0].elapsed_time(self.forward_events[phase][1])
        copy = sum(start.elapsed_time(end) for count, (start, end) in
                   zip(rows, self.copy_events[phase], strict=True) if count)
        gemm = [start.elapsed_time(end) for start, end in self.gemm_events[phase]]
        prefetch_copy = copied_prefetch_rows = wait = 0.0
        if phase == 1 and self.prefetch is not None:
            prefetch_copy, copied_prefetch_rows, wait = self.prefetch.collect_times(batch_size, prefetch_rows)
        misses = sum(rows)
        value = [max(0.0, total - copy - wait - sum(gemm)) / physical, sum(gemm) / logical, wait]
        key = (phase, batch_size)
        old = self.mean_cpu.get(key, value)
        mean = [0.8 * a + 0.2 * b for a, b in zip(old, value, strict=True)]
        self.mean_cpu[key] = mean
        self.means[phase, batch_size] = torch.tensor(mean, device=self.engine.device)
        self.samples[phase][batch_size] += 1
        copied = misses + copied_prefetch_rows
        if copied:
            unit = (copy + prefetch_copy) / copied
            self.expert_ms_cpu = (0.8 * self.expert_ms_cpu + 0.2 * unit
                                  if self.expert_sample_rows else unit)
            self.expert_sample_rows += int(copied)
            self.expert_ms.fill_(self.expert_ms_cpu)
        if phase == 1:
            self.gemm_ms[batch_size] = torch.tensor(gemm, device=self.engine.device)
        self.gpu_ms[self.phases[phase]] += total
        self.gpu_ms["moe_compute"] += sum(gemm)
        self.gpu_ms["demand_copy"] += copy
        self.gpu_ms["prefetch_copy"] += prefetch_copy
        self.gpu_ms["prefetch_wait"] += wait
        forecast = self.transfer_predictions[self.phases[phase]]
        forecast["predicted_experts"] += predicted
        forecast["actual_experts"] += int(misses)
        forecast["abs_error_experts"] += abs(predicted - misses)

    def collect_ready(self):
        ready = [p for p in self.pending if self.forward_events[p][1].query()]
        if ready:
            tensors = [self.copy_rows.flatten().float(), self.predicted_misses]
            if self.prefetch is not None:
                tensors.append(self.prefetch.rows.float())
            data = torch.cat(tensors).cpu().tolist()
            for phase in ready:
                start = phase * self.layers
                self._collect(phase, data[start:start + self.layers],
                              data[3 * self.layers + phase], data[3 * self.layers + 3:])

    def _mean(self, phase, batch_size):
        known = [b for b, count in enumerate(self.samples[phase]) if count]
        if known:
            reference = min(known, key=lambda b: abs(b - batch_size))
            return self.means[phase, reference]
        if phase == 0:
            return self._mean(2, batch_size)
        return self.means[phase, batch_size]

    def _survival(self, batch_size):
        empirical = (self.accepted[batch_size] + 2 * self.prior) / (self.trials[batch_size] + 2)
        return empirical.cummin(dim=0).values

    def _physical_verify(self, batch_size, logical):
        graphs = self.engine.graph_runner.speculative
        if graphs is not None and logical > graphs.exact_tokens:
            return min(batch_size * graphs.query_width, graphs.max_tokens)
        return logical

    def _expected_misses(self, phase, batch_size, logical):
        cold = self.cache.slot_for_id < 0
        if phase == 1 and self.engine.config.speculative_draft_residency == "router":
            if not self.engine.config.speculative_draft_load_missing:
                return self.expert_ms.new_zeros(())
            needed = (self.engine.config.speculative_draft_experts - (~cold).sum(dim=1)).clamp_min(0)
            mass = self.target_use * cold
            probability = (mass * needed[:, None] / mass.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
        else:
            probability = self.use[0 if phase == 1 else 1]
        positions = logical
        if phase == 2:
            positions = (logical - self.predicted_positions).clamp_min(0)
        union = 1 - (1 - probability).pow(positions)
        if phase == 2:
            union = torch.where(self.predicted, 1.0, union)
            if self.prefetch is not None:
                union = 1 - (1 - union) * (1 - self.prefetch.evicted * self.target_use)
        return (union * cold).sum()

    def _estimate(self, phase, batch_size, physical, logical=None):
        logical = physical if logical is None else logical
        other, experts, wait = self._mean(phase, batch_size).unbind()
        cold = self._expected_misses(phase, batch_size, logical)
        # The compute term has its measured transfers removed; charge misses once.
        return other * physical + experts * logical + wait + cold * self.expert_ms

    def admit(self, lengths, residency_ok):
        started = time.perf_counter()
        batch_size = len(lengths)
        self.rounds += 1
        self.probe_depth = 0
        self.predicted.zero_()
        self.predicted_positions.zero_()
        if self.prefetch is not None:
            self.prefetch.evicted.zero_()
            self.prefetch.scores.zero_()
        bootstrap = self.adaptive and not self.samples[1][batch_size]
        if not self.adaptive:
            decision = torch.ones((), dtype=torch.bool, device=self.engine.device)
        elif not self.samples[0][batch_size]:
            decision = torch.zeros((), dtype=torch.bool, device=self.engine.device)
        elif bootstrap or self.rounds % 16 == 0:
            self.probe_depth = min(self.limit, self.depth_seen[batch_size] + 1)
            decision = torch.ones((), dtype=torch.bool, device=self.engine.device)
        else:
            ar = self._estimate(0, batch_size, batch_size)
            survival = self._survival(batch_size)
            draft_cost = torch.zeros_like(ar)
            best = torch.zeros((), dtype=torch.bool, device=self.engine.device)
            expected = torch.ones_like(ar)
            for step in range(max(lengths)):
                active = sum(length > step for length in lengths)
                draft_cost = draft_cost + self._estimate(1, active, active)
                logical = batch_size + sum(min(length, step + 1) for length in lengths)
                verify = self._estimate(2, batch_size, self._physical_verify(batch_size, logical), logical)
                expected = expected + survival[step] * active / batch_size
                best = best | (draft_cost + verify < expected * ar)
            decision = best
        resident, allowed = torch.stack((residency_ok, decision)).cpu().tolist()
        self.control_ms += (time.perf_counter() - started) * 1000
        eligible = sum(length > 0 for length in lengths)
        if not allowed:
            self.ar_requests += eligible
        elif resident and self.probe_depth:
            self.probe_requests += eligible
        return bool(allowed), bool(resident), 1 if bootstrap else self.limit

    def continue_draft(self, lengths, completed, batch_size):
        started = time.perf_counter()
        active = sum(length > completed for length in lengths)
        if not active:
            self.control_ms += (time.perf_counter() - started) * 1000
            return False
        decision = torch.ones((), dtype=torch.bool, device=self.engine.device)
        if self.adaptive and completed >= self.probe_depth:
            draft = self._estimate(1, active, active)
            logical = batch_size + sum(min(length, completed) for length in lengths)
            next_logical = logical + active
            verify = self._estimate(2, batch_size, self._physical_verify(batch_size, logical), logical)
            next_verify = self._estimate(2, batch_size, self._physical_verify(batch_size, next_logical), next_logical)
            benefit = self._estimate(0, active, active) * self._survival(batch_size)[completed]
            decision = draft + (next_verify - verify).clamp_min(0) < benefit
        # This is the single batch feedback for the next draft step. Timer reads
        # below are already complete and introduce no per-layer synchronization.
        tensors = [decision.float().reshape(1), self.copy_rows[1].float(), self.predicted_misses[1:2]]
        if self.prefetch is not None:
            tensors.append(self.prefetch.rows.float())
        packet = torch.cat(tensors).cpu().tolist()
        self._collect(1, packet[1:1 + self.layers], packet[1 + self.layers], packet[2 + self.layers:])
        self.control_ms += (time.perf_counter() - started) * 1000
        if not packet[0]:
            self.stopped_requests += active
        return bool(packet[0])

    def observe_acceptance(self, lengths, accepted):
        batch_size = len(lengths)
        drafted = torch.tensor(lengths, device=accepted.device)
        self.trials[batch_size] += (drafted[:, None] >= self.steps).sum(dim=0)
        self.accepted[batch_size] += (accepted[:, None] >= self.steps).sum(dim=0)
        self.depth_seen[batch_size] = max(self.depth_seen[batch_size], max(lengths))

    def snapshot(self):
        started = time.perf_counter()
        self.collect_ready()
        self.control_ms += (time.perf_counter() - started) * 1000
        result = {"cost_ar_requests": self.ar_requests, "cost_stopped_requests": self.stopped_requests,
                "cost_probe_requests": self.probe_requests, "cost_control_ms": self.control_ms,
                "cost_samples": {name: sum(self.samples[i]) for i, name in enumerate(self.phases)},
                "cost_gpu_ms": dict(self.gpu_ms), "cost_transfer_predictions": self.transfer_predictions}
        if self.prefetch is not None:
            result.update(self.prefetch.snapshot())
        return result

    def before_capture(self):
        self.collect_ready()
        prefetch = self.prefetch.before_capture() if self.prefetch is not None else None
        return self.use.clone(), prefetch

    def after_capture(self, state):
        use, prefetch = state
        self.use.copy_(use)
        self.routes.zero_()
        self.predicted.zero_()
        self.predicted_positions.zero_()
        self.copy_rows.zero_()
        if self.prefetch is not None:
            self.prefetch.after_capture(prefetch)

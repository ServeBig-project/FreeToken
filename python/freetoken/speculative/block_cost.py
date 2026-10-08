"""Choose each round's whole DFlash block on the host from measured whole-round costs.

A round costs what the engine stream spends from the decision to the queued replies, timed by
events read only after the scheduler's own output synchronization. Its yield is one token per
request plus the drafts the target accepts. Both are kept per executed shape, decayed by how
many decode rounds ago they were seen, and never extrapolated to shapes not measured.
"""
from __future__ import annotations

import time
from collections import OrderedDict, deque

import torch

_HALF_LIFE = 8          # decode rounds over which an old observation loses half its weight
_STALE = 128            # rounds after which an unrefreshed price no longer decides
_TABLE = 1024           # entries per table
_PROBE_SHARE = 0.03     # of measured round time saved up for probes
_PROBE_GAP = 16         # rounds between probes
_SWITCH = 0.05          # predicted gain to enter or change a draft length
_KEEP = 0.02            # predicted gain over AR a draft length must keep
_SLOTS = 4              # event sets in flight
_DECISION_BUCKET_US = 5


class RoundTimer:
    """Whole-round and segment times from a small ring of CUDA events, without waiting."""

    def __init__(self):
        event = lambda: torch.cuda.Event(enable_timing=True)
        self.free = [[event() for _ in range(5)] for _ in range(_SLOTS)]
        self.pending = deque()
        self.current = None
        self.dropped = 0

    def begin(self):
        if not self.free:
            self.collect_ready()
        if not self.free:
            self.dropped += 1  # never overwrite events still in flight
            self.current = None
            return
        self.current = [self.free.pop(), None, False]
        self.current[0][0].record()

    def describe(self, info):
        if self.current is not None:
            self.current[1] = info

    def mark(self, index):
        if self.current is not None:
            self.current[0][index].record()
            self.current[2] = True

    def end(self):
        if self.current is not None:
            self.current[0][4].record()
            self.pending.append(self.current)
            self.current = None

    def collect_ready(self):
        """(info, round ms, proposal ms, verify ms) of every finished round, oldest first."""
        out = []
        while self.pending and self.pending[0][0][4].query():
            events, info, marked = self.pending.popleft()
            self.free.append(events)
            if info is None:
                continue
            total = events[0].elapsed_time(events[4])
            segments = ((events[1].elapsed_time(events[2]), events[2].elapsed_time(events[3]))
                        if marked else (0.0, 0.0))
            out.append((info, total, *segments))
        return out


def _bounded_get(table, key):
    entry = table.get(key)
    if entry is not None:
        table.move_to_end(key)
    return entry


def _bounded_put(table, key, entry):
    table[key] = entry
    table.move_to_end(key)
    if len(table) > _TABLE:
        table.popitem(last=False)


class BlockController:
    """Per-round choice among AR and the legal 2/4/8 blocks, each clipped per request."""

    def __init__(self, engine, runtime):
        config = engine.config
        self.engine, self.runtime = engine, runtime
        self.limit = config.speculative_num_steps
        self.nominals = [0, *sorted({min(n, self.limit) for n in (2, 4, 8)})]
        self.observe_only = config.dflash_adaptive_observe_only
        # Drafter attention work per query position, by the history each layer group reads.
        groups = {}
        for limit in runtime.context.layout.limits:
            groups[limit] = groups.get(limit, 0) + 1
        self.groups = list(groups.items())
        self.timer = RoundTimer()
        self.stats = dict.fromkeys((
            "init_rounds", "probe_rounds", "selected_rounds", "fixed_rounds", "switches",
            "clipped_requests", "samples", "compared_rounds", "compared_candidates",
            "decisions"), 0)
        self.times = dict.fromkeys(("round_ms", "proposal_ms", "verify_ms", "init_ms",
                                    "probe_ms", "decision_us"), 0.0)
        self.decision_hist = [0] * 21
        self.executed = dict.fromkeys(self.nominals, 0)
        self.suggested = dict.fromkeys(self.nominals, 0)
        self._reset()

    def _reset(self):
        """A new serving geometry starts over: its prices and acceptance are unknown."""
        self.runner = self.engine.graph_runner
        self.costs, self.accepts = OrderedDict(), OrderedDict()
        self.round = 0
        self.init = list(self.nominals)
        self.current = 0          # the nominal length in use
        self.weak = 0             # consecutive rounds the current draft barely beats AR
        self.credit = 0.0
        self.since_probe = 0
        self.probe_at = 0
        self.recent_ms = None
        self.round_info = None
        for item in self.timer.pending:
            item[1] = None  # an older geometry's rounds price nothing here

    # ------------------------------------------------------------------ round
    def begin(self):
        self.timer.begin()

    def mark(self, index):
        self.timer.mark(index)

    def plan(self, batch, caps):
        started = time.perf_counter()
        if self.engine.graph_runner is not self.runner:
            self._reset()
        self._collect()
        self.round += 1
        reqs = batch.reqs
        greedy = [r.sampling_params.is_greedy for r in reqs]
        history = [r.cached_len for r in reqs]
        buckets = [h.bit_length() for h in history]
        # Nominal lengths clipped to the same per-request vector are one option.
        vectors = {n: tuple(min(n, cap) for cap in caps) for n in self.nominals}
        options = {}
        for nominal in self.nominals:
            options.setdefault(vectors[nominal], nominal)
        evaluate = lambda vector: self._score(vector, greedy, history, buckets)
        kind, nominal = self._decide(vectors, options, evaluate)
        if self.observe_only and kind != "init":
            self.suggested[nominal] += 1
            kind, nominal = "fixed", self.limit
        vector = vectors[nominal]
        self.executed[nominal] += 1
        self.stats[f"{kind}_rounds"] += 1
        self.stats["clipped_requests"] += sum(cap < nominal for cap in caps)
        key = self._cost_key(vector, greedy, history)
        accept = [(n, b, g) for n, b, g in zip(vector, buckets, greedy)]
        self.round_info = (kind, key, accept)
        self.timer.describe((kind, key))
        spent = (time.perf_counter() - started) * 1e6
        self.times["decision_us"] += spent
        self.stats["decisions"] += 1
        self.decision_hist[min(int(spent // _DECISION_BUCKET_US), 20)] += 1
        return list(vector)

    def observe(self, accepted):
        """``accepted[i]``: drafts of request i the target accepted, None when unknown."""
        info, self.round_info = self.round_info, None
        if info is None:
            return
        merged = {}
        for (n, bucket, greedy), a in zip(info[2], accepted, strict=True):
            if n and a is not None:
                counts = merged.setdefault((n, bucket, greedy), [0] * (n + 1))
                counts[0] += 1
                for j in range(1, a + 1):
                    counts[j] += 1
        for key, counts in merged.items():
            entry = _bounded_get(self.accepts, key)
            if entry is None:
                entry = [[0.0] * (len(counts) - 1), 0.0, self.round]
            decay = 2.0 ** (-(self.round - entry[2]) / _HALF_LIFE)
            entry[0] = [decay * old + new for old, new in zip(entry[0], counts[1:])]
            entry[1] = decay * entry[1] + counts[0]
            entry[2] = self.round
            _bounded_put(self.accepts, key, entry)

    def end(self):
        self.timer.end()

    # ------------------------------------------------------------------ estimates
    def _cost_key(self, vector, greedy, history):
        target = sum((n + 1) * h for n, h in zip(vector, history)).bit_length()
        if not any(vector):
            return ("ar", len(vector), target)
        drafter = sum((n + 1) * count * (h if limit is None else min(h, limit))
                      for n, h in zip(vector, history) for limit, count in self.groups)
        real = len(vector) + sum(vector)
        graphs = self.engine.graph_runner.speculative
        verify = graphs.verify_tokens(len(vector), real) if graphs is not None else real
        shape = tuple(sorted(zip(greedy, vector)))
        return (len(vector), shape, real, self.runtime.physical_tokens(len(vector), real),
                verify, target, drafter.bit_length())

    def _price(self, key):
        entry = _bounded_get(self.costs, key)
        if entry is None or self.round - entry[2] > _STALE:
            return None
        return entry[0] / entry[1]

    def _yield(self, vector, greedy, buckets):
        total = len(vector)
        for n, bucket, g in zip(vector, buckets, greedy):
            if not n:
                continue
            entry = _bounded_get(self.accepts, (n, bucket, g))
            if entry is None or self.round - entry[2] > _STALE:
                return None
            total += sum(entry[0]) / entry[1]
        return total

    def _score(self, vector, greedy, history, buckets):
        price = self._price(self._cost_key(vector, greedy, history))
        produced = self._yield(vector, greedy, buckets) if price is not None else None
        return None if produced is None else (price, price / produced)

    # ------------------------------------------------------------------ decision
    def _decide(self, vectors, options, evaluate):
        """(kind, nominal length to run) for this round."""
        if self.init:
            for vector, nominal in options.items():
                if nominal in self.init and (nominal == 0 or any(vector)):
                    self.init.remove(nominal)
                    return "init", nominal
        scores = {nominal: evaluate(vector) for vector, nominal in options.items()}
        known = {n: score for n, score in scores.items() if score is not None}
        if len(known) > 1:
            self.stats["compared_rounds"] += 1
            self.stats["compared_candidates"] += len(known)
        current = options[vectors[self.current]]
        probe = self._probe(scores, current)
        if probe is not None:
            return "probe", probe
        choice = self._choose(known, current)
        if choice != current:
            self.stats["switches"] += 1
            self.current = choice
        return "selected", self.current

    def _probe(self, scores, current):
        self.since_probe += 1
        unknown = (self.recent_ms or 0.0) * (self.limit + 1)
        reserve = lambda n: scores[n][0] if scores[n] is not None else unknown
        self.credit = min(self.credit, 2 * max(map(reserve, scores)))
        if self.since_probe < _PROBE_GAP or self.recent_ms is None:
            return None
        others = [n for n in scores if n != current]
        if not others:
            return None
        self.probe_at += 1
        nominal = others[self.probe_at % len(others)]
        if self.credit < reserve(nominal):
            return None
        self.since_probe = 0
        return nominal

    def _choose(self, known, current):
        if current not in known:
            return current  # keep running it to price this new shape
        best = min(known, key=lambda n: known[n][1])
        ar = known.get(0)
        if current and ar is not None:
            gain = (ar[1] - known[current][1]) / ar[1]
            self.weak = self.weak + 1 if gain < _KEEP else 0
            if self.weak >= 2:
                self.weak = 0
                return 0
        if best != current and known[best][1] < (1 - _SWITCH) * known[current][1]:
            self.weak = 0
            return best
        return current

    def _collect(self):
        for (kind, key), total, proposal, verify in self.timer.collect_ready():
            entry = _bounded_get(self.costs, key)
            if entry is None:
                entry = [0.0, 0.0, self.round]
            decay = 2.0 ** (-(self.round - entry[2]) / _HALF_LIFE)
            entry[:] = [decay * entry[0] + total, decay * entry[1] + 1, self.round]
            _bounded_put(self.costs, key, entry)
            self.recent_ms = total
            self.stats["samples"] += 1
            self.times["round_ms"] += total
            self.times["proposal_ms"] += proposal
            self.times["verify_ms"] += verify
            if kind == "init":
                self.times["init_ms"] += total
            elif kind == "probe":
                self.times["probe_ms"] += total
                self.credit -= total
            else:
                self.credit += _PROBE_SHARE * total

    def snapshot(self):
        self._collect()
        return {
            "dflash_control": "observe" if self.observe_only else "adaptive",
            **{f"dflash_{k}": v for k, v in self.stats.items()},
            **{f"dflash_{k}": v for k, v in self.times.items()},
            "dflash_rounds_dropped": self.timer.dropped,
            "dflash_executed_nominal": {str(n): c for n, c in self.executed.items()},
            "dflash_suggested_nominal": {str(n): c for n, c in self.suggested.items()},
            "dflash_decision_us_histogram": dict(bucket_us=_DECISION_BUCKET_US,
                                                 counts=list(self.decision_hist)),
            "dflash_timing_scope": (
                "round_ms: engine stream from the decision to the queued replies, including host "
                "gaps; proposal_ms: draft forward, probabilities, sampling and candidate writes; "
                "verify_ms: target verification forward; round_ms contains both"),
        }

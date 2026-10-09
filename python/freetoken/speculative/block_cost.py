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


class BlockController:
    """Per-round choice among AR and the legal 2/4/8 blocks, each clipped per request."""

    def __init__(self, engine, runtime):
        config = engine.config
        self.engine, self.runtime = engine, runtime
        self.limit = config.speculative_num_steps
        self.nominals = [0, *sorted({min(n, self.limit) for n in (2, 4, 8)})]
        self.observe_only = config.dflash_adaptive_observe_only
        # Drafter attention work per query position: full-history layers read all of it, the
        # others at most their limit.
        limits = runtime.context.layout.limits
        self.full_layers = limits.count(None)
        self.limited = [(limit, limits.count(limit)) for limit in set(limits) if limit is not None]
        self.timer = RoundTimer()
        self.stats = dict.fromkeys((
            "init_rounds", "probe_rounds", "selected_rounds", "fixed_rounds", "switches",
            "clipped_requests", "samples", "compared_rounds", "compared_candidates",
            "decisions", "round_ms", "proposal_ms", "verify_ms", "init_ms", "probe_ms",
            "decision_us"), 0)
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

    def plan(self, batch, caps, *, reserve=None):
        started = time.perf_counter()
        if self.engine.graph_runner is not self.runner:
            self._reset()
        self.round += 1
        reqs = batch.reqs
        # Requests alike in sampling, length cap and history bucket clip and score alike, so the
        # options are priced per group: [requests, history, drafter attention work].
        alike, members = {}, []
        for req, cap in zip(reqs, caps, strict=True):
            h = req.cached_len
            member = (req.sampling_params.is_greedy, cap, h.bit_length())
            members.append(member)
            group = alike.setdefault(member, [0, 0, 0])
            group[0] += 1
            group[1] += h
            work = self.full_layers * h
            for limit, count in self.limited:
                work += count * (h if h < limit else limit)
            group[2] += work
        groups = list(alike.items())
        # Nominal lengths clipped to the same per-group lengths are one option.
        clipped = {n: tuple(min(n, cap) for (_, cap, _), _ in groups) for n in self.nominals}
        options = {}
        for nominal in self.nominals:
            options.setdefault(clipped[nominal], nominal)
        keys = {lengths: self._cost_key(lengths, groups) for lengths in options}
        kind, nominal = self._decide(
            clipped, options, lambda lengths: self._score(lengths, keys[lengths], groups))
        if self.observe_only and kind != "init":
            self.suggested[nominal] += 1
            kind, nominal = "fixed", self.limit
        vector = [min(nominal, cap) for cap in caps]
        if reserve is not None:
            vector = reserve(vector)
        actual = dict(zip(members, vector))
        lengths = tuple(actual[member] for member, _ in groups)
        if not any(vector):
            nominal = 0
        self.executed[nominal] += 1
        self.stats[f"{kind}_rounds"] += 1
        self.stats["clipped_requests"] += sum(length < nominal for length in vector)
        key = keys[lengths] if lengths in keys else self._cost_key(lengths, groups)
        accept = [(n, r.cached_len.bit_length(), r.sampling_params.is_greedy)
                  for n, r in zip(vector, reqs)]
        self.round_info = (kind, key, accept)
        self.timer.describe((kind, key))
        spent = (time.perf_counter() - started) * 1e6
        self.stats["decision_us"] += spent
        self.stats["decisions"] += 1
        self.decision_hist[min(int(spent // _DECISION_BUCKET_US), 20)] += 1
        return vector

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
            self._update(self.accepts, key, counts[1:], counts[0])

    def end(self):
        """The round's replies are queued; take in the earlier rounds the GPU has finished."""
        self.timer.end()
        self._collect()

    # ------------------------------------------------------------------ estimates
    def _cost_key(self, lengths, groups):
        """The executed shape: batch, (sampling, length) histogram, real and physical draft and
        verify positions, target and drafter attention-work buckets."""
        batch = target = drafter = real = 0
        shape = {}
        for n, ((greedy, _, _), (count, history, work)) in zip(lengths, groups):
            batch += count
            target += (n + 1) * history
            drafter += (n + 1) * work
            real += count * (n + 1)
            shape[greedy, n] = shape.get((greedy, n), 0) + count
        if real == batch:
            return ("ar", batch, shape.get((False, 0), 0), target.bit_length())
        graphs = self.engine.graph_runner.speculative
        verify = graphs.verify_tokens(batch, real) if graphs is not None else real
        return (batch, tuple(sorted(shape.items())), real,
                self.runtime.physical_tokens(batch, real), verify, target.bit_length(),
                drafter.bit_length())

    def _update(self, table, key, values, weight):
        """Fold one round's sums and weight into ``key``'s decayed record."""
        entry = table.pop(key, None) or [[0.0] * len(values), 0.0, self.round]
        decay = 2.0 ** (-(self.round - entry[2]) / _HALF_LIFE)
        table[key] = [[decay * old + new for old, new in zip(entry[0], values)],
                      decay * entry[1] + weight, self.round]
        if len(table) > _TABLE:
            table.popitem(last=False)

    def _recent(self, table, key):
        """Mean per unit weight of ``key``'s sums, or None when unseen or stale."""
        entry = table.get(key)
        if entry is None or self.round - entry[2] > _STALE:
            return None
        table.move_to_end(key)
        return [value / entry[1] for value in entry[0]]

    def _score(self, lengths, key, groups):
        """(round price, price per produced token), or None while either is unknown."""
        price = self._recent(self.costs, key)
        if price is None:
            return None
        produced = 0
        for n, ((greedy, _, bucket), (count, _, _)) in zip(lengths, groups):
            produced += count
            if n:
                curve = self._recent(self.accepts, (n, bucket, greedy))
                if curve is None:
                    return None
                produced += count * sum(curve)
        return price[0], price[0] / produced

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
        # Wait for the next candidate in turn: cheaper ones must not spend its budget first.
        nominal = others[self.probe_at % len(others)]
        if self.credit < reserve(nominal):
            return None
        self.probe_at += 1
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
            self._update(self.costs, key, [total], 1)
            self.recent_ms = total
            self.stats["samples"] += 1
            self.stats["round_ms"] += total
            self.stats["proposal_ms"] += proposal
            self.stats["verify_ms"] += verify
            if kind in ("init", "probe"):
                self.stats[f"{kind}_ms"] += total
            if kind == "probe":
                self.credit -= total
            elif kind != "init":
                self.credit += _PROBE_SHARE * total

    def snapshot(self):
        self._collect()
        return {
            "dflash_control": "observe" if self.observe_only else "adaptive",
            **{f"dflash_{k}": v for k, v in self.stats.items()},
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

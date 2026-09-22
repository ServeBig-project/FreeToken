# S2-MoE implementation decisions

Reference: official `4090` branch at `3fbeb17`; paper §4.1–4.3.

## Draft expansion

Paper §4.1 defines benefit as prefix confidence times measured autoregressive
token latency. Cost is newly introduced expert-verification latency plus draft
expansion latency. In the official `speculative.cpp`, the denominator is
`delta_c + eps`: `eps` occupies the draft-cost term, not merely a tiny numerical
epsilon. FreeToken's `draft_step_ms` supplies that measured term. Cold-expert
latency is bytes per expert divided by calibrated bandwidth; experts on the
explicit permanent resident list and experts already observed in this request's
round are excluded. A fused model has zero expert-transfer cost.

Confidence uses the product of untempered draft token probabilities, including
when client sampling is greedy. Requested temperature/top-k/top-p still govern
the proposal distribution used by exact acceptance/rejection.

The reference greedy program can prune an already proposed token. FreeToken
supports random sampling, so it instead stops **before drawing the next
proposal**, using only the existing prefix, its observed routes and cumulative
confidence. Previously proposed tokens remain eligible for verification. This
keeps cost-based stopping from conditionally filtering the proposal distribution.
The first candidate is retained; configured token/resource limits still apply.

These are a single draft chain and a calibrated heuristic cost model. No tree
branches, adaptive expert caps, or resident-only drafter are introduced.

# S2-MoE implementation decisions

Reference: official `4090` branch at `3fbeb17`; paper §4.1–4.3.

## Draft expansion

Paper §4.1 defines benefit as prefix confidence times measured autoregressive
token latency. Cost is newly introduced expert-verification latency plus draft
expansion latency. In the official `speculative.cpp`, the denominator is
`delta_c + eps`, and the CLI calls `eps` a cost-denominator epsilon. The source
does not establish how the published example's fixed value was calibrated.
This port follows the paper's explicit draft-cost term and supplies the locally
measured `draft_step_ms` as that fixed term. Cold-expert
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

## Verification reuse

The official `ggml-cuda/moe-reuse.cu` uses a router-based confidence proxy:
`max(router_logits) - mean(router_logits)` for each token, divided by the
largest such weight in the group. It aggregates weighted original router logits
to select the preferred expert set; equal importance is ordered by expert ID.
The runtime bias is the group's mean `top1 - top(k+1)` router-logit margin.
It adds that bias only to selection scores. Expert mixture weights still come
from the original logits, with the checkpoint's existing normalization rule.

FreeToken applies this computation independently to each request's verification
span at each layer in one fused Triton program per request. Request offsets are
prepared once per verification batch, shared across all layers. The kernel
streams token rows so its working storage does not grow with draft length.
It preserves the original route/weights when the selected
set is unchanged. A one-token span, zero confidence/margin, or selection of all
experts is a no-op. The paper describes confidence weighting more generally;
this port follows the concrete public 4090 implementation above.

The target computation now depends on its own verification group, so this is
explicitly approximate relative to the original autoregressive model. Neither
per-request grouping nor the ordinary SD acceptance formula establishes exact
sampling equivalence to that original model when reuse is enabled.

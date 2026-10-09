"""Expectations and tolerances, fixed before any candidate run (contract section 5).

Do not relax after seeing candidate results.
"""
# GSM8K test[0:32], thinking off, greedy, max_tokens 768.
GSM_MIN_CORRECT = 26  # absolute floor for every configuration
GSM_MAX_DROP = 2  # a configuration vs its comparison baseline (fp8/int8 vs off; candidate vs upstream)

# Byte ratios derived from the public tensor shapes / config (head_dim 256, 2 KV heads, BF16 index).
KV_INT8_BYTES_RATIO = (0.49, 0.57)  # int8 per-token K/V bytes (with BF16 scales) / bf16

# Prefix reuse: an exact repeat of a prompt this long must reuse at least this fraction (radix sessions).
HIT_MIN_PROMPT = 1024
HIT_MIN_FRACTION = 0.5

# Cold restore: GPU pressure sent before re-access, as a multiple of the reported GPU KV token capacity.
PRESSURE_FACTOR = 1.5

IDLE_TIMEOUT_S = 120  # cancelled / finished work must drain without new arrivals

# --moe-cache-auto fills free VRAM with expert slots (2772480 B each) above a fixed KV floor, so memory
# freed by a precision option must show up as extra slots in the otherwise identical session.
# fp8 dense: half of the >= 3.30 GiB that FP8 storage of the BF16 projections + head saves (public shapes).
FP8_MIN_EXTRA_SLOTS = 640
# int8 KV: half of 32896 tokens x (25344 - 13152) B/token.
INT8_MIN_EXTRA_SLOTS = 72

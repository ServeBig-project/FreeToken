"""Public request workloads with self-checking quality oracles (no model reference needed).

Thresholds here were fixed before any service run and are not tuned afterwards.
"""

import difflib
import re
import subprocess
import sys

import torch

from harness import document

# --- code completion: the generated body must pass hidden unit tests -------------------
CODE_TASKS = [
    ("def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is a prime number.\"\"\"\n", "is_prime",
     [((2,), True), ((1,), False), ((97,), True), ((91,), False), ((7919,), True)]),
    ("def gcd(a: int, b: int) -> int:\n    \"\"\"Greatest common divisor of a and b.\"\"\"\n", "gcd",
     [((12, 18), 6), ((17, 5), 1), ((0, 9), 9)]),
    ("def reverse_words(s: str) -> str:\n    \"\"\"Reverse the order of the words in s (single spaces).\"\"\"\n",
     "reverse_words", [(("a b c",), "c b a"), (("hello world",), "world hello")]),
    ("def fib(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number, fib(0)=0 and fib(1)=1.\"\"\"\n", "fib",
     [((0,), 0), ((1,), 1), ((10,), 55), ((20,), 6765)]),
]
CODE_HEADER = "# Python 3 reference solutions.\n\n"
CODE_STOP = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "```", "\n# "]
CODE_MIN_PASS = 3  # of 4


def code_passes(signature, name, cases, body):
    program = (signature + body + "\n\nimport sys\n" +
               "".join(f"assert {name}(*{args!r}) == {want!r}\n" for args, want in cases))
    try:
        return subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        return False


# --- long-context retrieval: the needle sits more than one drafter window before the question ---
NEEDLE = "48213"


def needle_prompt(sentences=200, at=40):
    lines = document(5, sentences).split(" Note ")
    lines.insert(at, f"Important: the secret access code is {NEEDLE}.")
    return (" Note ".join(lines) + "\n\nQuestion: What is the secret access code mentioned in the notes above?\n"
            "Answer: The secret access code is")


# --- copying: high-acceptance, long generation with a measurable fidelity -------------------------
COPY_MIN_RATIO = 0.9


def copy_prompt(seed, sentences):
    source = document(seed, sentences)
    return f"Copy the following notes exactly, without any changes.\n\n{source}\n\nExact copy:\n", source


def copy_ratio(source, output):
    """Similarity over the overlapping length (output may stop at max_tokens)."""
    output = output.strip()[:len(source)]
    return difflib.SequenceMatcher(None, source[:len(output)], output, autojunk=False).ratio()


# --- coin tosses: sampled-distribution comparison ---------------------------------------------------
COIN_PROMPT = ("Below are 60 independent tosses of a fair coin, written as H or T separated by single spaces.\n\n"
               "T H H T H T T T H H T H T H H T")
COIN_FLIPS = 6
COIN_ALPHA = 1e-3


def coin_category(text):
    flips = re.findall(r"\b([HT])\b", text)
    return "bad" if len(flips) < COIN_FLIPS else str(flips[:COIN_FLIPS].count("H"))


def chi_square_two_sample(a, b):
    """2xK contingency chi-square; categories with < 10 pooled observations merge into 'pooled'."""
    keys = sorted(set(a) | set(b))
    merged_a, merged_b = {}, {}
    for key in keys:
        target = key if a.get(key, 0) + b.get(key, 0) >= 10 else "pooled"
        merged_a[target] = merged_a.get(target, 0) + a.get(key, 0)
        merged_b[target] = merged_b.get(target, 0) + b.get(key, 0)
    na, nb = sum(merged_a.values()), sum(merged_b.values())
    stat = 0.0
    for key in merged_a:
        total = merged_a[key] + merged_b[key]
        for observed, n in ((merged_a[key], na), (merged_b[key], nb)):
            expected = total * n / (na + nb)
            stat += (observed - expected) ** 2 / expected
    dof = max(len(merged_a) - 1, 1)
    p = float(torch.special.gammaincc(torch.tensor(dof / 2, dtype=torch.float64),
                                      torch.tensor(stat / 2, dtype=torch.float64)))
    return stat, dof, p


# --- public stats helpers -----------------------------------------------------------------------------
SPEC_COUNTERS = ("draft_tokens", "accepted_draft_tokens", "verify_steps", "emitted_tokens", "verify_positions",
                 "verify_physical_positions", "dflash_draft_positions", "dflash_draft_physical_positions",
                 "cost_ar_requests", "cost_stopped_requests", "cost_probe_requests")


def spec_delta(after, before):
    a, b = after["speculative"], before["speculative"]
    out = {k: a[k] - b.get(k, 0) for k in SPEC_COUNTERS if isinstance(a.get(k), (int, float))}  # keys appear lazily
    out["histogram"] = [x - y for x, y in zip(a["draft_length_histogram"], b["draft_length_histogram"])]
    out["completion_tokens"] = after["requests"]["completion_tokens_total"] - before["requests"]["completion_tokens_total"]
    out["requests"] = after["requests"]["completed"] - before["requests"]["completed"]
    return out


def acceptance(delta):
    return delta["accepted_draft_tokens"] / delta["draft_tokens"] if delta.get("draft_tokens") else 0.0


def counter_consistency(delta):
    """Bounds implied by the public counters (histogram[0] counts request-rounds that drafted nothing)."""
    bad = []
    hist = delta["histogram"]
    drafted_rounds, rounds = sum(hist[1:]), sum(hist)
    if sum(i * n for i, n in enumerate(hist)) != delta["draft_tokens"]:
        bad.append("sum(i*histogram[i]) != draft_tokens")
    if not delta["draft_tokens"] + drafted_rounds <= delta["verify_positions"] <= delta["draft_tokens"] + rounds:
        bad.append("verify_positions outside [draft + drafted rounds, draft + all rounds]")
    if delta["accepted_draft_tokens"] > delta["draft_tokens"]:
        bad.append("accepted > drafted")
    if delta["emitted_tokens"] > delta["completion_tokens"]:
        bad.append("emitted > completion tokens")
    if delta["emitted_tokens"] > delta["accepted_draft_tokens"] + rounds:
        bad.append("emitted > accepted + request-rounds")
    if delta["verify_physical_positions"] < delta["verify_positions"]:
        bad.append("physical verify positions < real")
    if delta.get("dflash_draft_physical_positions", 0) < delta.get("dflash_draft_positions", 0):
        bad.append("physical draft positions < real")
    return bad

"""Fixed tasks with known answers, built with the checkpoint's own tokenizer and chat template."""
import json
import os
import random
import re
import subprocess
import sys

from transformers import AutoTokenizer

from . import env

ADJ = "quiet bright narrow ancient silver distant gentle hollow crimson steady humble rapid golden".split()
NOUN = ("river lantern harbor meadow engine orchard canyon falcon ribbon summit glacier compass anchor "
        "prairie window ladder garden tunnel beacon saddle").split()
VERB = "noticed carried painted measured followed repaired guarded described".split()
# Each " word" is one token for this tokenizer; used to pad prompts to an exact token count.
PAD = ("river mountain lantern copper violet harbor apple table green house water stone garden window "
       "silver forest engine bridge yellow castle market winter summer paper").split()


class Tok:
    def __init__(self, path=env.MODEL):
        self.t = AutoTokenizer.from_pretrained(path)

    def n(self, text):
        return len(self.t.encode(text, add_special_tokens=False))

    def chat(self, messages):
        """Rendered prompt with thinking disabled (the template's empty think block)."""
        return self.t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)

    def prose(self, n_tokens, seed):
        rng, out, total = random.Random(seed), [], 0
        while total < n_tokens:
            s = (f"The {rng.choice(ADJ)} {rng.choice(NOUN)} near the {rng.choice(NOUN)} was "
                 f"{rng.choice(VERB)} by a {rng.choice(ADJ)} {rng.choice(NOUN)}.")
            out.append(s)
            total += self.n(" " + s)
        return " ".join(out)

    def needle_doc(self, n_tokens, seed, needle, frac):
        words = self.prose(n_tokens, seed).split(". ")
        k = int(len(words) * frac)
        return ". ".join(words[:k] + [needle.rstrip(".")] + words[k:])

    def exact(self, head, tail, length, seed):
        """Raw completion prompt of exactly `length` tokens: head + one-token pad words + tail."""
        rng = random.Random(seed)
        base = self.n(head) + self.n(tail)
        pad = "".join(" " + rng.choice(PAD) for _ in range(length - base))
        text = head + pad + tail
        assert self.n(text) == length, (self.n(text), length)
        return text


def answer_only(text):
    return text.split("</think>")[-1].strip()


def numbers(text):
    return [x.replace(",", "") for x in re.findall(r"-?\d[\d,]*\.?\d*", answer_only(text))]


def code_block(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[-1] if m else answer_only(text)


def run_code(code, tests):
    """Execute model code plus our asserts in a subprocess; True when every assert holds."""
    r = subprocess.run([sys.executable, "-I", "-c", code + "\n" + tests], capture_output=True, text=True,
                       timeout=20)
    return r.returncode == 0, r.stderr[-400:]


PRIME_TESTS = """
P = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47}
assert all(is_prime(i) == (i in P) for i in range(-3, 50))
assert is_prime(7919) and not is_prime(7917) and not is_prime(1)
"""
MERGE_TESTS = """
assert [list(x) for x in merge_intervals([[1,3],[2,6],[8,10],[15,18]])] == [[1,6],[8,10],[15,18]]
assert [list(x) for x in merge_intervals([[1,2],[2,3]])] == [[1,3]]
assert [list(x) for x in merge_intervals([[5,7],[1,2]])] == [[1,2],[5,7]]
assert list(merge_intervals([])) == []
"""
COVER_TESTS = MERGE_TESTS + """
assert covered_length([[1,3],[2,6],[8,10]]) == 7
assert covered_length([[0,1],[1,2],[5,9]]) == 6
assert covered_length([]) == 0
"""


def single_tasks(tok):
    """name -> (prompt, max_tokens, checker(text) -> bool)."""
    u = lambda c: tok.chat([{"role": "user", "content": c}])
    near = lambda n, seed, needle, frac: tok.needle_doc(n, seed, needle, frac)
    t = {
        "arith": (u("What is 17 multiplied by 23? Reply with only the number."), 16,
                  lambda x: numbers(x) == ["391"]),
        "capital": (u("What is the capital city of Australia? Reply with one word."), 16,
                    lambda x: "canberra" in answer_only(x).lower()),
        "code_prime": (u("Write a Python function `is_prime(n: int) -> bool` that returns True exactly when n "
                         "is a prime number (n may be negative or zero). Reply with only one Python code "
                         "block, no explanation."), 400,
                       lambda x: run_code(code_block(x), PRIME_TESTS)[0]),
        "needle_3k": (u(near(3000, 11, "The access code for the north gate is 4817.", 0.4)
                        + "\n\nWhat is the access code for the north gate? Reply with only the number."), 16,
                      lambda x: numbers(x) == ["4817"]),
        "needle_6k": (u(near(6000, 12, "The vault combination is 5093.", 0.75)
                        + "\n\nWhat is the vault combination? Reply with only the number."), 16,
                      lambda x: numbers(x) == ["5093"]),
        "needle_12k": (u(near(12000, 13, "The ferry departs from pier 62.", 0.15)
                         + "\n\nFrom which pier does the ferry depart? Reply with only the number."), 16,
                       lambda x: numbers(x) == ["62"]),
    }
    return t


RAIN = ("Field report. Station Alder recorded 312 mm of rain in March. Station Birch recorded 187 mm of rain "
        "in March. Station Cedar recorded 254 mm of rain in March. ")

# Multi-turn: list of (user content, max_tokens, checker or None). History uses the model's own replies.
CONVERSATIONS = {
    "trip": [
        ("I am planning a trip. My total budget is 1200 dollars and the trip lasts 5 days. "
         "Acknowledge in one short sentence.", 48, None),
        ("How many dollars per day can I spend if I split the budget evenly? Reply with only the number.", 16,
         lambda x: numbers(x) == ["240"]),
        ("If I add 2 more days with the same total budget, how many dollars per day is that, rounded down to "
         "an integer? Reply with only the number.", 16, lambda x: numbers(x) == ["171"]),
    ],
    "code_mt": [
        ("Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] integer pairs "
         "and returns the merged list of overlapping intervals sorted by start. Intervals that touch, like "
         "[1,2] and [2,3], must merge. Reply with only one Python code block.", 400,
         lambda x: run_code(code_block(x), MERGE_TESTS)[0]),
        ("Now add `covered_length(intervals)` that uses merge_intervals and returns the total length covered "
         "(sum of end - start over the merged intervals). Reply with only one Python code block containing "
         "both functions.", 500, lambda x: run_code(code_block(x), COVER_TESTS)[0]),
    ],
    "research_mt": [
        (None, 16, lambda x: "alder" in answer_only(x).lower()),  # content filled by conversation()
        ("What is the difference in March rainfall between that station and Station Birch, in mm? "
         "Reply with only the number.", 16, lambda x: numbers(x) == ["125"]),
    ],
}


def research_first(tok):
    doc = tok.prose(1500, 21)
    k = len(doc) // 2
    return (doc[:k] + " " + RAIN + doc[k:] + "\n\nWhich station recorded the most rain in March? "
            "Reply with the station name only.")


def gsm8k():
    with open(os.path.join(os.path.dirname(__file__), "gsm8k_test32.json")) as f:
        items = json.load(f)
    return [(it["id"], it["question"] + "\nSolve it step by step briefly, then give the final answer on the "
             "last line as 'Answer: <number>'.", it["answer"]) for it in items]


def gsm8k_ok(text, gold):
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", answer_only(text))
    if not m:
        return False
    try:
        return abs(float(m[-1].replace(",", "")) - float(gold)) < 1e-6
    except ValueError:
        return False


# Exact prompt lengths around multiples of 4 and 64, the 1024-token prefill chunk and 2048 (QSA budget).
BOUNDARY_LENGTHS = [61, 62, 63, 64, 65, 66, 67, 127, 128, 129, 130, 255, 256, 257, 511, 513, 1023, 1024, 1025,
                    2047, 2048, 2049, 2052, 3071, 3072, 3073]


def boundary_prompt(tok, length):
    word = PAD[length % len(PAD)]
    code = 1000 + (length * 37) % 9000
    phrase = f"{word}-{code}"
    head = f"<|im_start|>user\nThe passphrase is {phrase}. Remember it. Ignore the noise words that follow.\n"
    tail = ("\nWhat is the passphrase? Reply with only the passphrase.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return tok.exact(head, tail, length, seed=length), phrase


def degenerate(text, window=12, repeats=4):
    """True when one character window repeats back-to-back `repeats` times (a decoding loop)."""
    s = answer_only(text)
    for i in range(0, max(0, len(s) - window * repeats)):
        w = s[i:i + window]
        if w.strip() and s[i:i + window * repeats] == w * repeats:
            return True
    return False

"""Subprocess body for the Python `LLM` entry: build, generate, print one JSON line."""
import json
import os
import sys

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM

model, port, kwargs = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
try:
    llm = LLM(model, dtype=torch.bfloat16, distributed_port=port, **kwargs)
except Exception as e:  # report the public error text to the parent test
    print("RESULT " + json.dumps({"error": f"{type(e).__name__}: {e}"}))
    sys.stdout.flush()
os._exit(0)
    os._exit(3)
out = llm.generate(["Continue the list: 1, 2, 3,", "Once upon a time"],
                   SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=24))
print("RESULT " + json.dumps({"outputs": [{k: (len(v) if isinstance(v, list) else v) for k, v in o.items()}
                                          for o in out]}))
sys.stdout.flush()
os._exit(0)

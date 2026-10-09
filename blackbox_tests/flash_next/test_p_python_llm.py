"""Python `LLM` with the documented dense_quantization / kv_dtype options (fp8 + int8)."""
import json
import os
import subprocess

from . import env, tasks
from .conftest import load, save
from .scenarios import consistent
from .sessions import session

SCRIPT = r"""
import json, sys
from freetoken.llm import LLM
from freetoken.core import SamplingParams
cases = json.load(open(sys.argv[1]))
llm = LLM(sys.argv[2], dense_quantization="fp8", kv_dtype="int8")
outs = llm.generate([c["prompt"] for c in cases], [SamplingParams(temperature=0.0, max_tokens=c["max_tokens"])
                                                    for c in cases])
json.dump([{k: v for k, v in o.items()} for o in outs], open(sys.argv[3], "w"), default=str)
llm.shutdown()
"""


def test_python_llm_fp8_int8():
    tok = tasks.Tok()
    single = tasks.single_tasks(tok)
    names = ["arith", "capital", "code_prime", "needle_3k"]
    cases = [dict(name=n, prompt=single[n][0], max_tokens=single[n][1]) for n in names]
    for L in (1025, 2049):
        p, ph = tasks.boundary_prompt(tok, L)
        cases.append(dict(name=f"boundary{L}", prompt=p, max_tokens=12, phrase=ph))
    os.makedirs(env.RESULTS, exist_ok=True)
    cin, cout = os.path.join(env.RESULTS, "P_cases.json"), os.path.join(env.RESULTS, "P_out.json")
    with open(cin, "w") as f:
        json.dump(cases, f)
    e = dict(os.environ, PYTHONPATH=os.path.join(env.IMPL, "python"))
    with open(os.path.join(env.RESULTS, "P_python_llm.log"), "w") as log:
        r = subprocess.run([env.PY, "-c", SCRIPT, cin, env.MODEL, cout], stdout=log, stderr=subprocess.STDOUT,
                           env=e, cwd=env.RESULTS, timeout=env.STARTUP_TIMEOUT + 1800)
    assert r.returncode == 0, f"LLM script exit {r.returncode}; see P_python_llm.log"
    with open(cout) as f:
        outs = json.load(f)
    assert len(outs) == len(cases)
    texts = [o.get("text") if isinstance(o.get("text"), str) else tok.t.decode(o["token_ids"],
             skip_special_tokens=True) for o in outs]
    save("P_python_llm", dict(outputs=dict(zip([c["name"] for c in cases], texts))))
    fails = []
    for c, t in zip(cases, texts):
        chk = (lambda x, ph=c["phrase"]: tasks.answer_only(x).startswith(ph)) if "phrase" in c \
            else single[c["name"]][2]
        if not chk(t):
            fails.append(f"{c['name']}: {t[:200]!r}")
        for key in ("C", "D"):  # same precision over HTTP
            other = load(session(key)["name"])
            ref = (other or {}).get("alone", {}).get(c["name"])
            if ref and not consistent(t, ref["text"], chk):
                fails.append(f"{c['name']}: Python LLM vs session {key} inconsistent")
    assert not fails, fails

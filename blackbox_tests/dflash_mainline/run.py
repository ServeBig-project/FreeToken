"""Run one service session (one launch, many scenarios) or the cross-session comparisons.

  python run.py --session nvfp4_n8 [--only name,...]
  python run.py --compare

Results: $FT_LOG_DIR/<session>.json (checks + artifacts); comparisons in $FT_LOG_DIR/compare.json.
Scenario functions take (server, checks) and return a JSON-serialisable artifact.
"""

import argparse
import importlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import LOG_DIR, SOURCE, Server, StartupError, process_gpu_mib  # noqa: E402

PHASES = [m for m in ("p1", "p2", "p3", "p4", "p5") if (Path(__file__).parent / f"{m}.py").exists()]


class Checks:
    def __init__(self, session):
        self.session, self.rows = session, []

    def check(self, name, ok, **detail):
        row = {"session": self.session, "check": name, "ok": bool(ok), **detail}
        self.rows.append(row)
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else " " + json.dumps(detail, default=str)[:600]),
              flush=True)
        return ok

    def note(self, name, **detail):
        self.rows.append({"session": self.session, "check": name, "ok": None, **detail})
        print("NOTE " + name + " " + json.dumps(detail, default=str)[:400], flush=True)


def responsive(server):
    try:
        server.stats()
        return True
    except Exception:
        return False


def registry():
    configs, plan, compares, aliases = {}, {}, [], {}
    for name in PHASES:
        module = importlib.import_module(name)
        configs.update(getattr(module, "CONFIGS", {}))
        aliases.update(getattr(module, "ALIASES", {}))
        for session, functions in getattr(module, "PLAN", {}).items():
            plan.setdefault(session, []).extend(functions)
        compares.extend(getattr(module, "COMPARE", []))
    for alias, source in aliases.items():
        plan.setdefault(alias, []).extend(f for f in plan.get(source, []) if f not in plan[alias])
    return configs, plan, compares


def run_session(session, only):
    configs, plan, _ = registry()
    checks = Checks(session)
    result = {"session": session, "artifacts": {}}
    functions = [f for f in plan.get(session, []) if not only or f.__name__ in only]
    config = configs[session]
    if callable(config):  # sessions that do not keep a server (e.g. expected startup rejections)
        result["artifacts"]["startup"] = config(checks)
    else:
        started = time.monotonic()
        result["impl_head"] = subprocess.run(["git", "-C", str(Path(SOURCE).parent), "rev-parse", "HEAD"],
                                             capture_output=True, text=True).stdout.strip()
        print(f"impl HEAD {result['impl_head']}", flush=True)
        try:
            server = Server(session, config)
        except StartupError as error:
            checks.check("startup", False, error=str(error), log_tail=error.log_tail)
            server = None
        if server is not None:
            result["cmd"] = server.cmd
            result["startup_seconds"] = time.monotonic() - started
            result["status0"], result["stats0"] = server.status(), server.stats()
            result["process_gpu_mib"] = process_gpu_mib(server.proc.pid)
            try:
                for function in functions:
                    print(f"--- {session}:{function.__name__}", flush=True)
                    try:
                        result["artifacts"][function.__name__] = function(server, checks)
                    except Exception as error:  # a scenario failure must not hide the others
                        checks.check(f"{function.__name__}:error", False, error=repr(error),
                                     trace=traceback.format_exc()[-2000:])
                    if not server.alive() or not responsive(server):
                        checks.check("server_alive", False, after=function.__name__)
                        break
                if server.alive():
                    result["stats_end"] = server.stats()
                    result["status_end"] = server.status()
            finally:
                server.close()
    result["checks"] = checks.rows
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{session}.json"
    if only and path.exists():  # merge a partial rerun into the earlier session result
        old = json.loads(path.read_text())
        old["artifacts"].update(result["artifacts"])
        old["checks"] = [r for r in old["checks"] if r["check"].split(":")[0] not in only] + checks.rows
        result = {**old, **{k: v for k, v in result.items() if k not in ("artifacts", "checks")}}
    path.write_text(json.dumps(result, indent=1, default=str))
    failed = [r for r in result["checks"] if r["ok"] is False]
    print(f"== {session}: {sum(r['ok'] is True for r in result['checks'])} pass, {len(failed)} fail")
    return not failed


def run_compare():
    _, _, compares = registry()
    sessions = {p.stem: json.loads(p.read_text()) for p in LOG_DIR.glob("*.json") if p.stem != "compare"}
    checks = Checks("compare")
    for function in compares:
        print(f"--- compare:{function.__name__}", flush=True)
        try:
            function(sessions, checks)
        except Exception as error:
            checks.check(f"{function.__name__}:error", False, error=repr(error), trace=traceback.format_exc()[-1500:])
    (LOG_DIR / "compare.json").write_text(json.dumps(checks.rows, indent=1, default=str))
    return not [r for r in checks.rows if r["ok"] is False]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session")
    parser.add_argument("--only", default="")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        configs, plan, _ = registry()
        for name in configs:
            print(name, [f.__name__ for f in plan.get(name, [])])
        sys.exit(0)
    ok = run_compare() if args.compare else run_session(args.session, set(filter(None, args.only.split(","))))
    sys.exit(0 if ok else 1)

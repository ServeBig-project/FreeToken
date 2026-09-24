"""Execute one generated function against its independent finite public contract."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import sys

task = json.load(sys.stdin)
result = {"passed": False, "passed_cases": 0, "total_cases": len(task["cases"])}
captured = io.StringIO()
try:
    with redirect_stdout(captured), redirect_stderr(captured):
        namespace = {}
        exec(compile(task["code"], "<generated>", "exec"), namespace)
        function = namespace[task["name"]]
        for index, (args, expected) in enumerate(task["cases"]):
            actual = function(*args)
            if actual != expected:
                result.update(failed_case=index, expected=expected, actual=repr(actual))
                break
            result["passed_cases"] += 1
        else:
            result["passed"] = True
except BaseException as error:
    result["error"] = f"{type(error).__name__}: {error}"
result["stdout_stderr"] = captured.getvalue()
print(json.dumps(result))

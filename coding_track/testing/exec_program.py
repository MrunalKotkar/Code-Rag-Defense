"""Execute one test program in its own process (standard library only).

Runs on: the test runtime, under a benchmark environment's Python, e.g.
    /content/envs/ds1000/bin/python exec_program.py job.json

Mirrors DS-1000's execution.py (exec in a fresh temporary directory, stdout and
stderr swallowed), but one process per answer: the parent enforces the time
limit by killing the process. Exit code 0 = passed, 1 = failed; the last line on
stderr says why.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import traceback


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        program = json.load(f)["program"]
    real_stderr = sys.stderr
    start_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as workdir:
        os.chdir(workdir)
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                # Empty globals, as in DS-1000's execution.py and HF code_eval.
                exec(compile(program, "<test_program>", "exec"), {})
        except BaseException as e:  # same as DS-1000: any exception is a failure
            last = traceback.format_exception_only(type(e), e)[-1].strip()
            print(f"FAILED: {last[:500]}", file=real_stderr)
            return 1
        finally:
            os.chdir(start_dir)   # leave the folder before it is deleted (as DS-1000's chdir does)
    return 0


if __name__ == "__main__":
    sys.exit(main())

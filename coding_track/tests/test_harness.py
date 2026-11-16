from coding_track.testing.harness import defines_function, ds1000_program, odex_program

ODEX_ROW = {"prompt": "def f_1():\n\treturn ", "suffix": "", "entry_point": "f_1",
            "test_start": "\ndef check(candidate):", "test": ["\n    assert candidate() == 3\n",
                                                           "\n    assert candidate() + 1 == 4\n"]}


def test_ds1000_program_matches_official_construction():
    problem = {"code_context": "def test_execution(s): pass\ndef test_string(s): pass"}
    sol = "result = df.iloc[List]\n"
    # Exactly what DS-1000's test_ds1000.py builds:
    official = (problem["code_context"] + "\n" + f"code = {repr(sol)}\n" + "test_execution(code)\n"
                + "test_string(code)\n")
    assert ds1000_program(problem, sol) == official


def test_ds1000_program_without_test_string():
    assert ds1000_program({"code_context": "def test_execution(s): pass"}, "x").endswith(
        "test_execution(code)\n\n")


def test_odex_reference_is_run_as_completion_with_all_tests():
    prog = odex_program(ODEX_ROW, "1 + 2", is_reference=True)
    assert prog.startswith("def f_1():\n    return 1 + 2\n")
    assert prog.count("assert candidate()") == 2            # every test case, not a sample
    assert prog.rstrip().endswith("check(f_1)")


def test_odex_full_function_answer_is_used_as_is():
    answer = "import math\ndef f_1():\n    return 3"
    assert odex_program(ODEX_ROW, answer).startswith(answer)


def test_odex_bare_expression_answer_becomes_completion():
    assert odex_program(ODEX_ROW, "3").startswith("def f_1():\n    return 3")


def test_odex_answer_with_its_own_return_is_the_function_body():
    assert odex_program(ODEX_ROW, "return 3\n").startswith("def f_1():\n    return 3\n")
    body = "import math\nx = math.floor(3.5)\nreturn x\n"
    prog = odex_program(ODEX_ROW, body)
    assert prog.startswith("def f_1():\n    import math\n    x = math.floor(3.5)\n    return x\n")
    exec(prog, {})                                                       # runs and passes
    # a reference solution is always a completion
    assert odex_program(ODEX_ROW, "3", is_reference=True).startswith("def f_1():\n    return 3")


def test_defines_function():
    assert defines_function("def f_1(a):\n  return a", "f_1")
    assert not defines_function("def g():\n  pass", "f_1")
    assert defines_function("def f_1(:\n  broken", "f_1")      # unparsable: text fallback


def test_odex_programs_execute(tmp_path):
    ns = {}
    exec(odex_program(ODEX_ROW, "def f_1():\n    return 3"), ns)          # passes
    try:
        exec(odex_program(ODEX_ROW, "def f_1():\n    return 4"), {})
    except AssertionError:
        pass
    else:
        raise AssertionError("a wrong answer must fail the ODEX tests")

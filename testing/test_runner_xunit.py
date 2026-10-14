# mypy: allow-untyped-defs
"""Test correct setup/teardowns at module, class, and instance level."""

from __future__ import annotations

from pathlib import Path

from _pytest.ensemble import build_module
from _pytest.ensemble import run_tests
from _pytest.pytester import Pytester
import pytest


# ensemble: kept on pytester as a canary. This is the only test here that
# exercises xunit setup through a real module import, where ``modlevel`` is a
# genuine module global shared between the xunit hooks and the test functions.
# An ensemble source keeps the *host* module's globals, so the ported tests
# below thread state through closures instead - which is exactly the mechanism
# this test is about, so it stays file-based.
def test_module_and_function_setup(pytester: Pytester) -> None:
    reprec = pytester.inline_runsource(
        """
        modlevel = []
        def setup_module(module):
            assert not modlevel
            module.modlevel.append(42)

        def teardown_module(module):
            modlevel.pop()

        def setup_function(function):
            function.answer = 17

        def teardown_function(function):
            del function.answer

        def test_modlevel():
            assert modlevel[0] == 42
            assert test_modlevel.answer == 17

        class TestFromClass(object):
            def test_module(self):
                assert modlevel[0] == 42
                assert not hasattr(test_modlevel, 'answer')
    """
    )
    rep = reprec.matchreport("test_modlevel")
    assert rep.passed
    rep = reprec.matchreport("test_module")
    assert rep.passed


def test_module_setup_failure_no_teardown(tmp_path: Path) -> None:
    values: list[int] = []

    def setup_module(module):
        values.append(1)
        raise ZeroDivisionError

    def test_nothing(): ...

    def teardown_module(module):
        values.append(2)

    module = build_module(
        "test_xunit_module_setup_failure", setup_module, test_nothing, teardown_module
    )
    record = run_tests(module, rootpath=tmp_path)
    # an xunit setup_module failure errors in the *setup* phase; the old
    # assertoutcome(failed=1) counted any failed report regardless of phase.
    record.assert_outcomes(errors=1)
    assert record["test_nothing"].setup is not None
    assert record["test_nothing"].setup.failed
    # teardown_module never ran
    assert values == [1]


def test_setup_function_failure_no_teardown(tmp_path: Path) -> None:
    modlevel: list[int] = []

    def setup_function(function):
        modlevel.append(1)
        raise ZeroDivisionError

    def teardown_function(module):
        modlevel.append(2)

    def test_func(): ...

    module = build_module(
        "test_xunit_function_setup_failure",
        setup_function,
        teardown_function,
        test_func,
    )
    record = run_tests(module, rootpath=tmp_path)
    # the original only looked at the recorded module globals; the setup
    # failure is an error, and asserting it is strictly more than before
    record.assert_outcomes(errors=1)
    # teardown_function never ran
    assert modlevel == [1]


def test_class_setup(tmp_path: Path) -> None:
    class TestSimpleClassSetup:
        clslevel: list[int] = []

        def setup_class(cls):
            cls.clslevel.append(23)

        def teardown_class(cls):
            cls.clslevel.pop()

        def test_classlevel(self):
            assert self.clslevel[0] == 23

    class TestInheritedClassSetupStillWorks(TestSimpleClassSetup):
        def test_classlevel_anothertime(self):
            assert self.clslevel == [23]

    def test_cleanup():
        assert not TestSimpleClassSetup.clslevel
        assert not TestInheritedClassSetupStillWorks.clslevel

    # collection follows argument order, so test_cleanup runs last
    record = run_tests(
        TestSimpleClassSetup,
        TestInheritedClassSetupStillWorks,
        test_cleanup,
        rootpath=tmp_path,
    )
    record.assert_outcomes(passed=1 + 2 + 1)
    assert TestSimpleClassSetup.clslevel == []


def test_class_setup_failure_no_teardown(tmp_path: Path) -> None:
    class TestSimpleClassSetup:
        clslevel: list[int] = []

        def setup_class(cls):
            raise ZeroDivisionError

        def teardown_class(cls):
            cls.clslevel.append(1)

        def test_classlevel(self): ...

    def test_cleanup():
        assert not TestSimpleClassSetup.clslevel

    # collection follows argument order, so test_cleanup runs last
    record = run_tests(TestSimpleClassSetup, test_cleanup, rootpath=tmp_path)
    # setup_class fails in the *setup* phase, so this is an error, not a
    # failure - assertoutcome(failed=1) did not distinguish the two.
    record.assert_outcomes(errors=1, passed=1)
    # teardown_class never ran
    assert TestSimpleClassSetup.clslevel == []


def test_method_setup(tmp_path: Path) -> None:
    class TestSetupMethod:
        def setup_method(self, meth):
            self.methsetup = meth

        def teardown_method(self, meth):
            del self.methsetup

        def test_some(self):
            assert self.methsetup == self.test_some

        def test_other(self):
            assert self.methsetup == self.test_other

    run_tests(TestSetupMethod, rootpath=tmp_path).assert_outcomes(passed=2)


def test_method_setup_failure_no_teardown(tmp_path: Path) -> None:
    class TestMethodSetup:
        clslevel: list[int] = []

        def setup_method(self, method):
            self.clslevel.append(1)
            raise ZeroDivisionError

        def teardown_method(self, method):
            self.clslevel.append(2)

        def test_method(self): ...

    def test_cleanup():
        assert TestMethodSetup.clslevel == [1]

    # collection follows argument order, so test_cleanup runs last
    record = run_tests(TestMethodSetup, test_cleanup, rootpath=tmp_path)
    # setup_method fails in the *setup* phase, so this is an error, not a
    # failure - assertoutcome(failed=1) did not distinguish the two.
    record.assert_outcomes(errors=1, passed=1)
    # teardown_method never ran
    assert TestMethodSetup.clslevel == [1]


def test_method_setup_uses_fresh_instances(tmp_path: Path) -> None:
    class TestSelfState1:
        memory: list[object] = []

        def test_hello(self):
            self.memory.append(self)

        def test_afterhello(self):
            assert self != self.memory[0]

    run_tests(TestSelfState1, rootpath=tmp_path).assert_outcomes(passed=2, failed=0)


# ensemble: kept on pytester as a canary. Together with the ported
# test_setup_fails_again_on_all_tests below it covers the same behaviour once
# more through a real file collected from a path argument, so path-based
# collection and nodeids stay exercised in this module.
def test_setup_that_skips_calledagain(pytester: Pytester) -> None:
    p = pytester.makepyfile(
        """
        import pytest
        def setup_module(mod):
            pytest.skip("x")
        def test_function1():
            pass
        def test_function2():
            pass
    """
    )
    reprec = pytester.inline_run(p)
    reprec.assertoutcome(skipped=2)


def test_setup_fails_again_on_all_tests(tmp_path: Path) -> None:
    def setup_module(mod):
        raise ValueError(42)

    def test_function1(): ...

    def test_function2(): ...

    module = build_module(
        "test_xunit_setup_fails_again", setup_module, test_function1, test_function2
    )
    record = run_tests(module, rootpath=tmp_path)
    # the module setup failure is re-raised for every test, in the *setup*
    # phase - so two errors, where assertoutcome counted two failed reports.
    record.assert_outcomes(errors=2)
    assert record["test_function1"].failed
    assert record["test_function2"].failed


def test_setup_funcarg_setup_when_outer_scope_fails(tmp_path: Path) -> None:
    hello_calls: list[object] = []

    def setup_module(mod):
        raise ValueError(42)

    @pytest.fixture
    def hello(request):
        hello_calls.append(request)
        raise ValueError("xyz43")

    def test_function1(hello): ...

    def test_function2(hello): ...

    module = build_module(
        "test_xunit_outer_scope_fails",
        setup_module,
        hello,
        test_function1,
        test_function2,
    )
    record = run_tests(module, rootpath=tmp_path, capture_output=True)
    record.assert_outcomes(errors=2)
    record.stdout.fnmatch_lines(
        [
            "*function1*",
            "*ValueError*42*",
            "*function2*",
            "*ValueError*42*",
            "*2 errors*",
        ]
    )
    record.stdout.no_fnmatch_line("*xyz43*")
    # the inner fixture is never even reached
    assert hello_calls == []


def _xunit_hooks_without_argument(trace: list[str]) -> dict[str, object]:
    """The xunit functions of the test below, all without the optional argument."""

    def setup_module():
        trace.append("setup_module")

    def teardown_module():
        trace.append("teardown_module")

    def setup_function():
        trace.append("setup_function")

    def teardown_function():
        trace.append("teardown_function")

    def test_function_1(): ...

    def test_function_2(): ...

    class Test:
        def setup_method(self):
            trace.append("setup_method")

        def teardown_method(self):
            trace.append("teardown_method")

        def test_method_1(self): ...

        def test_method_2(self): ...

    return dict(
        setup_module=setup_module,
        teardown_module=teardown_module,
        setup_function=setup_function,
        teardown_function=teardown_function,
        test_function_1=test_function_1,
        test_function_2=test_function_2,
        Test=Test,
    )


def _xunit_hooks_with_argument(trace: list[str]) -> dict[str, object]:
    """The xunit functions of the test below, all taking the optional argument."""

    def setup_module(arg):
        trace.append("setup_module")

    def teardown_module(arg):
        trace.append("teardown_module")

    def setup_function(arg):
        trace.append("setup_function")

    def teardown_function(arg):
        trace.append("teardown_function")

    def test_function_1(): ...

    def test_function_2(): ...

    class Test:
        def setup_method(self, arg):
            trace.append("setup_method")

        def teardown_method(self, arg):
            trace.append("teardown_method")

        def test_method_1(self): ...

        def test_method_2(self): ...

    return dict(
        setup_module=setup_module,
        teardown_module=teardown_module,
        setup_function=setup_function,
        teardown_function=teardown_function,
        test_function_1=test_function_1,
        test_function_2=test_function_2,
        Test=Test,
    )


@pytest.mark.parametrize(
    "hooks",
    [_xunit_hooks_without_argument, _xunit_hooks_with_argument],
    ids=["", "arg"],
)
def test_setup_teardown_function_level_with_optional_argument(
    tmp_path: Path,
    hooks,
) -> None:
    """Parameter to setup/teardown xunit-style functions parameter is now optional (#1728)."""
    trace_setups_teardowns: list[str] = []

    # the members keep the order they are handed to build_module in, which is
    # what decides collection order here
    module = build_module(
        "test_xunit_optional_argument", **hooks(trace_setups_teardowns)
    )
    run_tests(module, rootpath=tmp_path).assert_outcomes(passed=4)

    expected = [
        "setup_module",
        "setup_function",
        "teardown_function",
        "setup_function",
        "teardown_function",
        "setup_method",
        "teardown_method",
        "setup_method",
        "teardown_method",
        "teardown_module",
    ]
    assert trace_setups_teardowns == expected

# mypy: allow-untyped-defs
from __future__ import annotations

import abc
import gc
from pathlib import Path
import sys
import unittest

import _pytest._code
from _pytest.config import ExitCode
from _pytest.ensemble import build_module
from _pytest.ensemble import collect_tests
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import module_from_path
from _pytest.ensemble import run_tests
from _pytest.monkeypatch import MonkeyPatch
from _pytest.outcomes import Exit
from _pytest.pytester import Pytester
import pytest


#: The example scripts, run as themselves rather than copied somewhere first.
EXAMPLES = Path(__file__).parent / "example_scripts"


def test_simple_unittest(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        def testpassing(self):
            self.assertEqual("foo", "foo")

        def test_failing(self):
            self.assertEqual("foo", "bar")

    record = run_tests(MyTestCase, rootpath=tmp_path)
    assert record["testpassing"].passed
    assert record["test_failing"].failed


def test_runTest_method(tmp_path: Path) -> None:
    class MyTestCaseWithRunTest(unittest.TestCase):
        def runTest(self):
            self.assertEqual("foo", "foo")

    class MyTestCaseWithoutRunTest(unittest.TestCase):
        def runTest(self):
            self.assertEqual("foo", "foo")

        def test_something(self):
            pass

    sources = (MyTestCaseWithRunTest, MyTestCaseWithoutRunTest)
    items = collect_tests(*sources, rootpath=tmp_path)
    assert [item.nodeid.split("::", 1)[1] for item in items] == [
        "MyTestCaseWithRunTest::runTest",
        "MyTestCaseWithoutRunTest::test_something",
    ]
    run_tests(*sources, rootpath=tmp_path).assert_outcomes(passed=2)


def test_isclasscheck_issue53(tmp_path: Path) -> None:
    class _E:
        def __getattr__(self, tag):
            pass

    module = build_module("test_isclasscheck_issue53", E=_E())
    assert collect_tests(module, rootpath=tmp_path) == []


def test_setup(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        def setUp(self):
            self.foo = 1

        def setup_method(self, method):
            self.foo2 = 1

        def test_both(self):
            self.assertEqual(1, self.foo)
            assert self.foo2 == 1

        def teardown_method(self, method):
            assert 0, "42"

    record = run_tests(MyTestCase, rootpath=tmp_path)
    call = record["test_both"].call
    assert call is not None and call.passed
    teardown = record["test_both"].teardown
    assert teardown is not None
    assert teardown.failed and "42" in teardown.longreprtext


def test_setUpModule(tmp_path: Path) -> None:
    # the module level ``values`` of the original became a closure: a source
    # collected in-memory keeps this module's globals, so a module level list
    # would be *this* file's.
    values = []

    def setUpModule():
        values.append(1)

    def tearDownModule():
        del values[0]

    def test_hello():
        assert values == [1]

    def test_world():
        assert values == [1]

    module = build_module(
        "test_setUpModule", setUpModule, tearDownModule, test_hello, test_world
    )
    run_tests(module, rootpath=tmp_path).assert_outcomes(passed=2)
    assert values == []


def test_setUpModule_failing_no_teardown(tmp_path: Path) -> None:
    values = []

    def setUpModule():
        0 / 0  # noqa: B018

    def tearDownModule():
        values.append(1)

    def test_hello():
        pass

    module = build_module(
        "test_setUpModule_failing_no_teardown",
        setUpModule,
        tearDownModule,
        test_hello,
    )
    record = run_tests(module, rootpath=tmp_path)
    # setUpModule is an xunit *setup* fixture, so the terminal category is
    # "error" where HookRecorder.assertoutcome() only counted a failed report.
    record.assert_outcomes(passed=0, errors=1)
    assert values == []


def test_new_instances(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        def test_func1(self):
            self.x = 2

        def test_func2(self):
            assert not hasattr(self, "x")

    run_tests(MyTestCase, rootpath=tmp_path).assert_outcomes(passed=2)


def test_function_item_obj_is_instance(tmp_path: Path) -> None:
    """item.obj should be a bound method on unittest.TestCase function items (#5390)."""
    checked: list[bool] = []

    class CheckPlugin:
        def pytest_runtest_makereport(self, item, call):
            if call.when == "call":
                class_ = item.parent.obj
                checked.append(isinstance(item.obj.__self__, class_))

    class Test(unittest.TestCase):
        def test_foo(self):
            pass

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(CheckPlugin(),))
    run_tests(Test, spec=spec).assert_outcomes(passed=1)
    assert checked == [True]


def test_teardown(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        # deliberately shared class state, observed by Second below
        values: list[None] = []

        def test_one(self):
            pass

        def tearDown(self):
            self.values.append(None)

    class Second(unittest.TestCase):
        def test_check(self):
            self.assertEqual(MyTestCase.values, [None])

    run_tests(MyTestCase, Second, rootpath=tmp_path).assert_outcomes(passed=2)


def test_teardown_issue1649(tmp_path: Path) -> None:
    """
    Are TestCase objects cleaned up? Often unittest TestCase objects set
    attributes that are large and expensive during test run or setUp.

    The TestCase will not be cleaned up if the test fails, because it
    would then exist in the stackframe.

    Regression test for #1649 (see also #12367).
    """

    class TestCaseObjectsShouldBeCleanedUp(unittest.TestCase):
        def test_expensive(self):
            self.an_expensive_obj = object()

        def test_is_it_still_alive(self):
            gc.collect()
            for obj in gc.get_objects():
                if type(obj).__name__ == "TestCaseObjectsShouldBeCleanedUp":
                    assert not hasattr(obj, "an_expensive_obj")
                    break
            else:
                assert False, "Could not find TestCaseObjectsShouldBeCleanedUp instance"

    record = run_tests(TestCaseObjectsShouldBeCleanedUp, rootpath=tmp_path)
    record.assert_outcomes(passed=2)


def test_unittest_skip_issue148(tmp_path: Path) -> None:
    ran = []

    @unittest.skip("hello")
    class MyTestCase(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            ran.append("setUpClass")

        def test_one(self):
            ran.append("test_one")

        @classmethod
        def tearDownClass(cls):
            ran.append("tearDownClass")

    run_tests(MyTestCase, rootpath=tmp_path).assert_outcomes(skipped=1)
    # the original smuggled this in as a NameError on an undefined name
    assert ran == []


def test_unittest_skip_with_autouse_fixture(tmp_path: Path) -> None:
    """Autouse fixtures inside a @unittest.skipIf class should not run (#13885)."""

    @unittest.skipIf(True, "skip reason")
    class TestSkipped(unittest.TestCase):
        @pytest.fixture(autouse=True)
        def my_fixture(self):
            raise RuntimeError("fixture should not run")

        def test_one(self):
            pass

    run_tests(TestSkipped, rootpath=tmp_path).assert_outcomes(skipped=1)


def test_method_and_teardown_failing_reporting(tmp_path: Path) -> None:
    class TC(unittest.TestCase):
        def tearDown(self):
            assert 0, "down1"

        def test_method(self):
            assert False, "down2"

    record = run_tests(TC, rootpath=tmp_path)
    record.assert_outcomes(failed=1, errors=1)
    call = record["test_method"].call
    assert call is not None and call.failed
    assert "test_method" in call.longreprtext and "down2" in call.longreprtext
    teardown = record["test_method"].teardown
    assert teardown is not None and teardown.failed
    assert "tearDown" in teardown.longreprtext and "down1" in teardown.longreprtext


def test_setup_failure_is_shown(tmp_path: Path) -> None:
    ran = []

    class TC(unittest.TestCase):
        def setUp(self):
            assert 0, "down1"

        def test_method(self):
            ran.append("test_method")

    record = run_tests(TC, rootpath=tmp_path)
    # a failing unittest setUp is reported in the call phase, not as an error
    record.assert_outcomes(failed=1)
    call = record["test_method"].call
    assert call is not None
    assert "setUp" in call.longreprtext and "down1" in call.longreprtext
    # the test body itself must not have run (was: no_fnmatch_line("*never42*"))
    assert ran == []


def test_setup_setUpClass(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        x = 0

        @classmethod
        def setUpClass(cls):
            cls.x += 1

        def test_func1(self):
            assert self.x == 1

        def test_func2(self):
            assert self.x == 1

        @classmethod
        def tearDownClass(cls):
            cls.x -= 1

    # collection order follows the line numbers in *this* file, so
    # test_torn_down has to stay defined below the class it checks
    def test_torn_down():
        assert MyTestCase.x == 0

    record = run_tests(MyTestCase, test_torn_down, rootpath=tmp_path)
    record.assert_outcomes(passed=3)


# ensemble: --fixtures output has no in-memory equivalent
def test_fixtures_setup_setUpClass_issue8394(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import unittest
        class MyTestCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                pass
            def test_func1(self):
                pass
            @classmethod
            def tearDownClass(cls):
                pass
    """
    )
    result = pytester.runpytest("--fixtures")
    assert result.ret == 0
    result.stdout.no_fnmatch_line("*no docstring available*")

    result = pytester.runpytest("--fixtures", "-v")
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*no docstring available*"])


def test_setup_class(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        x = 0

        def setup_class(cls):
            cls.x += 1

        def test_func1(self):
            assert self.x == 1

        def test_func2(self):
            assert self.x == 1

        def teardown_class(cls):
            cls.x -= 1

    # must stay below the class: collection order follows this file's lines
    def test_torn_down():
        assert MyTestCase.x == 0

    record = run_tests(MyTestCase, test_torn_down, rootpath=tmp_path)
    record.assert_outcomes(passed=3)


@pytest.mark.parametrize("type", ["Error", "Failure"])
def test_testcase_adderrorandfailure_defers(tmp_path: Path, type: str) -> None:
    raised: list[BaseException] = []

    class MyTestCase(unittest.TestCase):
        def run(self, result=None):
            excinfo = pytest.raises(ZeroDivisionError, lambda: 0 / 0)
            try:
                getattr(result, f"add{type}")(self, excinfo._excinfo)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                # was: pytest.fail(f"add{type} should not raise")
                raised.append(e)

        def test_hello(self):
            pass

    record = run_tests(MyTestCase, rootpath=tmp_path)
    assert raised == []
    # the deferred exception info surfaces as the call phase failure
    record.assert_outcomes(failed=1)


@pytest.mark.parametrize("type", ["Error", "Failure"])
def test_testcase_custom_exception_info(tmp_path: Path, type: str) -> None:
    class MyTestCase(unittest.TestCase):
        def run(self, result=None):
            excinfo = pytest.raises(ZeroDivisionError, lambda: 0 / 0)

            # We fake an incompatible exception info.
            class FakeExceptionInfo:
                def __init__(self, *args, **kwargs):
                    mp.undo()
                    raise TypeError

                # stands in for Generic[E]: only used as ExceptionInfo[...]
                def __class_getitem__(cls, item):
                    return cls

                @classmethod
                def from_current(cls):
                    return cls()

                @classmethod
                def from_exc_info(cls, *args, **kwargs):
                    return cls()

            mp = pytest.MonkeyPatch()
            mp.setattr(_pytest._code, "ExceptionInfo", FakeExceptionInfo)
            try:
                getattr(result, f"add{type}")(self, excinfo._excinfo)
            finally:
                mp.undo()

        def test_hello(self):
            pass

    record = run_tests(MyTestCase, rootpath=tmp_path)
    record.assert_outcomes(failed=1)
    call = record["test_hello"].call
    assert call is not None
    assert "NOTE: Incompatible Exception Representation" in call.longreprtext
    assert "ZeroDivisionError" in call.longreprtext


def test_testcase_totally_incompatible_exception_info(tmp_path: Path) -> None:
    import _pytest.unittest

    class MyTestCase(unittest.TestCase):
        def test_hello(self):
            pass

    (item,) = collect_tests(MyTestCase, rootpath=tmp_path)
    assert isinstance(item, _pytest.unittest.TestCaseFunction)
    item.addError(None, 42)  # type: ignore[arg-type]
    excinfo = item._excinfo
    assert excinfo is not None
    assert "ERROR: Unknown Incompatible" in str(excinfo.pop(0).getrepr())


def test_module_level_pytestmark(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        def test_func1(self):
            assert 0

    module = build_module(
        "test_module_level_pytestmark", MyTestCase, pytestmark=pytest.mark.xfail
    )
    record = run_tests(module, rootpath=tmp_path)
    # assertoutcome() counted the xfail report as skipped; the terminal
    # category it lands in is xfailed.
    record.assert_outcomes(xfailed=1)


def set_attributes(**attrs: object):
    """Set attributes on a test method (trial's ``skip``/``todo``, ``__test__``).

    In the file based originals these were plain assignments in the class
    body, which type checkers reject on real (non-``exec``'d) code.
    """

    def decorate(func):
        for name, value in attrs.items():
            setattr(func, name, value)
        return func

    return decorate


class TestTrialUnittest:
    def setup_class(cls):
        cls.ut = pytest.importorskip("twisted.trial.unittest")
        # on windows trial uses a socket for a reactor and apparently doesn't close it properly
        # https://twistedmatrix.com/trac/ticket/9227
        # (was "-W always" on the command line; an ensemble inherits the host
        # suite's filterwarnings=error unless its own inicfg says otherwise)
        cls.ignore_unclosed_socket_inicfg = {"filterwarnings": ["always"]}

    def spec(self, tmp_path: Path) -> ConfigSpec:
        return ConfigSpec(rootpath=tmp_path, inicfg=self.ignore_unclosed_socket_inicfg)

    def test_trial_testcase_runtest_not_collected(self, tmp_path: Path) -> None:
        from twisted.trial.unittest import TestCase as TrialTestCase

        spec = self.spec(tmp_path)

        class TC(TrialTestCase):
            def test_hello(self):
                pass

        # trial's own inherited runTest is not collected next to test_hello
        run_tests(TC, spec=spec).assert_outcomes(passed=1)

        class TCWithRunTest(TrialTestCase):
            def runTest(self):
                pass

        run_tests(TCWithRunTest, spec=spec).assert_outcomes(passed=1)

    def test_trial_exceptions_with_skips(self, tmp_path: Path) -> None:
        from twisted.trial.unittest import TestCase as TrialTestCase

        class TC(TrialTestCase):
            def test_hello(self):
                pytest.skip("skip_in_method")

            @pytest.mark.skipif("sys.version_info != 1")
            def test_hello2(self):
                pass

            @pytest.mark.xfail(reason="iwanto")
            def test_hello3(self):
                assert 0

            def test_hello4(self):
                pytest.xfail("i2wanto")

            @set_attributes(skip="trialselfskip")
            def test_trial_skip(self):
                pass

            @set_attributes(todo="mytodo")
            def test_trial_todo(self):
                assert 0

            @set_attributes(todo="mytodo")
            def test_trial_todo_success(self):
                pass

        class TC2(TrialTestCase):
            def setup_class(cls):
                pytest.skip("skip_in_setup_class")

            def test_method(self):
                pass

        record = run_tests(TC, TC2, spec=self.spec(tmp_path))
        record.assert_outcomes(failed=1, skipped=4, xfailed=3)

        def reason(name: str) -> str:
            item = record[name]
            report = item.call if item.call is not None else item.setup
            assert report is not None
            return getattr(report, "wasxfail", "") + report.longreprtext

        assert "skip_in_method" in reason("test_hello")
        assert "sys.version_info" in reason("test_hello2")
        assert "iwanto" in reason("test_hello3")
        assert "i2wanto" in reason("test_hello4")
        assert "trialselfskip" in reason("test_trial_skip")
        assert "mytodo" in reason("test_trial_todo")
        assert record["test_trial_todo"].outcome == "xfailed"
        assert record["test_trial_todo_success"].failed
        assert "skip_in_setup_class" in reason("test_method")

    def test_trial_error(self, tmp_path: Path) -> None:
        from twisted.internet import reactor
        from twisted.internet.defer import Deferred
        from twisted.trial.unittest import TestCase as TrialTestCase

        class TC(TrialTestCase):
            def test_one(self):
                raise NameError("crash")

            def test_two(self):
                def f(_):
                    raise NameError("crash")

                d = Deferred()
                d.addCallback(f)
                reactor.callLater(0.3, d.callback, None)
                return d

            def test_three(self):
                def f():
                    pass  # will never get called

                reactor.callLater(0.3, f)

            # will crash at teardown

            def test_four(self):
                def f(_):
                    reactor.callLater(0.3, f)
                    raise NameError("crash")

                d = Deferred()
                d.addCallback(f)
                reactor.callLater(0.3, d.callback, None)
                return d

            # will crash both at test time and at teardown

        spec = ConfigSpec(
            rootpath=tmp_path,
            args=("-vv", "-oconsole_output_style=classic"),
            inicfg={"filterwarnings": ["ignore::DeprecationWarning"]},
        )
        # this one is about what gets *rendered*, so keep the glob matching
        record = run_tests(TC, spec=spec, name="test_trial_error", capture_output=True)
        record.stdout.fnmatch_lines(
            [
                # the ``*`` swallows the "<- <real source file>" annotation -v
                # adds because the in-memory module's path is synthetic
                "test_trial_error.py::TC::test_four *FAILED",
                "test_trial_error.py::TC::test_four *ERROR",
                "test_trial_error.py::TC::test_one *FAILED",
                "test_trial_error.py::TC::test_three *FAILED",
                "test_trial_error.py::TC::test_two *FAILED",
                "*ERRORS*",
                "*_ ERROR at teardown of TC.test_four _*",
                "*DelayedCalls*",
                "*= FAILURES =*",
                "*_ TC.test_four _*",
                "*NameError*crash*",
                "*_ TC.test_one _*",
                "*NameError*crash*",
                "*_ TC.test_three _*",
                "*DelayedCalls*",
                "*_ TC.test_two _*",
                "*NameError*crash*",
                "*= 4 failed, 1 error in *",
            ]
        )

    # ensemble: needs a terminal to type into (pexpect)
    def test_trial_pdb(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            """
            from twisted.trial import unittest
            import pytest
            class TC(unittest.TestCase):
                def test_hello(self):
                    assert 0, "hellopdb"
        """
        )
        child = pytester.spawn_pytest(str(p))
        child.expect("hellopdb")
        child.sendeof()

    def test_trial_testcase_skip_property(self, tmp_path: Path) -> None:
        from twisted.trial.unittest import TestCase as TrialTestCase

        class MyTestCase(TrialTestCase):
            skip = "dont run"

            def test_func(self):
                pass

        run_tests(MyTestCase, spec=self.spec(tmp_path)).assert_outcomes(skipped=1)

    def test_trial_testfunction_skip_property(self, tmp_path: Path) -> None:
        from twisted.trial.unittest import TestCase as TrialTestCase

        class MyTestCase(TrialTestCase):
            @set_attributes(skip="dont run")
            def test_func(self):
                pass

        run_tests(MyTestCase, spec=self.spec(tmp_path)).assert_outcomes(skipped=1)

    def test_trial_testcase_todo_property(self, tmp_path: Path) -> None:
        from twisted.trial.unittest import TestCase as TrialTestCase

        class MyTestCase(TrialTestCase):
            todo = "dont run"

            def test_func(self):
                assert 0

        # assertoutcome() counted the xfail report as skipped
        run_tests(MyTestCase, spec=self.spec(tmp_path)).assert_outcomes(xfailed=1)

    def test_trial_testfunction_todo_property(self, tmp_path: Path) -> None:
        from twisted.trial.unittest import TestCase as TrialTestCase

        class MyTestCase(TrialTestCase):
            @set_attributes(todo="dont run")
            def test_func(self):
                assert 0

        run_tests(MyTestCase, spec=self.spec(tmp_path)).assert_outcomes(xfailed=1)


def test_djangolike_testcase(tmp_path: Path) -> None:
    # contributed from Morten Breekevold
    events: list[str] = []

    class DjangoLikeTestCase(unittest.TestCase):
        def setUp(self):
            events.append("setUp()")

        def test_presetup_has_been_run(self):
            events.append("test_thing()")
            self.assertTrue(hasattr(self, "was_presetup"))

        def tearDown(self):
            events.append("tearDown()")

        def __call__(self, result=None):
            try:
                self._pre_setup()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                result.addError(self, sys.exc_info())
                return
            super().__call__(result)
            try:
                self._post_teardown()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                result.addError(self, sys.exc_info())
                return

        def _pre_setup(self):
            events.append("_pre_setup()")
            self.was_presetup = True

        def _post_teardown(self):
            events.append("_post_teardown()")

    record = run_tests(DjangoLikeTestCase, rootpath=tmp_path)
    record.assert_outcomes(passed=1)
    # asserting the order directly, instead of globbing printed lines
    assert events == [
        "_pre_setup()",
        "setUp()",
        "test_thing()",
        "tearDown()",
        "_post_teardown()",
    ]


def test_unittest_not_shown_in_traceback(tmp_path: Path) -> None:
    class t(unittest.TestCase):
        def test_hello(self):
            x = 3
            self.assertEqual(x, 4)

    record = run_tests(t, rootpath=tmp_path)
    record.assert_outcomes(failed=1)
    call = record["test_hello"].call
    assert call is not None
    # the failing line is shown, unittest's own frames leading to it are not
    assert "self.assertEqual(x, 4)" in call.longreprtext
    assert "failUnlessEqual" not in call.longreprtext


def test_unorderable_types(tmp_path: Path) -> None:
    class TestJoinEmpty(unittest.TestCase):
        pass

    def make_test():
        class Test(unittest.TestCase):
            pass

        Test.__name__ = "TestFoo"
        return Test

    module = build_module("test_unorderable_types", TestJoinEmpty, make_test())
    # collect_tests() raises on a collection error, so an empty list really
    # means "collected nothing", not "blew up on unorderable types"
    assert collect_tests(module, rootpath=tmp_path) == []


def test_unittest_typerror_traceback(tmp_path: Path) -> None:
    class TestJoinEmpty(unittest.TestCase):
        def test_hello(self, arg1):
            pass

    record = run_tests(TestJoinEmpty, rootpath=tmp_path)
    record.assert_outcomes(failed=1)
    call = record["test_hello"].call
    assert call is not None
    assert "TypeError" in call.longreprtext


# ensemble: the "unittest" variant runs the module as a script (runpython)
@pytest.mark.parametrize("runner", ["pytest", "unittest"])
def test_unittest_expected_failure_for_failing_test_is_xfail(
    pytester: Pytester, runner
) -> None:
    script = pytester.makepyfile(
        """
        import unittest
        class MyTestCase(unittest.TestCase):
            @unittest.expectedFailure
            def test_failing_test_is_xfail(self):
                assert False
        if __name__ == '__main__':
            unittest.main()
    """
    )
    if runner == "pytest":
        result = pytester.runpytest("-rxX")
        result.stdout.fnmatch_lines(
            ["*XFAIL*MyTestCase*test_failing_test_is_xfail*", "*1 xfailed*"]
        )
    else:
        result = pytester.runpython(script)
        result.stderr.fnmatch_lines(["*1 test in*", "*OK*(expected failures=1)*"])
    assert result.ret == 0


# ensemble: the "unittest" variant runs the module as a script (runpython)
@pytest.mark.parametrize("runner", ["pytest", "unittest"])
def test_unittest_expected_failure_for_passing_test_is_fail(
    pytester: Pytester,
    runner: str,
) -> None:
    script = pytester.makepyfile(
        """
        import unittest
        class MyTestCase(unittest.TestCase):
            @unittest.expectedFailure
            def test_passing_test_is_fail(self):
                assert True
        if __name__ == '__main__':
            unittest.main()
    """
    )

    if runner == "pytest":
        result = pytester.runpytest("-rxX")
        result.stdout.fnmatch_lines(
            [
                "*MyTestCase*test_passing_test_is_fail*",
                "Unexpected success",
                "*1 failed*",
            ]
        )
    else:
        result = pytester.runpython(script)
        result.stderr.fnmatch_lines(["*1 test in*", "*(unexpected successes=1)*"])

    assert result.ret == 1


@pytest.mark.parametrize("stmt", ["return", "yield"])
def test_unittest_setup_interaction(tmp_path: Path, stmt: str) -> None:
    # the string template parametrized the *shape* of the fixture bodies;
    # in-memory sources need both variants spelled out
    if stmt == "return":

        def perclass(cls, request):
            request.cls.hello = "world"
            return  # noqa: PLR1711

        def perfunction(self, request):
            request.instance.funcname = request.function.__name__
            return  # noqa: PLR1711

    else:

        def perclass(cls, request):
            request.cls.hello = "world"
            yield

        def perfunction(self, request):
            request.instance.funcname = request.function.__name__
            yield

    class MyTestCase(unittest.TestCase):
        # set by the fixtures below
        hello: str
        funcname: str

        def test_method1(self):
            assert self.funcname == "test_method1"
            assert self.hello == "world"

        def test_method2(self):
            assert self.funcname == "test_method2"

        def test_classattr(self):
            assert self.__class__.hello == "world"

    # the fixtures are attached after the fact: a class body cannot read the
    # enclosing function's locals under the same name it binds
    MyTestCase.perclass = pytest.fixture(scope="class", autouse=True)(  # type: ignore[attr-defined]
        classmethod(perclass)  # type: ignore[arg-type]
    )
    MyTestCase.perfunction = pytest.fixture(scope="function", autouse=True)(perfunction)  # type: ignore[attr-defined]

    run_tests(MyTestCase, rootpath=tmp_path).assert_outcomes(passed=3)


def test_non_unittest_no_setupclass_support(tmp_path: Path) -> None:
    class TestFoo:
        x = 0

        @classmethod
        def setUpClass(cls):
            cls.x = 1

        def test_method1(self):
            assert self.x == 0

        @classmethod
        def tearDownClass(cls):
            cls.x = 1

    # must stay below the class: collection order follows this file's lines
    def test_not_torn_down():
        assert TestFoo.x == 0

    record = run_tests(TestFoo, test_not_torn_down, rootpath=tmp_path)
    record.assert_outcomes(passed=2)


def test_no_teardown_if_setupclass_failed(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        x = 0

        @classmethod
        def setUpClass(cls):
            cls.x = 1
            assert False

        def test_func1(self):
            MyTestCase.x = 10

        @classmethod
        def tearDownClass(cls):
            cls.x = 100

    # must stay below the class: collection order follows this file's lines
    def test_notTornDown():
        assert MyTestCase.x == 1

    record = run_tests(MyTestCase, test_notTornDown, rootpath=tmp_path)
    # setUpClass runs as a class scoped fixture, so its failure is a setup
    # phase *error* where assertoutcome() only saw a failed report
    record.assert_outcomes(passed=1, errors=1)


def test_cleanup_functions(tmp_path: Path) -> None:
    """Ensure functions added with addCleanup are always called after each test ends (#6947)"""
    cleanups: list[str] = []

    class Test(unittest.TestCase):
        def test_func_1(self):
            self.addCleanup(cleanups.append, "test_func_1")

        def test_func_2(self):
            self.addCleanup(cleanups.append, "test_func_2")
            assert 0

        def test_func_3_check_cleanups(self):
            assert cleanups == ["test_func_1", "test_func_2"]

    record = run_tests(Test, rootpath=tmp_path)
    assert record["test_func_1"].passed
    assert record["test_func_2"].failed
    assert record["test_func_3_check_cleanups"].passed
    assert cleanups == ["test_func_1", "test_func_2"]


def test_issue333_result_clearing(tmp_path: Path) -> None:
    class FailAfterCallPlugin:
        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_call(self, item):
            yield
            assert 0

    class TestIt(unittest.TestCase):
        def test_func(self):
            0 / 0  # noqa: B018

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(FailAfterCallPlugin(),))
    run_tests(TestIt, spec=spec).assert_outcomes(failed=1)


# ensemble canary: kept file based so that unittest collection through a real
# module import, and the path based nodeid in the report, stay covered here
def test_unittest_raise_skip_issue748(pytester: Pytester) -> None:
    pytester.makepyfile(
        test_foo="""
        import unittest

        class MyTestCase(unittest.TestCase):
            def test_one(self):
                raise unittest.SkipTest('skipping due to reasons')
    """
    )
    result = pytester.runpytest("-v", "-rs")
    result.stdout.fnmatch_lines(
        """
        *SKIP*[1]*test_foo.py*skipping due to reasons*
        *1 skipped*
    """
    )


def test_unittest_skip_issue1169(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        @unittest.skip("skipping due to reasons")
        def test_skip(self):
            self.fail()

    record = run_tests(MyTestCase, rootpath=tmp_path)
    record.assert_outcomes(skipped=1)
    # a method level @unittest.skip is reported by unittest itself, i.e. in
    # the call phase (a class level one skips in setup instead)
    call = record["test_skip"].call
    assert call is not None
    assert "skipping due to reasons" in call.longreprtext


def test_class_method_containing_test_issue1558(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        def test_should_run(self):
            pass

        @set_attributes(__test__=False)
        def test_should_not_run(self):
            pass

    items = collect_tests(MyTestCase, rootpath=tmp_path)
    assert [item.name for item in items] == ["test_should_run"]
    run_tests(MyTestCase, rootpath=tmp_path).assert_outcomes(passed=1)


@pytest.mark.parametrize(
    "base", [object, unittest.TestCase], ids=["builtins.object", "unittest.TestCase"]
)
def test_usefixtures_marker_on_unittest(base, tmp_path: Path) -> None:
    """#3498"""
    seen: list[tuple[str, list[str]]] = []

    def node_and_marks(item):
        seen.append((item.name, [mark.name for mark in item.iter_markers()]))

    class ConftestPlugin:
        @pytest.fixture(scope="function")
        def fixture1(self, request, monkeypatch):
            monkeypatch.setattr(request.instance, "fixture1", True)

        @pytest.fixture(scope="function")
        def fixture2(self, request, monkeypatch):
            monkeypatch.setattr(request.instance, "fixture2", True)

        @pytest.fixture(autouse=True)
        def my_marks(self, request):
            node_and_marks(request.node)

        def pytest_collection_modifyitems(self, items):
            for item in items:
                node_and_marks(item)

    class Tests(base):
        fixture1 = False
        fixture2 = False

        @pytest.mark.usefixtures("fixture1")
        def test_one(self):
            assert self.fixture1
            assert not self.fixture2

        @pytest.mark.usefixtures("fixture1", "fixture2")
        def test_two(self):
            assert self.fixture1
            assert self.fixture2

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
    run_tests(Tests, spec=spec).assert_outcomes(passed=2)
    # the usefixtures marks are visible at collection *and* at fixture time
    # (this is what the conftest printed for eyeballing)
    assert seen.count(("test_one", ["usefixtures"])) == 2
    assert seen.count(("test_two", ["usefixtures"])) == 2


def test_skip_setup_class(tmp_path: Path) -> None:
    """Skipping tests in a class by raising unittest.SkipTest in `setUpClass` (#13985)."""

    class Test(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest("Skipping setupclass")

        def test_foo(self):
            assert False

        def test_bar(self):
            assert False

    run_tests(Test, rootpath=tmp_path).assert_outcomes(skipped=2)


def test_unittest_skip_function(tmp_path: Path) -> None:
    """
    Ensure raising an explicit unittest.SkipTest skips standard pytest functions.

    Support for this is debatable -- technically we only support unittest.SkipTest in TestCase subclasses,
    but stating this support here in this test because users currently expect this to work,
    so if we ever break it we at least know we are breaking this use case (#13985).
    """

    def test_foo():
        raise unittest.SkipTest("Skipping test_foo")

    run_tests(test_foo, rootpath=tmp_path).assert_outcomes(skipped=1)


def test_testcase_handles_init_exceptions(tmp_path: Path) -> None:
    """
    Regression test to make sure exceptions in the __init__ method are bubbled up correctly.
    See https://github.com/pytest-dev/pytest/issues/3788
    """

    class MyTestCase(unittest.TestCase):
        def __init__(self, *args, **kwargs):
            raise Exception("should raise this exception")

        def test_hello(self):
            pass

    record = run_tests(MyTestCase, rootpath=tmp_path)
    record.assert_outcomes(errors=1)
    (error,) = record.collect_errors
    assert "should raise this exception" in str(error.longrepr)
    # nothing was collected, so nothing ran and in particular there is no
    # teardown error (was: no_fnmatch_line("*ERROR at teardown of*"))
    assert record.reports == []


def test_error_message_with_parametrized_fixtures() -> None:
    example = EXAMPLES / "unittest/test_parametrized_fixture_error_message.py"
    module = module_from_path(example)
    record = run_tests(module, rootpath=example.parent, capture_output=True)
    record.stdout.fnmatch_lines(
        [
            "*test_two does not support fixtures*",
            "*TestSomethingElse::test_two",
            "*Function type: TestCaseFunction",
        ]
    )


@pytest.mark.parametrize(
    "test_name, expected_outcome, outcomes",
    [
        ("test_setup_skip.py", "1 skipped", {"skipped": 1}),
        ("test_setup_skip_class.py", "1 skipped", {"skipped": 1}),
        ("test_setup_skip_module.py", "1 error", {"errors": 1}),
    ],
)
def test_setup_inheritance_skipping(test_name, expected_outcome, outcomes) -> None:
    """Issue #4700"""
    example = EXAMPLES / "unittest" / test_name
    module = module_from_path(example)
    record = run_tests(module, rootpath=example.parent, capture_output=True)
    record.stdout.fnmatch_lines([f"* {expected_outcome} in *"])
    record.assert_outcomes(**outcomes)


def test_BdbQuit(tmp_path: Path) -> None:
    class MyTestCase(unittest.TestCase):
        def test_bdbquit(self):
            import bdb

            raise bdb.BdbQuit

        def test_should_not_run(self):
            pass

    run_tests(MyTestCase, rootpath=tmp_path).assert_outcomes(failed=1, passed=1)


def test_exit_outcome(tmp_path: Path) -> None:
    ran: list[str] = []

    class MyTestCase(unittest.TestCase):
        def test_exit_outcome(self):
            pytest.exit("pytest_exit called")

        def test_should_not_run(self):
            ran.append("test_should_not_run")

    # an ensemble has no session wrapper turning Exit into a summary line, so
    # the exit surfaces as the exception it is
    with pytest.raises(Exit, match="pytest_exit called"):
        run_tests(MyTestCase, rootpath=tmp_path)
    assert ran == []


def test_trace(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls = []

    def check_call(*args, **kwargs):
        calls.append((args, kwargs))
        assert args == ("runcall",)

        class _pdb:
            def runcall(*args, **kwargs):
                calls.append((args, kwargs))

        return _pdb

    monkeypatch.setattr("_pytest.debugging.pytestPDB._init_pdb", check_call)

    class MyTestCase(unittest.TestCase):
        def test(self):
            self.assertEqual("foo", "foo")

    spec = ConfigSpec(rootpath=tmp_path, args=("--trace",)).with_plugins("debugging")
    run_tests(MyTestCase, spec=spec).assert_outcomes(passed=1)
    assert len(calls) == 2


def test_pdb_teardown_called(tmp_path: Path) -> None:
    """Ensure tearDown() is always called when --pdb is given in the command-line.

    We delay the normal tearDown() calls when --pdb is given, so this ensures we are calling
    tearDown() eventually to avoid memory leaks when using --pdb.
    """
    teardowns: list[str] = []

    class MyTestCase(unittest.TestCase):
        def tearDown(self):
            teardowns.append(self.id())

        def test_1(self):
            pass

        def test_2(self):
            pass

    spec = ConfigSpec(rootpath=tmp_path, args=("--pdb",)).with_plugins("debugging")
    run_tests(MyTestCase, spec=spec).assert_outcomes(passed=2)
    # TestCase.id() is built from __qualname__, which for a class defined in
    # a test carries a "<locals>" segment
    assert [teardown.split("<locals>.")[-1] for teardown in teardowns] == [
        "MyTestCase.test_1",
        "MyTestCase.test_2",
    ]


@pytest.mark.parametrize(
    "mark",
    [
        pytest.param(unittest.skip("skipped for reasons"), id="unittest.skip"),
        pytest.param(pytest.mark.skip("skipped for reasons"), id="pytest.mark.skip"),
    ],
)
def test_pdb_teardown_skipped_for_functions(tmp_path: Path, mark) -> None:
    """
    With --pdb, setUp and tearDown should not be called for tests skipped
    via a decorator (#7215).
    """
    tracked: list[str] = []

    class MyTestCase(unittest.TestCase):
        def setUp(self):
            tracked.append("setUp:" + self.id())

        def tearDown(self):
            tracked.append("tearDown:" + self.id())

        @mark
        def test_1(self):
            pass

    spec = ConfigSpec(rootpath=tmp_path, args=("--pdb",)).with_plugins("debugging")
    run_tests(MyTestCase, spec=spec).assert_outcomes(skipped=1)
    assert tracked == []


@pytest.mark.parametrize(
    "mark",
    [
        pytest.param(unittest.skip("skipped for reasons"), id="unittest.skip"),
        pytest.param(pytest.mark.skip("skipped for reasons"), id="pytest.mark.skip"),
    ],
)
def test_pdb_teardown_skipped_for_classes(tmp_path: Path, mark) -> None:
    """
    With --pdb, setUp and tearDown should not be called for tests skipped
    via a decorator on the class (#10060).
    """
    tracked: list[str] = []

    @mark
    class MyTestCase(unittest.TestCase):
        def setUp(self):
            tracked.append("setUp:" + self.id())

        def tearDown(self):
            tracked.append("tearDown:" + self.id())

        def test_1(self):
            pass

    spec = ConfigSpec(rootpath=tmp_path, args=("--pdb",)).with_plugins("debugging")
    run_tests(MyTestCase, spec=spec).assert_outcomes(skipped=1)
    assert tracked == []


def test_async_support() -> None:
    pytest.importorskip("unittest.async_case")

    example = EXAMPLES / "unittest/test_unittest_asyncio.py"
    module = module_from_path(example)
    run_tests(module, rootpath=example.parent).assert_outcomes(failed=1, passed=2)


@pytest.mark.skipif(
    sys.version_info >= (3, 11), reason="asynctest is not compatible with Python 3.11+"
)
def test_asynctest_support() -> None:
    """Check asynctest support (#7110)"""
    pytest.importorskip("asynctest")
    example = EXAMPLES / "unittest/test_unittest_asynctest.py"
    module = module_from_path(example)
    run_tests(module, rootpath=example.parent).assert_outcomes(failed=1, passed=2)


# ensemble: needs a subprocess (the unawaited coroutine warning depends on gc),
# so the example script has to be copied somewhere the subprocess can run it.
def test_plain_unittest_does_not_support_async(pytester: Pytester) -> None:
    """Async functions in plain unittest.TestCase subclasses are not supported without plugins.

    This test exists here to avoid introducing this support by accident, leading users
    to expect that it works, rather than doing so intentionally as a feature.

    See https://github.com/pytest-dev/pytest-asyncio/issues/180 for more context.
    """
    pytester.copy_example("unittest/test_unittest_plain_async.py")
    result = pytester.runpytest_subprocess()
    if hasattr(sys, "pypy_version_info"):
        # in PyPy we can't reliable get the warning about the coroutine not being awaited,
        # because it depends on the coroutine being garbage collected; given that
        # we are running in a subprocess, that's difficult to enforce
        expected_lines = ["*1 passed*"]
    else:
        expected_lines = [
            "*RuntimeWarning: coroutine * was never awaited",
            "*1 passed*",
        ]
    result.stdout.fnmatch_lines(expected_lines)


def test_do_class_cleanups_on_success(tmp_path: Path) -> None:
    values: list[int] = []

    class MyTestCase(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            def cleanup():
                values.append(1)

            cls.addClassCleanup(cleanup)

        def test_one(self):
            pass

        def test_two(self):
            pass

    record = run_tests(MyTestCase, rootpath=tmp_path)
    record.assert_outcomes(passed=2)
    # was a trailing test function asserting this from the outside
    assert values == [1]


def test_do_class_cleanups_on_setupclass_failure(tmp_path: Path) -> None:
    values: list[int] = []

    class MyTestCase(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            def cleanup():
                values.append(1)

            cls.addClassCleanup(cleanup)
            assert False

        def test_one(self):
            pass

    record = run_tests(MyTestCase, rootpath=tmp_path)
    # setUpClass runs as a class scoped fixture: a setup phase error
    record.assert_outcomes(errors=1)
    assert values == [1]


def test_do_class_cleanups_on_teardownclass_failure(tmp_path: Path) -> None:
    values: list[int] = []

    class MyTestCase(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            def cleanup():
                values.append(1)

            cls.addClassCleanup(cleanup)

        @classmethod
        def tearDownClass(cls):
            assert False

        def test_one(self):
            pass

        def test_two(self):
            pass

    record = run_tests(MyTestCase, rootpath=tmp_path)
    # countoutcomes() ignored the teardown error the original also produced
    record.assert_outcomes(passed=2, errors=1)
    assert values == [1]


def test_do_cleanups_on_success(tmp_path: Path) -> None:
    values: list[int] = []

    class MyTestCase(unittest.TestCase):
        def setUp(self):
            def cleanup():
                values.append(1)

            self.addCleanup(cleanup)

        def test_one(self):
            pass

        def test_two(self):
            pass

    record = run_tests(MyTestCase, rootpath=tmp_path)
    record.assert_outcomes(passed=2)
    assert values == [1, 1]


def test_do_cleanups_on_setup_failure(tmp_path: Path) -> None:
    values: list[int] = []

    class MyTestCase(unittest.TestCase):
        def setUp(self):
            def cleanup():
                values.append(1)

            self.addCleanup(cleanup)
            assert False

        def test_one(self):
            pass

        def test_two(self):
            pass

    record = run_tests(MyTestCase, rootpath=tmp_path)
    # a unittest setUp failure is reported in the call phase, so these stay
    # failures rather than becoming errors
    record.assert_outcomes(failed=2)
    assert values == [1, 1]


def test_do_cleanups_on_teardown_failure(tmp_path: Path) -> None:
    values: list[int] = []

    class MyTestCase(unittest.TestCase):
        def setUp(self):
            def cleanup():
                values.append(1)

            self.addCleanup(cleanup)

        def tearDown(self):
            assert False

        def test_one(self):
            pass

        def test_two(self):
            pass

    record = run_tests(MyTestCase, rootpath=tmp_path)
    record.assert_outcomes(failed=2)
    assert values == [1, 1]


class TestClassCleanupErrors:
    """
    Make sure to show exceptions raised during class cleanup function (those registered
    via addClassCleanup()).

    See #11728.
    """

    def test_class_cleanups_failure_in_setup(self, tmp_path: Path) -> None:
        class MyTestCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                def cleanup(n):
                    raise Exception(f"fail {n}")

                cls.addClassCleanup(cleanup, 2)
                cls.addClassCleanup(cleanup, 1)
                raise Exception("fail 0")

            def test(self):
                pass

        record = run_tests(MyTestCase, rootpath=tmp_path)
        record.assert_outcomes(passed=0, errors=1)
        setup = record["test"].setup
        assert setup is not None and setup.failed
        text = setup.longreprtext
        assert "Unittest class cleanup errors" in text
        assert "2 sub-exceptions" in text
        assert "Exception: fail 1" in text
        assert "Exception: fail 2" in text
        assert "Exception: fail 0" in text

    def test_class_cleanups_failure_in_teardown(self, tmp_path: Path) -> None:
        class MyTestCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                def cleanup(n):
                    raise Exception(f"fail {n}")

                cls.addClassCleanup(cleanup, 2)
                cls.addClassCleanup(cleanup, 1)

            def test(self):
                pass

        record = run_tests(MyTestCase, rootpath=tmp_path)
        record.assert_outcomes(passed=1, errors=1)
        teardown = record["test"].teardown
        assert teardown is not None and teardown.failed
        text = teardown.longreprtext
        assert "Unittest class cleanup errors" in text
        assert "2 sub-exceptions" in text
        assert "Exception: fail 1" in text
        assert "Exception: fail 2" in text

    def test_class_cleanup_1_failure_in_teardown(self, tmp_path: Path) -> None:
        class MyTestCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                def cleanup(n):
                    raise Exception(f"fail {n}")

                cls.addClassCleanup(cleanup, 1)

            def test(self):
                pass

        record = run_tests(MyTestCase, rootpath=tmp_path)
        record.assert_outcomes(passed=1, errors=1)
        # was: "*ERROR at teardown of MyTestCase.test*"
        teardown = record["test"].teardown
        assert teardown is not None and teardown.failed
        assert "Exception: fail 1" in teardown.longreprtext


def test_traceback_pruning(tmp_path: Path) -> None:
    """Regression test for #9610 - doesn't crash during traceback pruning."""

    class MyTestCase(unittest.TestCase):
        def __init__(self, test_method):
            unittest.TestCase.__init__(self, test_method)

    class TestIt(MyTestCase):
        @classmethod
        def tearDownClass(cls) -> None:
            assert False

        def test_it(self):
            pass

    record = run_tests(TestIt, rootpath=tmp_path)
    # tearDownClass runs as a class scoped fixture: a teardown phase error
    record.assert_outcomes(passed=1, errors=1)
    teardown = record["test_it"].teardown
    assert teardown is not None and teardown.failed


# ensemble canary: a module level ``raise unittest.SkipTest`` needs a real
# module import, which in-memory sources by definition do not do
def test_raising_unittest_skiptest_during_collection(
    pytester: Pytester,
) -> None:
    pytester.makepyfile(
        """
        import unittest

        class TestIt(unittest.TestCase):
            def test_it(self): pass
            def test_it2(self): pass

        raise unittest.SkipTest()

        class TestIt2(unittest.TestCase):
            def test_it(self): pass
            def test_it2(self): pass
        """
    )
    reprec = pytester.inline_run()
    passed, skipped, failed = reprec.countoutcomes()
    assert passed == 0
    # Unittest reports one fake test for a skipped module.
    assert skipped == 1
    assert failed == 0
    assert reprec.ret == ExitCode.NO_TESTS_COLLECTED


def test_abstract_testcase_is_not_collected(tmp_path: Path) -> None:
    """Regression test for #12275."""

    class TestBase(unittest.TestCase, abc.ABC):
        @abc.abstractmethod
        def abstract1(self):
            pass

        @abc.abstractmethod
        def abstract2(self):
            pass

        def test_it(self):
            pass

    class TestPartial(TestBase):
        def abstract1(self):
            pass

    class TestConcrete(TestPartial):
        def abstract2(self):
            pass

    items = collect_tests(TestBase, TestPartial, TestConcrete, rootpath=tmp_path)
    assert [item.nodeid.split("::", 1)[1] for item in items] == [
        "TestConcrete::test_it"
    ]
    record = run_tests(TestBase, TestPartial, TestConcrete, rootpath=tmp_path)
    record.assert_outcomes(passed=1)

# mypy: allow-untyped-defs
from __future__ import annotations

from collections.abc import Callable
from functools import partial
import inspect
import os
from pathlib import Path
import sys
import types
from typing import cast

from _pytest import outcomes
from _pytest import reports
from _pytest import runner
from _pytest._code import ExceptionInfo
from _pytest._code.code import ExceptionChainRepr
from _pytest.config import ExitCode
from _pytest.ensemble import build_module
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import Ensemble
from _pytest.ensemble import EnsembleModule
from _pytest.ensemble import run_tests
from _pytest.ensemble import Source
from _pytest.monkeypatch import MonkeyPatch
from _pytest.outcomes import OutcomeException
from _pytest.pytester import Pytester
import pytest


if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


#: A runner for the runtest protocol of a single item, as the test classes
#: below hand out from ``getrunner``.
ProtocolRunner = Callable[[pytest.Item], list[reports.TestReport]]


def runitem(
    getrunner: Callable[[], ProtocolRunner],
    tmp_path: Path,
    *sources: Source,
) -> list[reports.TestReport]:
    """Collect exactly one item from in-memory sources and run it through the
    protocol runner the calling test class provides.

    The ensemble replacement for :meth:`Pytester.runitem`, which does the same
    thing with a file on disk and looks the runner up on the calling instance.
    """
    with Ensemble(*sources, rootpath=tmp_path) as ensemble:
        (item,) = ensemble.collect()
        return getrunner()(item)


class TestSetupState:
    def test_setup(self, tmp_path: Path) -> None:
        def test_func(): ...

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            ss = item.session._setupstate
            values = [1]
            ss.setup(item)
            ss.addfinalizer(values.pop, item)
            assert values
            ss.teardown_exact(None)
            assert not values

    def test_teardown_exact_stack_empty(self, tmp_path: Path) -> None:
        def test_func(): ...

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            ss = item.session._setupstate
            ss.setup(item)
            ss.teardown_exact(None)
            ss.teardown_exact(None)
            ss.teardown_exact(None)

    def test_setup_fails_and_failure_is_cached(self, tmp_path: Path) -> None:
        def setup_module(mod):
            raise ValueError(42)

        def test_func(): ...

        with Ensemble(setup_module, test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            ss = item.session._setupstate
            with pytest.raises(ValueError):
                ss.setup(item)
            with pytest.raises(ValueError):
                ss.setup(item)

    def test_teardown_multiple_one_fails(self, tmp_path: Path) -> None:
        r = []

        def fin1():
            r.append("fin1")

        def fin2():
            raise Exception("oops")

        def fin3():
            r.append("fin3")

        def test_func(): ...

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            ss = item.session._setupstate
            ss.setup(item)
            ss.addfinalizer(fin1, item)
            ss.addfinalizer(fin2, item)
            ss.addfinalizer(fin3, item)
            with pytest.raises(Exception) as err:
                ss.teardown_exact(None)
            assert err.value.args == ("oops",)
            assert r == ["fin3", "fin1"]

    def test_teardown_multiple_fail(self, tmp_path: Path) -> None:
        def fin1():
            raise Exception("oops1")

        def fin2():
            raise Exception("oops2")

        def test_func(): ...

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            ss = item.session._setupstate
            ss.setup(item)
            ss.addfinalizer(fin1, item)
            ss.addfinalizer(fin2, item)
            with pytest.raises(ExceptionGroup) as err:
                ss.teardown_exact(None)

            # Note that finalizers are run LIFO, but because FIFO is more intuitive
            # for users we reverse the order of messages, and see the error from
            # fin1 first.
            err1, err2 = err.value.exceptions
            assert err1.args == ("oops1",)
            assert err2.args == ("oops2",)

    def test_teardown_multiple_scopes_one_fails(self, tmp_path: Path) -> None:
        module_teardown = []

        def fin_func():
            raise Exception("oops1")

        def fin_module():
            module_teardown.append("fin_module")

        def test_func(): ...

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            mod = item.listchain()[-2]
            ss = item.session._setupstate
            ss.setup(item)
            ss.addfinalizer(fin_module, mod)
            ss.addfinalizer(fin_func, item)
            with pytest.raises(Exception, match="oops1"):
                ss.teardown_exact(None)
            assert module_teardown == ["fin_module"]

    def test_teardown_multiple_scopes_several_fail(self, tmp_path: Path) -> None:
        def raiser(exc):
            raise exc

        def test_func(): ...

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            mod = item.listchain()[-2]
            ss = item.session._setupstate
            ss.setup(item)
            ss.addfinalizer(partial(raiser, KeyError("from module scope")), mod)
            ss.addfinalizer(partial(raiser, TypeError("from function scope 1")), item)
            ss.addfinalizer(partial(raiser, ValueError("from function scope 2")), item)

            with pytest.raises(
                ExceptionGroup, match="errors during test teardown"
            ) as e:
                ss.teardown_exact(None)
        # renamed from ``mod``/``func``: the module collector above already
        # holds ``mod``, and mypy now checks this function because it takes an
        # annotated ``tmp_path`` rather than a bare ``pytester``
        mod_exc, func_exc = e.value.exceptions
        assert isinstance(mod_exc, KeyError)
        assert isinstance(func_exc.exceptions[0], TypeError)
        assert isinstance(func_exc.exceptions[1], ValueError)

    # ensemble: the subject is a *collector* whose ``setup()`` raises, which is
    # what caches the exception in ``SetupState``. Ensembles only collect
    # modules/classes/functions, so there is no way to hand one a custom
    # ``pytest.Collector`` (nor to serve one from ``pytest_collect_file``, since
    # the session's collect report is preset).
    def test_cached_exception_doesnt_get_longer(self, pytester: Pytester) -> None:
        """Regression test for #12204 (the "BTW" case)."""
        pytester.makepyfile(test="")
        # If the collector.setup() raises, all collected items error with this
        # exception.
        pytester.makeconftest(
            """
            import pytest

            class MyItem(pytest.Item):
                def runtest(self) -> None: pass

            class MyBadCollector(pytest.Collector):
                def collect(self):
                    return [
                        MyItem.from_parent(self, name="one"),
                        MyItem.from_parent(self, name="two"),
                        MyItem.from_parent(self, name="three"),
                    ]

                def setup(self):
                    1 / 0

            def pytest_collect_file(file_path, parent):
                if file_path.name == "test.py":
                    return MyBadCollector.from_parent(parent, name='bad')
            """
        )

        result = pytester.runpytest_inprocess("--tb=native")
        assert result.ret == ExitCode.TESTS_FAILED
        failures = result.reprec.getfailures()  # type: ignore[attr-defined]
        assert len(failures) == 3
        lines1 = failures[1].longrepr.reprtraceback.reprentries[0].lines
        lines2 = failures[2].longrepr.reprtraceback.reprentries[0].lines
        assert len(lines1) == len(lines2)


class BaseFunctionalTests:
    def getrunner(self) -> ProtocolRunner:
        """The runtest protocol runner; provided by the concrete subclass."""
        raise NotImplementedError

    def test_passfunction(self, tmp_path: Path) -> None:
        def test_func(): ...

        reports = runitem(self.getrunner, tmp_path, test_func)
        rep = reports[1]
        assert rep.passed
        assert not rep.failed
        assert rep.outcome == "passed"
        assert not rep.longrepr

    def test_failfunction(self, tmp_path: Path) -> None:
        def test_func():
            assert 0

        reports = runitem(self.getrunner, tmp_path, test_func)
        rep = reports[1]
        assert not rep.passed
        assert not rep.skipped
        assert rep.failed
        assert rep.when == "call"
        assert rep.outcome == "failed"
        # assert isinstance(rep.longrepr, ReprExceptionInfo)

    def test_skipfunction(self, tmp_path: Path) -> None:
        def test_func():
            pytest.skip("hello")

        reports = runitem(self.getrunner, tmp_path, test_func)
        rep = reports[1]
        assert not rep.failed
        assert not rep.passed
        assert rep.skipped
        assert rep.outcome == "skipped"
        # assert rep.skipped.when == "call"
        # assert rep.skipped.when == "call"
        # assert rep.skipped == "%sreason == "hello"
        # assert rep.skipped.location.lineno == 3
        # assert rep.skipped.location.path
        # assert not rep.skipped.failurerepr

    def test_skip_in_setup_function(self, tmp_path: Path) -> None:
        def setup_function(func):
            pytest.skip("hello")

        def test_func(): ...

        reports = runitem(self.getrunner, tmp_path, setup_function, test_func)
        print(reports)
        rep = reports[0]
        assert not rep.failed
        assert not rep.passed
        assert rep.skipped
        # assert rep.skipped.reason == "hello"
        # assert rep.skipped.location.lineno == 3
        # assert rep.skipped.location.lineno == 3
        assert len(reports) == 2
        assert reports[1].passed  # teardown

    def test_failure_in_setup_function(self, tmp_path: Path) -> None:
        def setup_function(func):
            raise ValueError(42)

        def test_func(): ...

        reports = runitem(self.getrunner, tmp_path, setup_function, test_func)
        rep = reports[0]
        assert not rep.skipped
        assert not rep.passed
        assert rep.failed
        assert rep.when == "setup"
        assert len(reports) == 2

    def test_failure_in_teardown_function(self, tmp_path: Path) -> None:
        def teardown_function(func):
            raise ValueError(42)

        def test_func(): ...

        reports = runitem(self.getrunner, tmp_path, teardown_function, test_func)
        print(reports)
        assert len(reports) == 3
        rep = reports[2]
        assert not rep.skipped
        assert not rep.passed
        assert rep.failed
        assert rep.when == "teardown"
        # assert rep.longrepr.reprcrash.lineno == 3
        # assert rep.longrepr.reprtraceback.reprentries

    def test_custom_failure_repr(self, tmp_path: Path) -> None:
        # The original wrote a conftest defining ``class Function(pytest.Function)``
        # with a custom ``repr_failure``. Node class customization by name was
        # removed long ago - ``python.py`` instantiates ``Function`` directly - so
        # that conftest was never consulted, and every assertion that would have
        # noticed is commented out below. Nothing is lost by dropping it.
        def test_func():
            assert 0

        reports = runitem(self.getrunner, tmp_path, test_func)
        rep = reports[1]
        assert not rep.skipped
        assert not rep.passed
        assert rep.failed
        # assert rep.outcome.when == "call"
        # assert rep.failed.where.lineno == 3
        # assert rep.failed.where.path.basename == "test_func.py"
        # assert rep.failed.failurerepr == "hello"

    def test_teardown_final_returncode(self, tmp_path: Path) -> None:
        def test_func(): ...

        def teardown_function(func):
            raise ValueError(42)

        with Ensemble(test_func, teardown_function, rootpath=tmp_path) as ensemble:
            record = ensemble.run()
            # ``inline_run`` reported ``ret == 1`` here; an ensemble has no
            # ``wrap_session`` to turn the session into an exit code, so assert
            # on the counter that exit code is computed from.
            assert ensemble.session.testsfailed == 1
        # the call passes and the teardown fails, so this is an error
        record.assert_outcomes(passed=1, errors=1)

    def test_logstart_logfinish_hooks(self, tmp_path: Path) -> None:
        events: list[tuple[str, str, tuple[str, int | None, str]]] = []

        class LogHooks:
            def pytest_runtest_logstart(self, nodeid, location):
                events.append(("pytest_runtest_logstart", nodeid, location))

            def pytest_runtest_logfinish(self, nodeid, location):
                events.append(("pytest_runtest_logfinish", nodeid, location))

        def test_func(): ...

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(LogHooks(),))
        with Ensemble(test_func, spec=spec) as ensemble:
            (item,) = ensemble.collect()
            ensemble.run()

        assert [name for name, _, _ in events] == [
            "pytest_runtest_logstart",
            "pytest_runtest_logfinish",
        ]
        for _, nodeid, location in events:
            assert nodeid == item.nodeid == "test_ensemble.py::test_func"
            # the path and line number are host-anchored - the item's code
            # object really lives in this file - so only the name transfers
            # verbatim; the rest is asserted against the item it came from.
            assert location == item.location
            assert location[2] == "test_func"

    def test_exact_teardown_issue90(self, tmp_path: Path) -> None:
        class TestClass:
            def test_method(self): ...

            def teardown_class(cls):
                raise Exception()

        def test_func():
            import traceback

            # on python2 exc_info is kept till a function exits
            # so we would end up calling test functions while
            # sys.exc_info would return the indexerror
            # from guessing the lastitem
            excinfo = sys.exc_info()
            assert excinfo[0] is None, traceback.format_exception(*excinfo)

        def teardown_function(func):
            raise ValueError(42)

        # collection follows argument order: the class first, then test_func
        record = run_tests(TestClass, test_func, teardown_function, rootpath=tmp_path)
        reps = record.reports
        print(reps)
        for i in range(2):
            assert reps[i].nodeid.endswith("test_method")
            assert reps[i].passed
        assert reps[2].when == "teardown"
        assert reps[2].failed
        assert len(reps) == 6
        for i in range(3, 5):
            assert reps[i].nodeid.endswith("test_func")
            assert reps[i].passed
        assert reps[5].when == "teardown"
        assert reps[5].nodeid.endswith("test_func")
        assert reps[5].failed

    def test_exact_teardown_issue1206(self, tmp_path: Path) -> None:
        """Issue shadowing error with wrong number of arguments on teardown_method."""

        class TestClass:
            def teardown_method(self, x, y, z): ...

            def test_method(self):
                assert True

        record = run_tests(TestClass, rootpath=tmp_path)
        reps = record.reports
        print(reps)
        assert len(reps) == 3
        #
        assert reps[0].nodeid.endswith("test_method")
        assert reps[0].passed
        assert reps[0].when == "setup"
        #
        assert reps[1].nodeid.endswith("test_method")
        assert reps[1].passed
        assert reps[1].when == "call"
        #
        assert reps[2].nodeid.endswith("test_method")
        assert reps[2].failed
        assert reps[2].when == "teardown"
        longrepr = reps[2].longrepr
        assert isinstance(longrepr, ExceptionChainRepr)
        assert longrepr.reprcrash
        # the qualname python puts in front is host-anchored - the class is
        # defined inside this test, so it reads
        # ``test_exact_teardown_issue1206.<locals>.TestClass.teardown_method``
        assert longrepr.reprcrash.message.startswith("TypeError: ")
        assert longrepr.reprcrash.message.endswith(
            "teardown_method() missing 2 required positional arguments: 'y' and 'z'"
        )

    def test_failure_in_setup_function_ignores_custom_repr(
        self, tmp_path: Path
    ) -> None:
        # As in test_custom_failure_repr above, the conftest ``Function`` subclass
        # the original defined has not been consulted by the collection machinery
        # for a long time, so dropping it changes nothing that was asserted.
        def setup_function(func):
            raise ValueError(42)

        def test_func(): ...

        reports = runitem(self.getrunner, tmp_path, setup_function, test_func)
        assert len(reports) == 2
        rep = reports[0]
        print(rep)
        assert not rep.skipped
        assert not rep.passed
        assert rep.failed
        # assert rep.outcome.when == "setup"
        # assert rep.outcome.where.lineno == 3
        # assert rep.outcome.where.path.basename == "test_func.py"
        # assert isinstance(rep.failed.failurerepr, PythonFailureRepr)

    def test_systemexit_does_not_bail_out(self, tmp_path: Path) -> None:
        def test_func():
            raise SystemExit(42)

        try:
            reports = runitem(self.getrunner, tmp_path, test_func)
        except SystemExit:
            assert False, "runner did not catch SystemExit"
        rep = reports[1]
        assert rep.failed
        assert rep.when == "call"

    def test_exit_propagates(self, tmp_path: Path) -> None:
        def test_func():
            raise pytest.exit.Exception()

        with pytest.raises(pytest.exit.Exception):
            runitem(self.getrunner, tmp_path, test_func)


class TestExecutionNonForked(BaseFunctionalTests):
    def getrunner(self) -> ProtocolRunner:
        def f(item):
            return runner.runtestprotocol(item, log=False)

        return f

    def test_keyboardinterrupt_propagates(self, tmp_path: Path) -> None:
        def test_func():
            raise KeyboardInterrupt("fake")

        with pytest.raises(KeyboardInterrupt):
            runitem(self.getrunner, tmp_path, test_func)

    def test_keyboardinterrupt_clears_request_and_funcargs(
        self, tmp_path: Path
    ) -> None:
        """Ensure that an item's fixtures are cleared quickly even if exiting
        early due to a keyboard interrupt (#13626)."""

        @pytest.fixture
        def resource():
            return object()

        def test_func(resource):
            raise KeyboardInterrupt("fake")

        with Ensemble(resource, test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            assert isinstance(item, pytest.Function)
            assert item._request
            assert item.funcargs == {}

            with pytest.raises(KeyboardInterrupt):
                runner.runtestprotocol(item, log=False)

            assert not cast(object, item._request)
            assert not item.funcargs


class TestSessionReports:
    def test_collect_result(self, tmp_path: Path) -> None:
        def test_func1(): ...

        class TestClass: ...

        module = build_module("test_collect_result", test_func1, TestClass)
        with Ensemble(rootpath=tmp_path) as ensemble:
            col = EnsembleModule.from_parent(ensemble.session, obj=module)
            rep = runner.collect_one_node(col)
        assert not rep.failed
        assert not rep.skipped
        assert rep.passed
        locinfo = rep.location
        assert locinfo is not None
        assert locinfo[0] == col.path.name
        assert not locinfo[1]
        assert locinfo[2] == col.path.name
        res = rep.result
        assert len(res) == 2
        assert res[0].name == "test_func1"
        assert res[1].name == "TestClass"


reporttypes: list[type[reports.BaseReport]] = [
    reports.BaseReport,
    reports.TestReport,
    reports.CollectReport,
]


@pytest.mark.parametrize(
    "reporttype", reporttypes, ids=[x.__name__ for x in reporttypes]
)
def test_report_extra_parameters(reporttype: type[reports.BaseReport]) -> None:
    args = list(inspect.signature(reporttype.__init__).parameters.keys())[1:]
    basekw: dict[str, list[object]] = {arg: [] for arg in args}
    report = reporttype(newthing=1, **basekw)
    assert report.newthing == 1


def test_callinfo() -> None:
    ci = runner.CallInfo.from_call(lambda: 0, "collect")
    assert ci.when == "collect"
    assert ci.result == 0
    assert "result" in repr(ci)
    assert repr(ci) == "<CallInfo when='collect' result: 0>"
    assert str(ci) == "<CallInfo when='collect' result: 0>"

    ci2 = runner.CallInfo.from_call(lambda: 0 / 0, "collect")
    assert ci2.when == "collect"
    assert not hasattr(ci2, "result")
    assert repr(ci2) == f"<CallInfo when='collect' excinfo={ci2.excinfo!r}>"
    assert str(ci2) == repr(ci2)
    assert ci2.excinfo

    # Newlines are escaped.
    def raise_assertion():
        assert 0, "assert_msg"

    ci3 = runner.CallInfo.from_call(raise_assertion, "call")
    assert repr(ci3) == f"<CallInfo when='call' excinfo={ci3.excinfo!r}>"
    assert "\n" not in repr(ci3)


# design question: do we want general hooks in python files?
# then something like the following functional tests makes sense


# ensemble: the subject is a ``pytest_runtest_setup`` hook defined at module
# level in the test file. Module-level hooks are registered by
# ``consider_module`` when the module is imported, and an ensemble module is
# served from memory instead of imported, so its hooks are never registered.
@pytest.mark.xfail
def test_runtest_in_module_ordering(pytester: Pytester) -> None:
    p1 = pytester.makepyfile(
        """
        import pytest
        def pytest_runtest_setup(item): # runs after class-level!
            item.function.mylist.append("module")
        class TestClass(object):
            def pytest_runtest_setup(self, item):
                assert not hasattr(item.function, 'mylist')
                item.function.mylist = ['class']
            @pytest.fixture
            def mylist(self, request):
                return request.function.mylist
            @pytest.hookimpl(wrapper=True)
            def pytest_runtest_call(self, item):
                try:
                    yield
                except ValueError:
                    pass
            def test_hello1(self, mylist):
                assert mylist == ['class', 'module'], mylist
                raise ValueError()
            def test_hello2(self, mylist):
                assert mylist == ['class', 'module'], mylist
        def pytest_runtest_teardown(item):
            del item.function.mylist
    """
    )
    result = pytester.runpytest(p1)
    result.stdout.fnmatch_lines(["*2 passed*"])


def test_outcomeexception_exceptionattributes() -> None:
    outcome = outcomes.OutcomeException("test")
    assert outcome.args[0] == outcome.msg


def test_outcomeexception_passes_except_Exception() -> None:
    with pytest.raises(outcomes.OutcomeException):
        try:
            raise outcomes.OutcomeException("test")
        except Exception as e:
            raise NotImplementedError from e


def test_pytest_exit() -> None:
    with pytest.raises(pytest.exit.Exception) as excinfo:
        pytest.exit("hello")
    assert excinfo.errisinstance(pytest.exit.Exception)


def test_pytest_fail() -> None:
    with pytest.raises(pytest.fail.Exception) as excinfo:
        pytest.fail("hello")
    s = excinfo.exconly(tryshort=True)
    assert s.startswith("Failed")


# ensemble: ``pytest.exit`` from ``pytest_configure`` is rendered onto stderr by
# ``wrap_session``, which ensembles do not run - the Exit would simply propagate
# out of ``configured()``.
def test_pytest_exit_msg(pytester: Pytester) -> None:
    pytester.makeconftest(
        """
    import pytest

    def pytest_configure(config):
        pytest.exit('oh noes')
    """
    )
    result = pytester.runpytest()
    result.stderr.fnmatch_lines(["Exit: oh noes"])


def _strip_resource_warnings(lines):
    # Assert no output on stderr, except for unreliable ResourceWarnings.
    # (https://github.com/pytest-dev/pytest/issues/5088)
    return [
        x
        for x in lines
        if not x.startswith(("Exception ignored in:", "ResourceWarning"))
    ]


# ensemble: about the process return code and the stderr message, both produced
# by ``wrap_session``; inside an ensemble ``pytest.exit`` just propagates as
# ``Exit``.
def test_pytest_exit_returncode(pytester: Pytester) -> None:
    pytester.makepyfile(
        """\
        import pytest
        def test_foo():
            pytest.exit("some exit msg", 99)
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*! *Exit: some exit msg !*"])

    assert _strip_resource_warnings(result.stderr.lines) == []
    assert result.ret == 99

    # It prints to stderr also in case of exit during pytest_sessionstart.
    pytester.makeconftest(
        """\
        import pytest

        def pytest_sessionstart():
            pytest.exit("during_sessionstart", 98)
        """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*! *Exit: during_sessionstart !*"])
    assert _strip_resource_warnings(result.stderr.lines) == [
        "Exit: during_sessionstart"
    ]
    assert result.ret == 98


def test_pytest_fail_notrace_runtest(tmp_path: Path) -> None:
    """Test pytest.fail(..., pytrace=False) does not show tracebacks during test run."""

    def test_hello():
        pytest.fail("hello", pytrace=False)

    def teardown_function(function):
        pytest.fail("world", pytrace=False)

    record = run_tests(
        test_hello, teardown_function, rootpath=tmp_path, capture_output=True
    )
    # the call fails and the teardown errors
    record.assert_outcomes(failed=1, errors=1)
    # errors are summarized before failures, hence "world" before "hello"
    record.stdout.fnmatch_lines(["world", "hello"])
    record.stdout.no_fnmatch_line("*def teardown_function*")


# ensemble: the failure happens while the test module is imported, and an
# ensemble module is built in memory rather than imported.
def test_pytest_fail_notrace_collection(pytester: Pytester) -> None:
    """Test pytest.fail(..., pytrace=False) does not show tracebacks during collection."""
    pytester.makepyfile(
        """
        import pytest
        def some_internal_function():
            pytest.fail("hello", pytrace=False)
        some_internal_function()
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["hello"])
    result.stdout.no_fnmatch_line("*def some_internal_function()*")


def test_pytest_fail_notrace_non_ascii(tmp_path: Path) -> None:
    """Fix pytest.fail with pytrace=False with non-ascii characters (#1178).

    This tests with native and unicode strings containing non-ascii chars.
    """

    def test_hello():
        pytest.fail("oh oh: ☺", pytrace=False)

    record = run_tests(test_hello, rootpath=tmp_path, capture_output=True)
    record.assert_outcomes(failed=1)
    record.stdout.fnmatch_lines(["*test_hello*", "oh oh: ☺"])
    record.stdout.no_fnmatch_line("*def test_hello*")


# ensemble: entirely about the exit status of a run, which is computed by
# ``wrap_session`` from a session an ensemble does not have.
def test_pytest_no_tests_collected_exit_status(pytester: Pytester) -> None:
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*collected 0 items*"])
    assert result.ret == ExitCode.NO_TESTS_COLLECTED

    pytester.makepyfile(
        test_foo="""
        def test_foo():
            assert 1
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*collected 1 item*"])
    result.stdout.fnmatch_lines(["*1 passed*"])
    assert result.ret == ExitCode.OK

    result = pytester.runpytest("-k nonmatch")
    result.stdout.fnmatch_lines(["*collected 1 item*"])
    result.stdout.fnmatch_lines(["*1 deselected*"])
    assert result.ret == ExitCode.NO_TESTS_COLLECTED


def test_exception_printing_skip() -> None:
    assert pytest.skip.Exception == pytest.skip.Exception
    try:
        pytest.skip("hello")
    except pytest.skip.Exception:
        excinfo = ExceptionInfo.from_current()
        s = excinfo.exconly(tryshort=True)
        assert s.startswith("Skipped")


def test_importorskip(monkeypatch) -> None:
    importorskip = pytest.importorskip

    def f():
        importorskip("asdlkj")

    try:
        sysmod = importorskip("sys")
        assert sysmod is sys
        # path = pytest.importorskip("os.path")
        # assert path == os.path
        excinfo = pytest.raises(pytest.skip.Exception, f)
        assert excinfo is not None
        excrepr = excinfo.getrepr()
        assert excrepr is not None
        assert excrepr.reprcrash is not None
        path = Path(excrepr.reprcrash.path)
        # check that importorskip reports the actual call
        # in this test the test_runner.py file
        assert path.stem == "test_runner"
        with pytest.raises(SyntaxError):
            pytest.importorskip("x y z")
        with pytest.raises(SyntaxError):
            pytest.importorskip("x=y")
        mod = types.ModuleType("hello123")
        mod.__version__ = "1.3"  # type: ignore
        monkeypatch.setitem(sys.modules, "hello123", mod)
        with pytest.raises(pytest.skip.Exception):
            pytest.importorskip("hello123", minversion="1.3.1")
        mod2 = pytest.importorskip("hello123", minversion="1.3")
        assert mod2 == mod
    except pytest.skip.Exception:  # pragma: no cover
        assert False, f"spurious skip: {ExceptionInfo.from_current()}"


def test_importorskip_imports_last_module_part() -> None:
    ospath = pytest.importorskip("os.path")
    assert os.path == ospath


class TestImportOrSkipExcType:
    """Tests for importorskip's exc_type behavior."""

    def test_module_not_found_skips_by_default(self) -> None:
        with pytest.raises(pytest.skip.Exception):
            pytest.importorskip(
                "TestImportOrSkipExcType_test_module_not_found_skips_without_warning"
            )

    # ensemble: needs a real importable module on sys.path that raises on import.
    def test_import_error_is_propagated_by_default(self, pytester: Pytester) -> None:
        fn = pytester.makepyfile("raise ImportError('some specific problem')")
        pytester.syspathinsert()

        with pytest.raises(ImportError, match="some specific problem"):
            pytest.importorskip(fn.stem)

    # ensemble: needs a real importable module on sys.path that raises on import.
    def test_import_error_can_be_captured_explicitly(self, pytester: Pytester) -> None:
        fn = pytester.makepyfile("raise ImportError('some specific problem')")
        pytester.syspathinsert()

        with pytest.raises(pytest.skip.Exception):
            pytest.importorskip(fn.stem, exc_type=ImportError)

    # ensemble: needs a second, real module whose import raises ImportError.
    def test_import_error_integration(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            def test_foo():
                pytest.importorskip("warning_integration_module")
            """
        )
        pytester.makepyfile(
            warning_integration_module="""
                raise ImportError("required library foobar not compiled properly")
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            ["*ImportError: required library foobar not compiled properly*"]
        )
        result.assert_outcomes(failed=1)


def test_importorskip_dev_module(monkeypatch) -> None:
    try:
        mod = types.ModuleType("mockmodule")
        mod.__version__ = "0.13.0.dev-43290"  # type: ignore
        monkeypatch.setitem(sys.modules, "mockmodule", mod)
        mod2 = pytest.importorskip("mockmodule", minversion="0.12.0")
        assert mod2 == mod
        with pytest.raises(pytest.skip.Exception):
            pytest.importorskip("mockmodule1", minversion="0.14.0")
    except pytest.skip.Exception:  # pragma: no cover
        assert False, f"spurious skip: {ExceptionInfo.from_current()}"


# ensemble: the skip is raised while the test module is imported, which is what
# turns it into a collection-level skip; an ensemble module is already built.
def test_importorskip_module_level(pytester: Pytester) -> None:
    """`importorskip` must be able to skip entire modules when used at module level."""
    pytester.makepyfile(
        """
        import pytest
        foobarbaz = pytest.importorskip("foobarbaz")

        def test_foo():
            pass
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*collected 0 items / 1 skipped*"])


# ensemble: as above, a module level skip raised during import.
def test_importorskip_custom_reason(pytester: Pytester) -> None:
    """Make sure custom reasons are used."""
    pytester.makepyfile(
        """
        import pytest
        foobarbaz = pytest.importorskip("foobarbaz2", reason="just because")

        def test_foo():
            pass
    """
    )
    result = pytester.runpytest("-ra")
    result.stdout.fnmatch_lines(["*just because*"])
    result.stdout.fnmatch_lines(["*collected 0 items / 1 skipped*"])


# ensemble: runs pytest in a subprocess.
def test_pytest_cmdline_main(pytester: Pytester) -> None:
    p = pytester.makepyfile(
        """
        import pytest
        def test_hello():
            assert 1
        if __name__ == '__main__':
           pytest.cmdline.main([__file__])
    """
    )
    import subprocess

    popen = subprocess.Popen([sys.executable, str(p)], stdout=subprocess.PIPE)
    popen.communicate()
    ret = popen.wait()
    assert ret == 0


# ensemble: the subject is writing a non-ascii longrepr to the *real* output
# stream and its encoding. An ensemble renders into a StringIO, where a
# UnicodeEncodeError can never happen, so the ported test would be vacuous.
def test_unicode_in_longrepr(pytester: Pytester) -> None:
    pytester.makeconftest(
        """\
        import pytest
        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_makereport():
            rep = yield
            if rep.when == "call":
                rep.longrepr = 'ä'
            return rep
        """
    )
    pytester.makepyfile(
        """
        def test_out():
            assert 0
    """
    )
    result = pytester.runpytest()
    assert result.ret == 1
    assert "UnicodeEncodeError" not in result.stderr.str()


def test_failure_in_setup(tmp_path: Path) -> None:
    def setup_module():
        raise ZeroDivisionError

    def test_func(): ...

    # ``--tb`` is a terminal plugin option, so this needs the rendering config
    spec = ConfigSpec(rootpath=tmp_path, args=("--tb=line",))
    record = run_tests(setup_module, test_func, spec=spec, capture_output=True)
    # the xunit setup failure is a setup phase error
    record.assert_outcomes(errors=1)
    record.stdout.no_fnmatch_line("*def setup_module*")


def test_makereport_getsource(tmp_path: Path) -> None:
    # the assertion below matches the rendered source line verbatim, so this
    # block is kept off the formatter
    # fmt: off
    def test_foo():
        if False: pass  # type: ignore[unreachable]  # noqa: E701
        else: assert False  # noqa: E701
    # fmt: on

    record = run_tests(test_foo, rootpath=tmp_path, capture_output=True)
    record.assert_outcomes(failed=1)
    record.stdout.no_fnmatch_line("*INTERNALERROR*")
    record.stdout.fnmatch_lines(["*else: assert False*"])


def test_makereport_getsource_dynamic_code(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Test that exception in dynamically generated code doesn't break getting the source line."""
    import inspect

    original_findsource = inspect.findsource

    def findsource(obj):
        # Can be triggered by dynamically created functions
        if obj.__name__ == "foo":
            raise IndexError()
        return original_findsource(obj)

    monkeypatch.setattr(inspect, "findsource", findsource)

    @pytest.fixture
    def foo(missing): ...

    def test_fix(foo):
        assert False

    spec = ConfigSpec(rootpath=tmp_path, args=("-vv",))
    record = run_tests(foo, test_fix, spec=spec, capture_output=True)
    # the fixture is missing, so setup errors rather than the call failing
    record.assert_outcomes(errors=1)
    record.stdout.no_fnmatch_line("*INTERNALERROR*")
    record.stdout.fnmatch_lines(["*test_fix*", "*fixture*'missing'*not found*"])


def test_store_except_info_on_error() -> None:
    """Test that upon test failure, the exception info is stored on
    sys.last_traceback and friends."""

    # Simulate item that might raise a specific exception, depending on `raise_error` class var
    class ItemMightRaise:
        nodeid = "item_that_raises"
        raise_error = True

        def runtest(self):
            if self.raise_error:
                raise IndexError("TEST")

    try:
        runner.pytest_runtest_call(ItemMightRaise())  # type: ignore[arg-type]
    except IndexError:
        pass
    # Check that exception info is stored on sys
    assert sys.last_type is IndexError
    assert isinstance(sys.last_value, IndexError)
    if sys.version_info >= (3, 12, 0):
        assert isinstance(sys.last_exc, IndexError)  # type:ignore[attr-defined]

    assert sys.last_value.args[0] == "TEST"
    assert sys.last_traceback

    # The next run should clear the exception info stored by the previous run
    ItemMightRaise.raise_error = False
    runner.pytest_runtest_call(ItemMightRaise())  # type: ignore[arg-type]
    assert not hasattr(sys, "last_type")
    assert not hasattr(sys, "last_value")
    if sys.version_info >= (3, 12, 0):
        assert not hasattr(sys, "last_exc")
    assert not hasattr(sys, "last_traceback")


def test_current_test_env_var(tmp_path: Path) -> None:
    # the smuggling through a ``sys`` attribute the original needed to get the
    # values out of a separately imported module is a plain closure here
    pytest_current_test_vars: list[tuple[str, str]] = []

    @pytest.fixture
    def fix():
        pytest_current_test_vars.append(("setup", os.environ["PYTEST_CURRENT_TEST"]))
        yield
        pytest_current_test_vars.append(("teardown", os.environ["PYTEST_CURRENT_TEST"]))

    def test(fix):
        pytest_current_test_vars.append(("call", os.environ["PYTEST_CURRENT_TEST"]))

    record = run_tests(fix, test, rootpath=tmp_path)
    record.assert_outcomes(passed=1)
    test_id = "test_ensemble.py::test"
    assert pytest_current_test_vars == [
        ("setup", test_id + " (setup)"),
        ("call", test_id + " (call)"),
        ("teardown", test_id + " (teardown)"),
    ]
    # the inner run deletes the variable outright when the item is done, so it
    # is gone even though the host run set it for this very test
    assert "PYTEST_CURRENT_TEST" not in os.environ


class TestReportContents:
    """Test user-level API of ``TestReport`` objects."""

    def getrunner(self) -> ProtocolRunner:
        return lambda item: runner.runtestprotocol(item, log=False)

    def test_longreprtext_pass(self, tmp_path: Path) -> None:
        def test_func(): ...

        reports = runitem(self.getrunner, tmp_path, test_func)
        rep = reports[1]
        assert rep.longreprtext == ""

    def test_longreprtext_skip(self, tmp_path: Path) -> None:
        """TestReport.longreprtext can handle non-str ``longrepr`` attributes (#7559)"""

        def test_func():
            pytest.skip()

        reports = runitem(self.getrunner, tmp_path, test_func)
        _, call_rep, _ = reports
        assert isinstance(call_rep.longrepr, tuple)
        assert "Skipped" in call_rep.longreprtext

    # ensemble: the skip is raised while the module is imported, and an ensemble
    # module is built in memory instead of imported, so there is no collect
    # report to carry it.
    def test_longreprtext_collect_skip(self, pytester: Pytester) -> None:
        """CollectReport.longreprtext can handle non-str ``longrepr`` attributes (#7559)"""
        pytester.makepyfile(
            """
            import pytest
            pytest.skip(allow_module_level=True)
            """
        )
        rec = pytester.inline_run()
        calls = rec.getcalls("pytest_collectreport")
        _, call, _ = calls
        assert isinstance(call.report.longrepr, tuple)
        assert "Skipped" in call.report.longreprtext

    def test_longreprtext_failure(self, tmp_path: Path) -> None:
        def test_func():
            x = 1
            assert x == 4

        reports = runitem(self.getrunner, tmp_path, test_func)
        rep = reports[1]
        assert "assert 1 == 4" in rep.longreprtext

    # ensemble: the subject is the captured stdout/stderr on the reports, and
    # the capture plugin is not loaded inside an ensemble.
    def test_captured_text(self, pytester: Pytester) -> None:
        reports = pytester.runitem(
            """
            import pytest
            import sys

            @pytest.fixture
            def fix():
                sys.stdout.write('setup: stdout\\n')
                sys.stderr.write('setup: stderr\\n')
                yield
                sys.stdout.write('teardown: stdout\\n')
                sys.stderr.write('teardown: stderr\\n')
                assert 0

            def test_func(fix):
                sys.stdout.write('call: stdout\\n')
                sys.stderr.write('call: stderr\\n')
                assert 0
        """
        )
        setup, call, teardown = reports
        assert setup.capstdout == "setup: stdout\n"
        assert call.capstdout == "setup: stdout\ncall: stdout\n"
        assert teardown.capstdout == "setup: stdout\ncall: stdout\nteardown: stdout\n"

        assert setup.capstderr == "setup: stderr\n"
        assert call.capstderr == "setup: stderr\ncall: stderr\n"
        assert teardown.capstderr == "setup: stderr\ncall: stderr\nteardown: stderr\n"

    # ensemble: without the capture plugin ``capstdout``/``capstderr`` are empty
    # no matter what the test does, so a ported version would assert nothing.
    def test_no_captured_text(self, pytester: Pytester) -> None:
        reports = pytester.runitem(
            """
            def test_func():
                pass
        """
        )
        rep = reports[1]
        assert rep.capstdout == ""
        assert rep.capstderr == ""

    def test_longrepr_type(self, tmp_path: Path) -> None:
        def test_func():
            pytest.fail(pytrace=False)

        reports = runitem(self.getrunner, tmp_path, test_func)
        rep = reports[1]
        assert isinstance(rep.longrepr, ExceptionChainRepr)


def test_outcome_exception_bad_msg() -> None:
    """Check that OutcomeExceptions validate their input to prevent confusing errors (#5578)"""

    def func() -> None:
        raise NotImplementedError()

    expected = (
        "OutcomeException expected string as 'msg' parameter, got 'function' instead.\n"
        "Perhaps you meant to use a mark?"
    )
    with pytest.raises(TypeError) as excinfo:
        OutcomeException(func)  # type: ignore
    assert str(excinfo.value) == expected


# ensemble: ``PYTEST_VERSION`` is set and restored around ``main()``, which an
# ensemble never goes through - it builds its config directly.
def test_pytest_version_env_var(pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_VERSION", "old version")
    pytester.makepyfile(
        """
        import pytest
        import os


        def test():
            assert os.environ.get("PYTEST_VERSION") == pytest.__version__
    """
    )
    result = pytester.runpytest_inprocess()
    assert result.ret == ExitCode.OK
    assert os.environ["PYTEST_VERSION"] == "old version"


# ensemble: the subject is the session bailing out mid-run on ``--maxfail`` and
# tearing higher scoped fixtures down against the last item. ``run_items``
# drives ``pytest_runtest_protocol`` per item without the ``shouldstop`` /
# ``shouldfail`` handling ``pytest_runtestloop`` does, so nothing bails.
def test_teardown_session_failed(pytester: Pytester) -> None:
    """Test that higher-scoped fixture teardowns run in the context of the last
    item after the test session bails early due to --maxfail.

    Regression test for #11706.
    """
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture(scope="module")
        def baz():
            yield
            pytest.fail("This is a failing teardown")

        def test_foo(baz):
            pytest.fail("This is a failing test")

        def test_bar(): pass
        """
    )
    result = pytester.runpytest("--maxfail=1")
    result.assert_outcomes(failed=1, errors=1)


# ensemble: as above, plus ``--stepwise``, whose plugin (and the cache it needs)
# an ensemble config does not load.
def test_teardown_session_stopped(pytester: Pytester) -> None:
    """Test that higher-scoped fixture teardowns run in the context of the last
    item after the test session bails early due to --stepwise.

    Regression test for #11706.
    """
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture(scope="module")
        def baz():
            yield
            pytest.fail("This is a failing teardown")

        def test_foo(baz):
            pytest.fail("This is a failing test")

        def test_bar(): pass
        """
    )
    result = pytester.runpytest("--stepwise")
    result.assert_outcomes(failed=1, errors=1)

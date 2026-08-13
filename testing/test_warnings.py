# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import sys
import warnings

from _pytest.config import ExitCode
from _pytest.config import UsageError
from _pytest.ensemble import build_module
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import Ensemble
from _pytest.ensemble import run_tests
from _pytest.ensemble import RunRecord
from _pytest.ensemble import Source
from _pytest.fixtures import FixtureRequest
from _pytest.pytester import Pytester
import pytest


WARNINGS_SUMMARY_HEADER = "warnings summary"


class WarningCollector:
    """Ensemble plugin recording the full ``pytest_warning_recorded`` payload.

    ``RunRecord.warnings`` keeps only the ``WarningMessage``; the ``when``
    and ``nodeid`` a warning was reported with are what several of these
    tests are about, so they are collected here instead.
    """

    def __init__(self) -> None:
        self.collected: list[tuple[str, str, str, tuple[str, int, str] | None]] = []

    def pytest_warning_recorded(self, warning_message, when, nodeid, location):
        self.collected.append((str(warning_message.message), when, nodeid, location))


@pytest.fixture
def pyfile_with_warnings(pytester: Pytester, request: FixtureRequest) -> str:
    """Create a test file which calls a function in a module which generates warnings."""
    pytester.syspathinsert()
    module_name = request.function.__name__[len("test_") :] + "_module"
    test_file = pytester.makepyfile(
        f"""
        import {module_name}
        def test_func():
            assert {module_name}.foo() == 1
        """,
        **{
            module_name: """
            import warnings
            def foo():
                warnings.warn(UserWarning("user warning"))
                warnings.warn(RuntimeWarning("runtime warning"))
                return 1
            """,
        },
    )
    return str(test_file)


# ensemble: the subject is the rendered warnings summary, whose per-warning
# lines quote the file and line the warning was raised at; those are anchored
# in this host file for an ensemble source, so the patterns do not transfer.
@pytest.mark.filterwarnings("default::UserWarning", "default::RuntimeWarning")
def test_normal_flow(pytester: Pytester, pyfile_with_warnings) -> None:
    """Check that the warnings section is displayed."""
    result = pytester.runpytest(pyfile_with_warnings)
    result.stdout.fnmatch_lines(
        [
            f"*== {WARNINGS_SUMMARY_HEADER} ==*",
            "test_normal_flow.py::test_func",
            "*normal_flow_module.py:3: UserWarning: user warning",
            '*  warnings.warn(UserWarning("user warning"))',
            "*normal_flow_module.py:4: RuntimeWarning: runtime warning",
            '*  warnings.warn(RuntimeWarning("runtime warning"))',
            "* 1 passed, 2 warnings*",
        ]
    )


def emit_module_warnings() -> int:
    """Stand-in for the helper module imported by ``pyfile_with_warnings``."""
    warnings.warn(UserWarning("user warning"))
    warnings.warn(RuntimeWarning("runtime warning"))
    return 1


def test_setup_teardown_warnings(tmp_path: Path) -> None:
    @pytest.fixture
    def fix():
        warnings.warn(UserWarning("warning during setup"))
        yield
        warnings.warn(UserWarning("warning during teardown"))

    def test_func(fix):
        pass

    record = run_tests(
        fix,
        test_func,
        rootpath=tmp_path,
        spec=ConfigSpec(inicfg={"filterwarnings": ["always::UserWarning"]}),
    )
    # The original matched the two rendered warning lines; the file and line
    # they quote are host-anchored, so the messages are asserted directly.
    record.assert_outcomes(passed=1, warnings=2)
    setup_warning, teardown_warning = record.warnings
    assert setup_warning.category is UserWarning
    assert str(setup_warning.message) == "warning during setup"
    assert teardown_warning.category is UserWarning
    assert str(teardown_warning.message) == "warning during teardown"


@pytest.mark.parametrize("method", ["cmdline", "ini"])
def test_as_errors(tmp_path: Path, method) -> None:
    # The original needed a subprocess because ``-W error`` on a real command
    # line changes the process-wide filters; an ensemble's filters live in a
    # ``warnings.catch_warnings`` block scoped to its own config.
    def test_func():
        assert emit_module_warnings() == 1

    if method == "cmdline":
        spec = ConfigSpec(rootpath=tmp_path, args=("-W", "error"))
    else:
        spec = ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["error"]})
    record = run_tests(test_func, spec=spec, capture_output=True)
    record.assert_outcomes(failed=1)
    record.stdout.fnmatch_lines(
        [
            "E       UserWarning: user warning",
            "* 1 failed in *",
        ]
    )


@pytest.mark.parametrize("method", ["cmdline", "ini"])
def test_ignore(tmp_path: Path, method) -> None:
    def test_func():
        assert emit_module_warnings() == 1

    if method == "cmdline":
        spec = ConfigSpec(rootpath=tmp_path, args=("-W", "ignore"))
    else:
        spec = ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["ignore"]})
    record = run_tests(test_func, spec=spec)
    # Stronger than the original's "no warnings summary was rendered": not a
    # single warning was recorded.
    record.assert_outcomes(passed=1, warnings=0)
    assert record.warnings == []


def test_unicode(tmp_path: Path) -> None:
    @pytest.fixture
    def fix():
        warnings.warn("测试")
        yield

    def test_func(fix):
        pass

    record = run_tests(
        fix,
        test_func,
        rootpath=tmp_path,
        spec=ConfigSpec(inicfg={"filterwarnings": ["always::UserWarning"]}),
    )
    record.assert_outcomes(passed=1, warnings=1)
    (warning,) = record.warnings
    assert warning.category is UserWarning
    assert str(warning.message) == "测试"


# ensemble: the pre-installed filter is registered by module-level code run
# at import time, and an ensemble module body is never executed.
@pytest.mark.skip("issue #13485")
def test_works_with_filterwarnings(pytester: Pytester) -> None:
    """Ensure our warnings capture does not mess with pre-installed filters (#2430)."""
    pytester.makepyfile(
        """
        import warnings

        class MyWarning(Warning):
            pass

        warnings.filterwarnings("error", category=MyWarning)

        class TestWarnings(object):
            def test_my_warning(self):
                try:
                    warnings.warn(MyWarning("warn!"))
                    assert False
                except MyWarning:
                    assert True
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*== 1 passed in *"])


@pytest.mark.parametrize("default_config", ["ini", "cmdline"])
def test_filterwarnings_mark(tmp_path: Path, default_config) -> None:
    """Test ``filterwarnings`` mark works and takes precedence over command
    line and ini options."""

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_ignore_runtime_warning():
        warnings.warn(RuntimeWarning())

    @pytest.mark.filterwarnings("error")
    def test_warning_error():
        warnings.warn(RuntimeWarning())

    def test_show_warning():
        warnings.warn(RuntimeWarning())

    if default_config == "ini":
        spec = ConfigSpec(
            rootpath=tmp_path,
            inicfg={"filterwarnings": ["always::RuntimeWarning"]},
        )
    else:
        spec = ConfigSpec(rootpath=tmp_path, args=("-W", "always::RuntimeWarning"))
    record = run_tests(
        test_ignore_runtime_warning, test_warning_error, test_show_warning, spec=spec
    )
    record.assert_outcomes(passed=2, failed=1, warnings=1)
    assert record["test_warning_error"].failed
    (warning,) = record.warnings
    assert warning.category is RuntimeWarning


def test_non_string_warning_argument(tmp_path: Path) -> None:
    """Non-str argument passed to warning breaks pytest (#2956)"""

    def test():
        warnings.warn(UserWarning(1, "foo"))

    record = run_tests(
        test,
        rootpath=tmp_path,
        spec=ConfigSpec(args=("-W", "always::UserWarning")),
    )
    record.assert_outcomes(passed=1, warnings=1)
    (warning,) = record.warnings
    assert warning.category is UserWarning
    assert isinstance(warning.message, UserWarning)
    assert warning.message.args == (1, "foo")


def test_filterwarnings_mark_registration(tmp_path: Path) -> None:
    """Ensure filterwarnings mark is registered"""

    @pytest.mark.filterwarnings("error")
    def test_func():
        pass

    record = run_tests(
        test_func, rootpath=tmp_path, spec=ConfigSpec(args=("--strict-markers",))
    )
    # ``--strict-markers`` turns an unregistered mark into a collection error,
    # so a clean pass is exactly the original's ``ret == 0``.
    record.assert_outcomes(passed=1)
    assert record.collect_errors == []


def test_warning_recorded_hook(tmp_path: Path) -> None:
    class ConfigWarner:
        """Stands in for the conftest of the original."""

        def pytest_configure(self, config):
            config.issue_config_time_warning(
                UserWarning("config warning"), stacklevel=2
            )

    class CollectWarner:
        """Stands in for the module-level ``warnings.warn`` of the original;
        an ensemble module body is never executed, so the collect-phase
        warning is issued from a collection hook instead."""

        def pytest_collection_modifyitems(self):
            warnings.warn(UserWarning("collect warning"))

    @pytest.fixture
    def fix():
        warnings.warn(UserWarning("setup warning"))
        yield 1
        warnings.warn(UserWarning("teardown warning"))

    def test_func(fix):
        warnings.warn(UserWarning("call warning"))
        assert fix == 1

    warning_collector = WarningCollector()
    record = run_tests(
        build_module("test_warning_recorded_hook", fix=fix, test_func=test_func),
        spec=ConfigSpec(
            rootpath=tmp_path,
            inicfg={"filterwarnings": ["always::UserWarning"]},
            extra_plugins=(ConfigWarner(), CollectWarner(), warning_collector),
        ),
    )
    record.assert_outcomes(passed=1)
    collected = warning_collector.collected

    expected = [
        ("config warning", "config", ""),
        ("collect warning", "collect", ""),
        ("setup warning", "runtest", "test_warning_recorded_hook.py::test_func"),
        ("call warning", "runtest", "test_warning_recorded_hook.py::test_func"),
        ("teardown warning", "runtest", "test_warning_recorded_hook.py::test_func"),
    ]
    for collected_result, expected_result in zip(collected, expected, strict=True):
        assert collected_result[0] == expected_result[0], str(collected)
        assert collected_result[1] == expected_result[1], str(collected)
        assert collected_result[2] == expected_result[2], str(collected)

        # NOTE: collected_result[3] is location, which differs based on the platform you are on
        #       thus, the best we can do here is assert the types of the parameters match what we expect
        #       and not try and preload it in the expected array
        if collected_result[3] is not None:
            assert type(collected_result[3][0]) is str, str(collected)
            assert type(collected_result[3][1]) is int, str(collected)
            assert type(collected_result[3][2]) is str, str(collected)
        else:
            assert collected_result[3] is None, str(collected)


def test_collection_warnings(tmp_path: Path) -> None:
    """Check that we also capture warnings issued during test collection (#3251)."""

    class CollectWarner:
        """The original warns from the module body, which runs while the
        module is imported for collection; an ensemble module body is never
        executed, so the warning is issued from a collection hook."""

        def pytest_collection_modifyitems(self):
            warnings.warn(UserWarning("collection warning"))

    def test_foo():
        pass

    warning_collector = WarningCollector()
    record = run_tests(
        test_foo,
        spec=ConfigSpec(
            rootpath=tmp_path,
            inicfg={"filterwarnings": ["always::UserWarning"]},
            extra_plugins=(CollectWarner(), warning_collector),
        ),
    )
    record.assert_outcomes(passed=1, warnings=1)
    (warning,) = record.warnings
    assert warning.category is UserWarning
    assert str(warning.message) == "collection warning"
    # Stronger than the original: the warning is reported as a collect-phase
    # one, not merely rendered somewhere in the summary.
    assert warning_collector.collected == [("collection warning", "collect", "", None)]


def test_mark_regex_escape(tmp_path: Path) -> None:
    """@pytest.mark.filterwarnings should not try to escape regex characters (#3936)"""

    @pytest.mark.filterwarnings(r"ignore:some \(warning\)")
    def test_foo():
        warnings.warn(UserWarning("some (warning)"))

    record = run_tests(
        test_foo,
        rootpath=tmp_path,
        spec=ConfigSpec(inicfg={"filterwarnings": ["always::UserWarning"]}),
    )
    record.assert_outcomes(passed=1, warnings=0)
    assert record.warnings == []


@pytest.mark.filterwarnings("default::pytest.PytestWarning")
@pytest.mark.parametrize("ignore_pytest_warnings", ["no", "ini", "cmdline"])
def test_hide_pytest_internal_warnings(tmp_path: Path, ignore_pytest_warnings) -> None:
    """Make sure we can ignore internal pytest warnings using a warnings filter."""

    def test_bar():
        warnings.warn(pytest.PytestWarning("some internal warning"))

    # As in the original, the "no" case relies on the enclosing
    # ``default::pytest.PytestWarning`` mark: the inner run inherits the
    # process-global filters of the run driving it.
    inicfg: dict[str, object] = {}
    args: tuple[str, ...] = ()
    if ignore_pytest_warnings == "ini":
        inicfg = {"filterwarnings": ["ignore::pytest.PytestWarning"]}
    elif ignore_pytest_warnings == "cmdline":
        args = ("-W", "ignore::pytest.PytestWarning")
    record = run_tests(
        test_bar,
        spec=ConfigSpec(rootpath=tmp_path, inicfg=inicfg, args=args),
    )
    if ignore_pytest_warnings != "no":
        record.assert_outcomes(passed=1, warnings=0)
        assert record.warnings == []
    else:
        record.assert_outcomes(passed=1, warnings=1)
        (warning,) = record.warnings
        assert warning.category is pytest.PytestWarning
        assert str(warning.message) == "some internal warning"


@pytest.mark.parametrize("ignore_on_cmdline", [True, False])
def test_option_precedence_cmdline_over_ini(tmp_path: Path, ignore_on_cmdline) -> None:
    """Filters defined in the command-line should take precedence over filters in config files (#3946)."""

    def test():
        warnings.warn(UserWarning("hello"))

    spec = ConfigSpec(
        rootpath=tmp_path,
        inicfg={"filterwarnings": ["error::UserWarning"]},
        args=("-W", "ignore") if ignore_on_cmdline else (),
    )
    record = run_tests(test, spec=spec)
    if ignore_on_cmdline:
        record.assert_outcomes(passed=1, warnings=0)
    else:
        record.assert_outcomes(failed=1)


def test_option_precedence_mark(tmp_path: Path) -> None:
    """Filters defined by marks should always take precedence (#3946)."""

    @pytest.mark.filterwarnings("error")
    def test():
        warnings.warn(UserWarning("hello"))

    spec = ConfigSpec(
        rootpath=tmp_path,
        inicfg={"filterwarnings": ["ignore"]},
        args=("-W", "ignore"),
    )
    record = run_tests(test, spec=spec)
    record.assert_outcomes(failed=1)


def test_accept_unknown_category(tmp_path: Path, recwarn) -> None:
    """Category types that can't be imported don't cause failure (#13732)."""

    def test():
        pass

    spec = ConfigSpec(
        rootpath=tmp_path,
        inicfg={
            "filterwarnings": [
                "always:Failed to import filter module.*:pytest.PytestConfigWarning",
                "ignore::foobar.Foobar",
            ]
        },
        args=("-W", "ignore::bizbaz.Bizbaz"),
    )
    # The filters are (re)applied on entering every warnings-catching block,
    # so the same PytestConfigWarning is reported once per block rather than
    # exactly once; `recwarn` keeps the config-time ones, which escape an
    # ensemble, out of this run's own warnings summary.
    record = run_tests(test, spec=spec)
    record.assert_outcomes(passed=1)
    assert {w.category for w in record.warnings} == {pytest.PytestConfigWarning}
    assert {str(w.message) for w in record.warnings} == {
        "Failed to import filter module 'foobar': ignore::foobar.Foobar",
        "Failed to import filter module 'bizbaz': ignore::bizbaz.Bizbaz",
    }


class TestDeprecationWarningsByDefault:
    """
    Note: the original pytest runs are all executed in a subprocess so we don't
    inherit warning filters from pytest's own test suite. An ensemble does
    inherit them, but the "always" filters a config installs for
    (Pending)DeprecationWarning are prepended to whatever is in force, so the
    default still wins over this suite's ``filterwarnings = error``.
    """

    def create_sources(self, mark=None) -> tuple[object, Source]:
        """The sources of ``create_file``.

        The module-level ``warnings.warn`` of the original runs while the
        module is imported for collection; an ensemble module body is never
        executed, so that warning is issued from a collection hook instead.
        """

        class CollectWarner:
            def pytest_collection_modifyitems(self):
                warnings.warn(DeprecationWarning("collection"))

        def test_foo():
            warnings.warn(PendingDeprecationWarning("test run"))

        return CollectWarner(), test_foo if mark is None else mark(test_foo)

    @pytest.mark.parametrize("customize_filters", [True, False])
    def test_shown_by_default(self, tmp_path: Path, customize_filters) -> None:
        """Show deprecation warnings by default, even if user has customized the warnings filters (#4013)."""
        collect_warner, test_foo = self.create_sources()
        inicfg = {"filterwarnings": ["once::UserWarning"]} if customize_filters else {}
        record = run_tests(
            test_foo,
            spec=ConfigSpec(
                rootpath=tmp_path,
                inicfg=inicfg,
                extra_plugins=(collect_warner,),
            ),
        )
        record.assert_outcomes(passed=1, warnings=2)
        collection_warning, test_run_warning = record.warnings
        assert collection_warning.category is DeprecationWarning
        assert str(collection_warning.message) == "collection"
        assert test_run_warning.category is PendingDeprecationWarning
        assert str(test_run_warning.message) == "test run"

    def test_hidden_by_ini(self, tmp_path: Path) -> None:
        collect_warner, test_foo = self.create_sources()
        record = run_tests(
            test_foo,
            spec=ConfigSpec(
                rootpath=tmp_path,
                inicfg={
                    "filterwarnings": [
                        "ignore::DeprecationWarning",
                        "ignore::PendingDeprecationWarning",
                    ]
                },
                extra_plugins=(collect_warner,),
            ),
        )
        record.assert_outcomes(passed=1, warnings=0)
        assert record.warnings == []

    def test_hidden_by_mark(self, tmp_path: Path) -> None:
        """Should hide the deprecation warning from the function, but the warning during collection should
        be displayed normally.
        """
        collect_warner, test_foo = self.create_sources(
            mark=pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
        )
        record = run_tests(
            test_foo,
            spec=ConfigSpec(rootpath=tmp_path, extra_plugins=(collect_warner,)),
        )
        record.assert_outcomes(passed=1, warnings=1)
        (warning,) = record.warnings
        assert warning.category is DeprecationWarning
        assert str(warning.message) == "collection"

    def test_hidden_by_cmdline(self, tmp_path: Path) -> None:
        collect_warner, test_foo = self.create_sources()
        record = run_tests(
            test_foo,
            spec=ConfigSpec(
                rootpath=tmp_path,
                args=(
                    "-W",
                    "ignore::DeprecationWarning",
                    "-W",
                    "ignore::PendingDeprecationWarning",
                ),
                extra_plugins=(collect_warner,),
            ),
        )
        record.assert_outcomes(passed=1, warnings=0)
        assert record.warnings == []

    # ensemble: PYTHONWARNINGS is read by the interpreter at startup, so this
    # needs a fresh process.
    def test_hidden_by_system(self, pytester: Pytester, monkeypatch) -> None:
        pytester.makepyfile(
            """
            import pytest, warnings

            warnings.warn(DeprecationWarning("collection"))

            def test_foo():
                warnings.warn(PendingDeprecationWarning("test run"))
        """
        )
        monkeypatch.setenv("PYTHONWARNINGS", "once::UserWarning")
        result = pytester.runpytest_subprocess()
        assert WARNINGS_SUMMARY_HEADER not in result.stdout.str()

    def test_invalid_regex_in_filterwarning(self, tmp_path: Path) -> None:
        collect_warner, test_foo = self.create_sources()
        spec = ConfigSpec(
            rootpath=tmp_path,
            inicfg={"filterwarnings": ["ignore::DeprecationWarning:*"]},
            extra_plugins=(collect_warner,),
        )
        # The original asserted the rendered stderr of a usage error; here the
        # UsageError itself is caught, with the same message.
        with pytest.raises(UsageError) as excinfo:
            run_tests(test_foo, spec=spec)
        assert str(excinfo.value) == (
            "while parsing the following warning configuration:\n"
            "\n"
            "  ignore::DeprecationWarning:*\n"
            "\n"
            "This error occurred:\n"
            "\n"
            "Invalid regex '*': nothing to repeat at position 0\n"
        )


# ensemble: the source names ``pytest.PytestRemovedIn10Warning``, which does
# not exist yet; it only survives as text inside a makepyfile string.
@pytest.mark.skip("not relevant until pytest 10.0")
@pytest.mark.parametrize("change_default", [None, "ini", "cmdline"])
def test_removed_in_x_warning_as_error(pytester: Pytester, change_default) -> None:
    """This ensures that PytestRemovedInXWarnings raised by pytest are turned into errors.

    This test should be enabled as part of each major release, and skipped again afterwards
    to ensure our deprecations are turning into warnings as expected.
    """
    pytester.makepyfile(
        """
        import warnings, pytest
        def test():
            warnings.warn(pytest.PytestRemovedIn10Warning("some warning"))
    """
    )
    if change_default == "ini":
        pytester.makeini(
            """
            [pytest]
            filterwarnings =
                ignore::pytest.PytestRemovedIn10Warning
        """
        )

    args = (
        ("-Wignore::pytest.PytestRemovedIn10Warning",)
        if change_default == "cmdline"
        else ()
    )
    result = pytester.runpytest(*args)
    if change_default is None:
        result.stdout.fnmatch_lines(["* 1 failed in *"])
    else:
        assert change_default in ("ini", "cmdline")
        result.stdout.fnmatch_lines(["* 1 passed in *"])


class TestAssertionWarnings:
    @staticmethod
    def assert_result_warns(result, msg) -> None:
        result.stdout.fnmatch_lines([f"*PytestAssertRewriteWarning: {msg}*"])

    # ensemble: the warning is issued by the assertion rewriter, which is not
    # applied to ensemble sources.
    def test_tuple_warning(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """\
            def test_foo():
                assert (1,2)
            """
        )
        result = pytester.runpytest()
        self.assert_result_warns(
            result, "assertion is always true, perhaps remove parentheses?"
        )


def test_warnings_checker_twice() -> None:
    """Issue #4617"""
    expectation = pytest.warns(UserWarning)
    with expectation:
        warnings.warn("Message A", UserWarning)
    with expectation:
        warnings.warn("Message B", UserWarning)


# ensemble: the subject is how the summary groups warnings by rendered
# location, and every expected line quotes the example file's own path.
@pytest.mark.filterwarnings("always::UserWarning")
def test_group_warnings_by_message(pytester: Pytester) -> None:
    pytester.copy_example("warnings/test_group_warnings_by_message.py")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        [
            f"*== {WARNINGS_SUMMARY_HEADER} ==*",
            "test_group_warnings_by_message.py::test_foo[[]0[]]",
            "test_group_warnings_by_message.py::test_foo[[]1[]]",
            "test_group_warnings_by_message.py::test_foo[[]2[]]",
            "test_group_warnings_by_message.py::test_foo[[]3[]]",
            "test_group_warnings_by_message.py::test_foo[[]4[]]",
            "test_group_warnings_by_message.py::test_foo_1",
            "  */test_group_warnings_by_message.py:*: UserWarning: foo",
            "    warnings.warn(UserWarning(msg))",
            "",
            "test_group_warnings_by_message.py::test_bar[[]0[]]",
            "test_group_warnings_by_message.py::test_bar[[]1[]]",
            "test_group_warnings_by_message.py::test_bar[[]2[]]",
            "test_group_warnings_by_message.py::test_bar[[]3[]]",
            "test_group_warnings_by_message.py::test_bar[[]4[]]",
            "  */test_group_warnings_by_message.py:*: UserWarning: bar",
            "    warnings.warn(UserWarning(msg))",
            "",
            "-- Docs: *",
            "*= 11 passed, 11 warnings *",
        ],
        consecutive=True,
    )


# ensemble: as above, and the grouping counts are per real test file.
@pytest.mark.filterwarnings("always::UserWarning")
def test_group_warnings_by_message_summary(pytester: Pytester) -> None:
    pytester.copy_example("warnings/test_group_warnings_by_message_summary")
    pytester.syspathinsert()
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        [
            f"*== {WARNINGS_SUMMARY_HEADER} ==*",
            "test_1.py: 21 warnings",
            "test_2.py: 1 warning",
            "  */test_1.py:10: UserWarning: foo",
            "    warnings.warn(UserWarning(msg))",
            "",
            "test_1.py: 20 warnings",
            "  */test_1.py:10: UserWarning: bar",
            "    warnings.warn(UserWarning(msg))",
            "",
            "-- Docs: *",
            "*= 42 passed, 42 warnings *",
        ],
        consecutive=True,
    )


def test_pytest_configure_warning(tmp_path: Path, recwarn) -> None:
    """Issue 5115."""

    class ConfigureWarner:
        """Stands in for the conftest of the original."""

        def pytest_configure(self):
            warnings.warn("from pytest_configure")

    # A warning issued from ``pytest_configure`` is not recorded (the config
    # catches those without recording), but it must not blow the run up
    # either; the original's ``ret == 5`` was "no tests collected, no
    # internal error".
    record = run_tests(
        spec=ConfigSpec(rootpath=tmp_path, extra_plugins=(ConfigureWarner(),))
    )
    record.assert_outcomes()
    warning = recwarn.pop()
    assert str(warning.message) == "from pytest_configure"


@pytest.mark.parametrize("tryfirst", [True, False])
def test_pytest_configure_warning_filter(tmp_path: Path, tryfirst: bool) -> None:
    """Issue 10128.

    Parametrize over ``tryfirst`` to guard against hooks that run early
    from avoiding the filterwarnings configuration.
    """

    class ConfigureWarner:
        @pytest.hookimpl(tryfirst=tryfirst)
        def pytest_configure(self):
            warnings.warn("from pytest_configure", UserWarning)

    def test_it():
        pass

    # If the ini filter were not in force around ``pytest_configure`` the
    # warning would escape to this suite, which runs with
    # ``filterwarnings = error``, and blow the run up - the in-process
    # equivalent of the original's "nothing on stdout or stderr".
    record = run_tests(
        test_it,
        spec=ConfigSpec(
            rootpath=tmp_path,
            inicfg={"filterwarnings": ["ignore::UserWarning"]},
            extra_plugins=(ConfigureWarner(),),
        ),
    )
    record.assert_outcomes(passed=1, warnings=0)


# ensemble: every test here needs a real importable plugin module, and the
# warning is caught by ``Config._capture_plugin_import_warnings`` from the
# plugin-loading phase of ``Config.parse`` - an ensemble imports the plugins
# named in its spec itself and never runs that phase.
class TestPluginImportWarning:
    """filterwarnings apply to warnings emitted whilst importing plugins.

    Issue #12697.
    """

    @staticmethod
    def _make_plugin_with_import_warning(pytester: Pytester) -> None:
        pytester.makepyfile(
            warning_plugin="""
                import warnings
                warnings.warn("from plugin import", DeprecationWarning)
            """,
            test_it="def test_it(): pass",
        )

    def test_plugin_import_warning(self, pytester: Pytester) -> None:
        self._make_plugin_with_import_warning(pytester)
        pytester.plugins = ["warning_plugin"]

        result = pytester.runpytest_subprocess()

        result.assert_outcomes(passed=1, warnings=1)
        result.stdout.fnmatch_lines("*DeprecationWarning: from plugin import")

    def test_plugin_import_warning_without_warnings_plugin(
        self,
        pytester: Pytester,
    ) -> None:
        pytester.makeini(
            """
            [pytest]
            filterwarnings =
                error::DeprecationWarning
            """
        )
        self._make_plugin_with_import_warning(pytester)
        pytester.plugins = ["warning_plugin"]

        result = pytester.runpytest_subprocess("-p", "no:warnings")

        result.assert_outcomes(passed=1)
        result.stdout.no_fnmatch_line("*from plugin import*")
        result.stderr.no_fnmatch_line("*from plugin import*")

    def test_plugin_import_warning_with_warnings_plugin_reenabled(
        self,
        pytester: Pytester,
    ) -> None:
        self._make_plugin_with_import_warning(pytester)
        pytester.syspathinsert()

        result = pytester.runpytest(
            "-p", "warning_plugin", "-p", "no:warnings", "-p", "warnings"
        )

        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines("*DeprecationWarning: from plugin import")

    def test_plugin_import_warning_from_pytest_plugins(
        self,
        pytester: Pytester,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._make_plugin_with_import_warning(pytester)
        monkeypatch.setenv("PYTEST_PLUGINS", "warning_plugin")

        result = pytester.runpytest_subprocess()

        result.assert_outcomes(passed=1, warnings=1)
        result.stdout.fnmatch_lines("*DeprecationWarning: from plugin import")


# ensemble: the whole class asserts the file, line and function a warning is
# attributed to. Ensemble sources are anchored in this host file, and the
# conftest loading these exercise has no ensemble equivalent.
class TestStackLevel:
    @pytest.fixture
    def capwarn(self, pytester: Pytester):
        class CapturedWarnings:
            captured: list[
                tuple[warnings.WarningMessage, tuple[str, int, str] | None]
            ] = []

            @classmethod
            def pytest_warning_recorded(cls, warning_message, when, nodeid, location):
                cls.captured.append((warning_message, location))

        pytester.plugins = [CapturedWarnings()]

        return CapturedWarnings

    def test_issue4445_rewrite(self, pytester: Pytester, capwarn) -> None:
        """#4445: Make sure the warning points to a reasonable location
        See origin of _issue_warning_captured at: _pytest.assertion.rewrite.py:241
        """
        pytester.makepyfile(some_mod="")
        conftest = pytester.makeconftest(
            """
                import some_mod
                import pytest

                pytest.register_assert_rewrite("some_mod")
            """
        )
        pytester.parseconfig()

        # with stacklevel=5 the warning originates from register_assert_rewrite
        # function in the created conftest.py
        assert len(capwarn.captured) == 1
        warning, location = capwarn.captured.pop()
        file, lineno, func = location

        assert "Module already imported" in str(warning.message)
        assert file == str(conftest)
        assert func == "<module>"  # the above conftest.py
        assert lineno == 4

    def test_issue4445_initial_conftest(self, pytester: Pytester, capwarn) -> None:
        """#4445: Make sure the warning points to a reasonable location."""
        pytester.makeconftest(
            """
            import nothing
            """
        )
        pytester.parseconfig("--help")

        # with stacklevel=2 the warning should originate from the conftest
        # loading phase of config.parse and is thrown by an erroneous
        # conftest.py
        assert len(capwarn.captured) == 1
        warning, location = capwarn.captured.pop()
        file, _, func = location

        assert "could not load initial conftests" in str(warning.message)
        assert f"config{os.sep}__init__.py" in file
        assert func == "_load_initial_conftests_phase"

    @pytest.mark.filterwarnings("default")
    def test_conftest_warning_captured(self, pytester: Pytester) -> None:
        """Warnings raised during importing of conftest.py files is captured (#2891)."""
        pytester.makeconftest(
            """
            import warnings
            warnings.warn(UserWarning("my custom warning"))
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            ["conftest.py:2", "*UserWarning: my custom warning*"]
        )

    def test_issue4445_import_plugin(self, pytester: Pytester, capwarn) -> None:
        """#4445: Make sure the warning points to a reasonable location"""
        pytester.makepyfile(
            some_plugin="""
            import pytest
            pytest.skip("thing", allow_module_level=True)
            """
        )
        pytester.syspathinsert()
        pytester.parseconfig("-p", "some_plugin")

        # with stacklevel=2 the warning should originate from
        # config.PytestPluginManager.import_plugin is thrown by a skipped plugin

        assert len(capwarn.captured) == 1
        warning, location = capwarn.captured.pop()
        file, _, func = location

        assert "skipped plugin 'some_plugin': thing" in str(warning.message)
        assert f"config{os.sep}__init__.py" in file
        assert func == "_warn_about_skipped_plugins"

    def test_issue4445_issue5928_mark_generator(self, pytester: Pytester) -> None:
        """#4445 and #5928: Make sure the warning from an unknown mark points to
        the test file where this mark is used.
        """
        testfile = pytester.makepyfile(
            """
            import pytest

            @pytest.mark.unknown
            def test_it():
                pass
            """
        )
        result = pytester.runpytest_subprocess()
        # with stacklevel=2 the warning should originate from the above created test file
        result.stdout.fnmatch_lines_random(
            [
                f"*{testfile}:3*",
                "*Unknown pytest.mark.unknown*",
            ]
        )


# ensemble: the warning comes from the ``testpaths`` glob expansion in
# ``Config._decide_args``, which an ensemble skips - its args are taken
# verbatim and its collection never walks the filesystem.
def test_warning_on_testpaths_not_found(pytester: Pytester) -> None:
    # Check for warning when testpaths set, but not found by glob
    pytester.makeini(
        """
        [pytest]
        testpaths = absent
        """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        ["*ConfigWarning: No files were found in testpaths*", "*1 warning*"]
    )


# ensemble: needs a fresh interpreter started with ``-Xdev`` and a
# PYTHONTRACEMALLOC environment.
def test_resource_warning(pytester: Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    # Some platforms (notably PyPy) don't have tracemalloc.
    # We choose to explicitly not skip this in case tracemalloc is not
    # available, using `importorskip("tracemalloc")` for example,
    # because we want to ensure the same code path does not break in those platforms.
    try:
        import tracemalloc  # noqa: F401

        has_tracemalloc = True
    except ImportError:
        has_tracemalloc = False

    # Explicitly disable PYTHONTRACEMALLOC in case pytest's test suite is running
    # with it enabled.
    monkeypatch.delenv("PYTHONTRACEMALLOC", raising=False)

    pytester.makepyfile(
        """
        def open_file(p):
            f = p.open("r", encoding="utf-8")
            assert p.read_text() == "hello"

        def test_resource_warning(tmp_path):
            p = tmp_path.joinpath("foo.txt")
            p.write_text("hello", encoding="utf-8")
            open_file(p)
        """
    )
    result = pytester.run(sys.executable, "-Xdev", "-m", "pytest")
    expected_extra = (
        [
            "*ResourceWarning* unclosed file*",
            "*Enable tracemalloc to get traceback where the object was allocated*",
            "*See https* for more info.",
        ]
        if has_tracemalloc
        else []
    )
    result.stdout.fnmatch_lines([*expected_extra, "*1 passed*"])

    monkeypatch.setenv("PYTHONTRACEMALLOC", "20")

    result = pytester.run(sys.executable, "-Xdev", "-m", "pytest")
    expected_extra = (
        [
            "*ResourceWarning* unclosed file*",
            "*Object allocated at*",
        ]
        if has_tracemalloc
        else []
    )
    result.stdout.fnmatch_lines([*expected_extra, "*1 passed*"])


def run_for_exitstatus(
    *sources: Source,
    spec: ConfigSpec,
) -> tuple[RunRecord, int | ExitCode]:
    """Run an ensemble and make the session-level exit status decision.

    ``--max-warnings`` is enforced by the terminal reporter in
    ``pytest_sessionfinish``, which promotes ``session.exitstatus`` from
    ``OK`` to ``MAX_WARNINGS_ERROR``. That attribute is normally set by
    ``_pytest.main.wrap_session``, which an ensemble does not run, so the
    same OK/TESTS_FAILED decision is made explicitly here - the promotion
    itself, which is what these tests are about, is left to pytest.
    """
    with Ensemble(*sources, spec=spec, capture_output=True) as ensemble:
        record = ensemble.run()
        session = ensemble.session
        session.exitstatus = (
            ExitCode.TESTS_FAILED if session.testsfailed else ExitCode.OK
        )
    # The summary line is only written on the way out of the block.
    return dataclasses.replace(record, output=ensemble.output), session.exitstatus


class TestMaxWarnings:
    """Tests for the --max-warnings feature."""

    @staticmethod
    def sources() -> tuple[Source, Source]:
        """The two warning-emitting tests of the original ``PYFILE``."""

        def test_one():
            warnings.warn(UserWarning("warning one"))

        def test_two():
            warnings.warn(UserWarning("warning two"))

        return test_one, test_two

    @staticmethod
    def spec(
        tmp_path: Path,
        *,
        args: tuple[str, ...] = (),
        inicfg: dict[str, object] | None = None,
    ) -> ConfigSpec:
        """A spec showing UserWarnings, as the enclosing marks did."""
        return ConfigSpec(
            rootpath=tmp_path,
            args=args,
            inicfg={"filterwarnings": ["default::UserWarning"], **(inicfg or {})},
        )

    def test_max_warnings_not_set(self, tmp_path: Path) -> None:
        """Without --max-warnings, warnings don't affect exit code."""
        record, exitstatus = run_for_exitstatus(
            *self.sources(), spec=self.spec(tmp_path)
        )
        record.assert_outcomes(passed=2, warnings=2)
        assert exitstatus == ExitCode.OK

    def test_max_warnings_not_exceeded(self, tmp_path: Path) -> None:
        """When warning count is below the threshold, exit code is OK."""
        record, exitstatus = run_for_exitstatus(
            *self.sources(), spec=self.spec(tmp_path, args=("--max-warnings", "10"))
        )
        record.assert_outcomes(passed=2, warnings=2)
        assert exitstatus == ExitCode.OK

    def test_max_warnings_exceeded(self, tmp_path: Path) -> None:
        """When warning count exceeds threshold, exit code is MAX_WARNINGS_ERROR."""
        _, exitstatus = run_for_exitstatus(
            *self.sources(), spec=self.spec(tmp_path, args=("--max-warnings", "1"))
        )
        assert exitstatus == ExitCode.MAX_WARNINGS_ERROR

    def test_max_warnings_equal_to_count(self, tmp_path: Path) -> None:
        """When warning count equals threshold exactly, exit code is OK."""
        record, exitstatus = run_for_exitstatus(
            *self.sources(), spec=self.spec(tmp_path, args=("--max-warnings", "2"))
        )
        record.assert_outcomes(passed=2, warnings=2)
        assert exitstatus == ExitCode.OK

    def test_max_warnings_zero(self, tmp_path: Path) -> None:
        """--max-warnings 0 means no warnings are allowed."""
        _, exitstatus = run_for_exitstatus(
            *self.sources(), spec=self.spec(tmp_path, args=("--max-warnings", "0"))
        )
        assert exitstatus == ExitCode.MAX_WARNINGS_ERROR

    def test_max_warnings_exceeded_message(self, tmp_path: Path) -> None:
        """Verify the output message when max warnings is exceeded."""
        record, _ = run_for_exitstatus(
            *self.sources(), spec=self.spec(tmp_path, args=("--max-warnings", "1"))
        )
        record.stdout.fnmatch_lines(
            ["*Tests pass, but maximum allowed warnings exceeded: 2 > 1*"]
        )

    def test_max_warnings_ini_option(self, tmp_path: Path) -> None:
        """max_warnings can be set via INI configuration."""
        _, exitstatus = run_for_exitstatus(
            *self.sources(), spec=self.spec(tmp_path, inicfg={"max_warnings": "1"})
        )
        assert exitstatus == ExitCode.MAX_WARNINGS_ERROR

    def test_max_warnings_with_test_failure(self, tmp_path: Path) -> None:
        """When tests fail AND warnings exceed max, TESTS_FAILED takes priority."""

        def test_fail():
            warnings.warn(UserWarning("a warning"))
            raise AssertionError

        record, exitstatus = run_for_exitstatus(
            test_fail, spec=self.spec(tmp_path, args=("--max-warnings", "0"))
        )
        record.assert_outcomes(failed=1, warnings=1)
        assert exitstatus == ExitCode.TESTS_FAILED

    def test_max_warnings_with_filterwarnings_ignore(self, tmp_path: Path) -> None:
        """Filtered (ignored) warnings don't count toward max_warnings."""

        def test_one():
            warnings.warn(UserWarning("counted"))
            warnings.warn(RuntimeWarning("ignored"))

        record, exitstatus = run_for_exitstatus(
            test_one,
            spec=self.spec(
                tmp_path,
                args=("--max-warnings", "1", "-W", "ignore::RuntimeWarning"),
            ),
        )
        record.assert_outcomes(passed=1, warnings=1)
        assert exitstatus == ExitCode.OK

    def test_max_warnings_with_filterwarnings_error(self, tmp_path: Path) -> None:
        """Warnings turned into errors via filterwarnings don't count as warnings."""

        def test_one():
            warnings.warn(UserWarning("still a warning"))

        def test_two():
            warnings.warn(RuntimeWarning("becomes an error"))

        record, exitstatus = run_for_exitstatus(
            test_one,
            test_two,
            spec=self.spec(
                tmp_path,
                args=("--max-warnings", "0", "-W", "error::RuntimeWarning"),
            ),
        )
        record.assert_outcomes(passed=1, failed=1, warnings=1)
        # The RuntimeWarning becomes a test error, so TESTS_FAILED takes priority.
        assert exitstatus == ExitCode.TESTS_FAILED

    def test_max_warnings_with_filterwarnings_ini_ignore(self, tmp_path: Path) -> None:
        """Warnings ignored via ini filterwarnings don't count toward max_warnings."""

        def test_one():
            warnings.warn(UserWarning("counted"))
            warnings.warn(RuntimeWarning("ignored by ini"))

        record, exitstatus = run_for_exitstatus(
            test_one,
            spec=self.spec(
                tmp_path,
                inicfg={
                    "filterwarnings": [
                        "default::UserWarning",
                        "ignore::RuntimeWarning",
                    ],
                    "max_warnings": "1",
                },
            ),
        )
        record.assert_outcomes(passed=1, warnings=1)
        assert exitstatus == ExitCode.OK


# ensemble: the duplication regressed on is produced by ``Config.parse``
# invoking the argument parser several times over the same args; an ensemble
# builds its namespace with a single ``parse_known_args`` call, so the same
# assertion there could not fail.
def test_pythonwarnings_not_duplicated(pytester: Pytester) -> None:
    """Regression test for #13484: -W values should not be duplicated in
    known_args_namespace due to the arg parser being called multiple times."""
    config = pytester.parseconfig("-W", "error")
    warnings_list = config.known_args_namespace.pythonwarnings
    assert warnings_list is not None
    assert warnings_list == ["error"]

# mypy: allow-untyped-defs
"""Terminal reporting of the full testing process."""

from __future__ import annotations

from io import StringIO
import os
from pathlib import Path
import sys
import textwrap
from types import ModuleType
from types import SimpleNamespace
from typing import cast
from typing import Literal
from typing import NamedTuple
from unittest import mock

import pluggy

from _pytest._io.wcwidth import wcswidth
import _pytest.config
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.ensemble import build_module
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import Ensemble
from _pytest.ensemble import run_tests
from _pytest.ensemble import RunRecord
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester
from _pytest.reports import BaseReport
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
import _pytest.terminal
from _pytest.terminal import _folded_skips
from _pytest.terminal import _format_trimmed
from _pytest.terminal import _get_line_with_reprcrash_message
from _pytest.terminal import _get_raw_skip_reason
from _pytest.terminal import _plugin_nameversions
from _pytest.terminal import getreportopt
from _pytest.terminal import TerminalProgressPlugin
from _pytest.terminal import TerminalReporter
import pytest


def unregistered_mark(name: str, *args: object, **kwargs: object) -> MarkDecorator:
    """Build a mark decorator without consulting the host configuration.

    ``pytest.mark.<name>`` resolves against the *host* config at decoration
    time, and this suite runs with strict markers, so a deliberately
    unregistered mark applied to an ensemble source has to be built directly.
    """
    return MarkDecorator(Mark(name, args, kwargs, _ispytest=True), _ispytest=True)


class DistInfo(NamedTuple):
    project_name: str
    version: int


TRANS_FNMATCH = str.maketrans({"[": "[[]", "]": "[]]"})


class Option:
    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    @property
    def args(self):
        values = []
        values.append(f"--verbosity={self.verbosity}")
        return values


@pytest.fixture(
    params=[Option(verbosity=0), Option(verbosity=1), Option(verbosity=-1)],
    ids=["default", "verbose", "quiet"],
)
def option(request):
    return request.param


@pytest.mark.parametrize(
    "input,expected",
    [
        ([DistInfo(project_name="test", version=1)], ["test-1"]),
        ([DistInfo(project_name="pytest-test", version=1)], ["test-1"]),
        (
            [
                DistInfo(project_name="test", version=1),
                DistInfo(project_name="test", version=1),
            ],
            ["test-1"],
        ),
    ],
    ids=["normal", "prefix-strip", "deduplicate"],
)
def test_plugin_nameversion(input, expected):
    pluginlist = [(None, x) for x in input]
    result = _plugin_nameversions(pluginlist)
    assert result == expected


class TestTerminal:
    def test_pass_skip_fail(self, tmp_path: Path, option) -> None:
        def test_ok():
            pass

        def test_skip():
            pytest.skip("xx")

        def test_func():
            assert 0

        spec = ConfigSpec(rootpath=tmp_path, args=tuple(option.args))
        record = run_tests(
            test_ok,
            test_skip,
            test_func,
            spec=spec,
            name="test_pass_skip_fail",
            capture_output=True,
        )
        record.assert_outcomes(passed=1, skipped=1, failed=1)
        if option.verbosity > 0:
            record.stdout.fnmatch_lines(
                [
                    "*test_pass_skip_fail.py::test_ok PASS*",
                    "*test_pass_skip_fail.py::test_skip SKIP*",
                    "*test_pass_skip_fail.py::test_func FAIL*",
                ]
            )
        elif option.verbosity == 0:
            record.stdout.fnmatch_lines(["*test_pass_skip_fail.py .sF*"])
        else:
            record.stdout.fnmatch_lines([".sF*"])
        record.stdout.fnmatch_lines(
            ["    def test_func():", ">       assert 0", "E       assert 0"]
        )

    # ensemble: `console_output_style=times` needs the progress column, which
    # only appears once capturing is active and the run goes through
    # `pytest_runtestloop` - an ensemble has neither, so the code path this
    # regression test exercises would never run.
    def test_console_output_style_times_with_skipped_and_passed(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            test_repro="""
                def test_hello():
                    pass
            """,
            test_repro_skip="""
                import pytest
                pytest.importorskip("fakepackage_does_not_exist")
            """,
        )
        result = pytester.runpytest(
            "test_repro.py",
            "test_repro_skip.py",
            "-o",
            "console_output_style=times",
        )

        result.stdout.fnmatch_lines("* 1 passed, 1 skipped in *")

        combined = "\n".join(result.stdout.lines + result.stderr.lines)
        assert "INTERNALERROR" not in combined

    def test_internalerror(self, tmp_path: Path, linecomp) -> None:
        def test_one():
            pass

        with Ensemble(test_one, rootpath=tmp_path, capture_output=True) as ensemble:
            rep = TerminalReporter(ensemble.config, file=linecomp.stringio)
            with pytest.raises(ValueError) as excinfo:
                raise ValueError("hello")
            rep.pytest_internalerror(excinfo.getrepr())
        linecomp.assert_contains_lines(["INTERNALERROR> *ValueError*hello*"])

    def test_writeline(self, tmp_path: Path, linecomp) -> None:
        def test_one():
            pass

        with Ensemble(
            test_one, rootpath=tmp_path, name="test_writeline", capture_output=True
        ) as ensemble:
            (item,) = ensemble.collect()
            modcol = item.parent
            assert modcol is not None
            rep = TerminalReporter(ensemble.config, file=linecomp.stringio)
            rep.write_fspath_result(modcol.nodeid, ".")
            rep.write_line("hello world")
            lines = linecomp.stringio.getvalue().split("\n")
            assert not lines[0]
            assert lines[1].endswith(modcol.name + " .")
            assert lines[2] == "hello world"

    def test_show_runtest_logstart(self, tmp_path: Path, linecomp) -> None:
        def test_func():
            pass

        with Ensemble(
            test_func,
            rootpath=tmp_path,
            name="test_show_runtest_logstart",
            capture_output=True,
        ) as ensemble:
            (item,) = ensemble.collect()
            tr = TerminalReporter(item.config, file=linecomp.stringio)
            item.config.pluginmanager.register(tr)
            location = item.reportinfo()
            tr.config.hook.pytest_runtest_logstart(
                nodeid=item.nodeid, location=location, fspath=str(item.path)
            )
            item.config.pluginmanager.unregister(tr)
        linecomp.assert_contains_lines(["*test_show_runtest_logstart.py*"])

    # ensemble: drives a real pytest through a pty to watch output appear
    # before the test finishes.
    def test_runtest_location_shown_before_test_starts(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            """
            def test_1():
                import time
                time.sleep(20)
        """
        )
        child = pytester.spawn_pytest("")
        child.expect(".*test_runtest_location.*py")
        child.sendeof()
        child.kill(15)

    # ensemble: drives a real pytest through a pty.
    def test_report_collect_after_half_a_second(
        self, pytester: Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        """Test for "collecting" being updated after 0.5s"""
        pytester.makepyfile(
            **{
                "test1.py": """
                import _pytest.terminal

                _pytest.terminal.REPORT_COLLECTING_RESOLUTION = 0

                def test_1():
                    pass
                    """,
                "test2.py": "def test_2(): pass",
            }
        )
        # Explicitly test colored output.
        monkeypatch.setenv("PY_COLORS", "1")

        child = pytester.spawn_pytest("-v test1.py test2.py")
        child.expect(r"collecting \.\.\.")
        child.expect(r"collecting 1 item")
        child.expect(r"collecting 2 items")
        child.expect(r"collected 2 items")
        rest = child.read().decode("utf8")
        assert "= \x1b[32m\x1b[1m2 passed\x1b[0m\x1b[32m in" in rest

    # ensemble: asserts the `<- test_p1.py` suffix, which is derived from the
    # item's reportinfo. Ensemble items report the *host* file, so every
    # ensemble item would carry that suffix at -vv and none of these lines
    # would match.
    def test_itemreport_subclasses_show_subclassed_file(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            **{
                "tests/test_p1": """
            class BaseTests(object):
                fail = False

                def test_p1(self):
                    if self.fail: assert 0
                """,
                "tests/test_p2": """
            from test_p1 import BaseTests

            class TestMore(BaseTests): pass
                """,
                "tests/test_p3.py": """
            from test_p1 import BaseTests

            BaseTests.fail = True

            class TestMore(BaseTests): pass
        """,
            }
        )
        result = pytester.runpytest("tests/test_p2.py", "--rootdir=tests")
        result.stdout.fnmatch_lines(["tests/test_p2.py .*", "=* 1 passed in *"])

        result = pytester.runpytest("-vv", "-rA", "tests/test_p2.py", "--rootdir=tests")
        result.stdout.fnmatch_lines(
            [
                "tests/test_p2.py::TestMore::test_p1 <- test_p1.py PASSED *",
                "*= short test summary info =*",
                "PASSED tests/test_p2.py::TestMore::test_p1",
            ]
        )
        result = pytester.runpytest("-vv", "-rA", "tests/test_p3.py", "--rootdir=tests")
        result.stdout.fnmatch_lines(
            [
                "tests/test_p3.py::TestMore::test_p1 <- test_p1.py FAILED *",
                "*_ TestMore.test_p1 _*",
                "    def test_p1(self):",
                ">       if self.fail: assert 0",
                "E       assert 0",
                "",
                "tests/test_p1.py:5: AssertionError",
                "*= short test summary info =*",
                "FAILED tests/test_p3.py::TestMore::test_p1 - assert 0",
                "*= 1 failed in *",
            ]
        )

    # ensemble: asserts `no_fnmatch_line("* <- *")`, which an ensemble can
    # never satisfy - the item's reportinfo is the host file, so -vv always
    # renders a `<- .../test_terminal.py` suffix.
    def test_itemreport_directclasses_not_shown_as_subclasses(
        self, pytester: Pytester
    ) -> None:
        a = pytester.mkpydir("a123")
        a.joinpath("test_hello123.py").write_text(
            textwrap.dedent(
                """\
                class TestClass(object):
                    def test_method(self):
                        pass
                """
            ),
            encoding="utf-8",
        )
        result = pytester.runpytest("-vv")
        assert result.ret == 0
        result.stdout.fnmatch_lines(["*a123/test_hello123.py*PASS*"])
        result.stdout.no_fnmatch_line("* <- *")

    # ensemble: KeyboardInterrupt reporting is done by `wrap_session`, which
    # an ensemble never enters.
    @pytest.mark.parametrize("fulltrace", ("", "--fulltrace"))
    def test_keyboard_interrupt(self, pytester: Pytester, fulltrace) -> None:
        pytester.makepyfile(
            """
            def test_foobar():
                assert 0
            def test_spamegg():
                import py; pytest.skip('skip me please!')
            def test_interrupt_me():
                raise KeyboardInterrupt   # simulating the user
        """
        )

        result = pytester.runpytest(fulltrace, no_reraise_ctrlc=True)
        result.stdout.fnmatch_lines(
            [
                "    def test_foobar():",
                ">       assert 0",
                "E       assert 0",
                "*_keyboard_interrupt.py:6: KeyboardInterrupt*",
            ]
        )
        if fulltrace:
            result.stdout.fnmatch_lines(
                ["*raise KeyboardInterrupt   # simulating the user*"]
            )
        else:
            result.stdout.fnmatch_lines(
                ["(to show a full traceback on KeyboardInterrupt use --full-trace)"]
            )
        result.stdout.fnmatch_lines(["*KeyboardInterrupt*"])

    # ensemble: asserts the exit code of an interrupted session; interrupt
    # handling lives in `wrap_session`.
    def test_keyboard_in_sessionstart(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            def pytest_sessionstart():
                raise KeyboardInterrupt
        """
        )
        pytester.makepyfile(
            """
            def test_foobar():
                pass
        """
        )

        result = pytester.runpytest(no_reraise_ctrlc=True)
        assert result.ret == 2
        result.stdout.fnmatch_lines(["*KeyboardInterrupt*"])

    def test_collect_single_item(self, tmp_path: Path) -> None:
        """Use singular 'item' when reporting a single test item"""

        def test_foobar():
            pass

        record = run_tests(test_foobar, rootpath=tmp_path, capture_output=True)
        record.assert_outcomes(passed=1)
        record.stdout.fnmatch_lines(["collected 1 item"])

    def test_rewrite(self, tmp_path: Path, monkeypatch) -> None:
        with Ensemble(rootpath=tmp_path, capture_output=True) as ensemble:
            config = ensemble.config
            f = StringIO()
            monkeypatch.setattr(f, "isatty", lambda *args: True)
            tr = TerminalReporter(config, f)
            tr._tw.fullwidth = 10
            tr.write("hello")
            tr.rewrite("hey", erase=True)
            assert f.getvalue() == "hello" + "\r" + "hey" + (6 * " ")

    # ensemble: `assert not result.stderr.lines` has no in-process equivalent -
    # an ensemble renders to a single buffer and has no stderr of its own - so
    # porting would drop half of what this test checks.
    @pytest.mark.parametrize("category", ["foo", "failed", "error", "passed"])
    def test_report_teststatus_explicit_markup(
        self, monkeypatch: MonkeyPatch, pytester: Pytester, color_mapping, category: str
    ) -> None:
        """Test that TerminalReporter handles markup explicitly provided by
        a pytest_report_teststatus hook."""
        monkeypatch.setenv("PY_COLORS", "1")
        pytester.makeconftest(
            f"""
            def pytest_report_teststatus(report):
                return {category!r}, 'F', ('FOO', {{'red': True}})
        """
        )
        pytester.makepyfile(
            """
            def test_foobar():
                pass
        """
        )

        result = pytester.runpytest("-v")
        assert not result.stderr.lines
        result.stdout.fnmatch_lines(
            color_mapping.format_for_fnmatch(["*{red}FOO{reset}*"])
        )

    # ensemble: the -vv half asserts on lines that, for an ensemble item, gain
    # a `<- .../test_terminal.py` suffix from the host-anchored reportinfo,
    # which also shifts where the skip reason gets wrapped.
    def test_verbose_skip_reason(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.skip(reason="123")
            def test_1():
                pass

            @pytest.mark.xfail(reason="456")
            def test_2():
                pass

            @pytest.mark.xfail(reason="789")
            def test_3():
                assert False

            @pytest.mark.xfail(reason="")
            def test_4():
                assert False

            @pytest.mark.skip
            def test_5():
                pass

            @pytest.mark.xfail
            def test_6():
                pass

            def test_7():
                pytest.skip()

            def test_8():
                pytest.skip("888 is great")

            def test_9():
                pytest.xfail()

            def test_10():
                pytest.xfail("It's 🕙 o'clock")

            @pytest.mark.skip(
                reason="1 cannot do foobar because baz is missing due to I don't know what"
            )
            def test_long_skip():
                pass

            @pytest.mark.xfail(
                reason="2 cannot do foobar because baz is missing due to I don't know what"
            )
            def test_long_xfail():
                print(1 / 0)
        """
        )

        common_output = [
            "test_verbose_skip_reason.py::test_1 SKIPPED (123) *",
            "test_verbose_skip_reason.py::test_2 XPASS (456) *",
            "test_verbose_skip_reason.py::test_3 XFAIL (789) *",
            "test_verbose_skip_reason.py::test_4 XFAIL  *",
            "test_verbose_skip_reason.py::test_5 SKIPPED (unconditional skip) *",
            "test_verbose_skip_reason.py::test_6 XPASS  *",
            "test_verbose_skip_reason.py::test_7 SKIPPED  *",
            "test_verbose_skip_reason.py::test_8 SKIPPED (888 is great) *",
            "test_verbose_skip_reason.py::test_9 XFAIL  *",
            "test_verbose_skip_reason.py::test_10 XFAIL (It's 🕙 o'clock) *",
        ]

        result = pytester.runpytest("-v")
        result.stdout.fnmatch_lines(
            [
                *common_output,
                "test_verbose_skip_reason.py::test_long_skip SKIPPED (1 cannot *...) *",
                "test_verbose_skip_reason.py::test_long_xfail XFAIL (2 cannot *...) *",
            ]
        )

        result = pytester.runpytest("-vv")
        result.stdout.fnmatch_lines(
            [
                *common_output,
                "test_verbose_skip_reason.py::test_long_skip SKIPPED"
                " (1 cannot do foobar",
                "because baz is missing due to I don't know what) *",
                "test_verbose_skip_reason.py::test_long_xfail XFAIL"
                " (2 cannot do foobar",
                "because baz is missing due to I don't know what) *",
            ]
        )

    @pytest.mark.parametrize("isatty", [True, False])
    def test_isatty(self, tmp_path: Path, monkeypatch, isatty: bool) -> None:
        with Ensemble(rootpath=tmp_path, capture_output=True) as ensemble:
            config = ensemble.config
            f = StringIO()
            monkeypatch.setattr(f, "isatty", lambda: isatty)
            tr = TerminalReporter(config, f)
            assert tr.isatty() == isatty
            # It was incorrectly implemented as a boolean so we still support using it as one.
            assert bool(tr.isatty) == isatty


# ensemble: every test below asserts the rendering of `--collect-only`, which
# is served from `pytest_cmdline_main`. An ensemble never reaches that path:
# it still runs the collected items, and its collection tree renders as
# `<EnsembleModule ...>` with no enclosing `<Dir ...>`.
class TestCollectonly:
    def test_collectonly_basic(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            def test_func():
                pass
        """
        )
        result = pytester.runpytest("--collect-only")
        result.stdout.fnmatch_lines(
            [
                "<Dir test_collectonly_basic0>",
                "  <Module test_collectonly_basic.py>",
                "    <Function test_func>",
            ]
        )

    def test_collectonly_skipped_module(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            pytest.skip("hello")
        """
        )
        result = pytester.runpytest("--collect-only", "-rs")
        result.stdout.fnmatch_lines(["*ERROR collecting*"])

    def test_collectonly_displays_test_description(
        self, pytester: Pytester, dummy_yaml_custom_test
    ) -> None:
        """Used dummy_yaml_custom_test for an Item without ``obj``."""
        pytester.makepyfile(
            """
            def test_with_description():
                '''  This test has a description.

                  more1.
                    more2.'''
            """
        )
        result = pytester.runpytest("--collect-only", "--verbose")
        result.stdout.fnmatch_lines(
            [
                "<Dir test_collectonly_displays_test_description0>",
                "  <YamlFile test1.yaml>",
                "    <YamlItem test1.yaml>",
                "  <Module test_collectonly_displays_test_description.py>",
                "    <Function test_with_description>",
                "      This test has a description.",
                "      ",
                "      more1.",
                "        more2.",
            ],
            consecutive=True,
        )

    def test_collectonly_failed_module(self, pytester: Pytester) -> None:
        pytester.makepyfile("""raise ValueError(0)""")
        result = pytester.runpytest("--collect-only")
        result.stdout.fnmatch_lines(["*raise ValueError*", "*1 error*"])

    def test_collectonly_fatal(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            def pytest_collectstart(collector):
                assert 0, "urgs"
        """
        )
        result = pytester.runpytest("--collect-only")
        result.stdout.fnmatch_lines(["*INTERNAL*args*"])
        assert result.ret == 3

    def test_collectonly_simple(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            """
            def test_func1():
                pass
            class TestClass(object):
                def test_method(self):
                    pass
        """
        )
        result = pytester.runpytest("--collect-only", p)
        # assert stderr.startswith("inserting into sys.path")
        assert result.ret == 0
        result.stdout.fnmatch_lines(
            [
                "*<Module *.py>",
                "* <Function test_func1>",
                "* <Class TestClass>",
                "*   <Function test_method>",
            ]
        )

    def test_collectonly_error(self, pytester: Pytester) -> None:
        p = pytester.makepyfile("import Errlkjqweqwe")
        result = pytester.runpytest("--collect-only", p)
        assert result.ret == 2
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                *ERROR*
                *ImportError*
                *No module named *Errlk*
                *1 error*
                """
            ).strip()
        )

    def test_collectonly_missing_path(self, pytester: Pytester) -> None:
        """Issue 115: failure in parseargs will cause session not to
        have the items attribute."""
        result = pytester.runpytest("--collect-only", "uhm_missing_path")
        assert result.ret == 4
        result.stderr.fnmatch_lines(
            ["*ERROR: file or directory not found: uhm_missing_path"]
        )

    def test_collectonly_quiet(self, pytester: Pytester) -> None:
        pytester.makepyfile("def test_foo(): pass")
        result = pytester.runpytest("--collect-only", "-q")
        result.stdout.fnmatch_lines(["*test_foo*"])

    def test_collectonly_more_quiet(self, pytester: Pytester) -> None:
        pytester.makepyfile(test_fun="def test_foo(): pass")
        result = pytester.runpytest("--collect-only", "-qq")
        result.stdout.fnmatch_lines(["*test_fun.py: 1*"])

    def test_collect_only_summary_status(self, pytester: Pytester) -> None:
        """Custom status depending on test selection using -k or -m. #7701."""
        pytester.makepyfile(
            test_collect_foo="""
            def test_foo(): pass
            """,
            test_collect_bar="""
            def test_foobar(): pass
            def test_bar(): pass
            """,
        )
        result = pytester.runpytest("--collect-only")
        result.stdout.fnmatch_lines("*== 3 tests collected in * ==*")

        result = pytester.runpytest("--collect-only", "test_collect_foo.py")
        result.stdout.fnmatch_lines("*== 1 test collected in * ==*")

        result = pytester.runpytest("--collect-only", "-k", "foo")
        result.stdout.fnmatch_lines("*== 2/3 tests collected (1 deselected) in * ==*")

        result = pytester.runpytest("--collect-only", "-k", "test_bar")
        result.stdout.fnmatch_lines("*== 1/3 tests collected (2 deselected) in * ==*")

        result = pytester.runpytest("--collect-only", "-k", "invalid")
        result.stdout.fnmatch_lines("*== no tests collected (3 deselected) in * ==*")

        pytester.mkdir("no_tests_here")
        result = pytester.runpytest("--collect-only", "no_tests_here")
        result.stdout.fnmatch_lines("*== no tests collected in * ==*")

        pytester.makepyfile(
            test_contains_error="""
            raise RuntimeError
            """,
        )
        result = pytester.runpytest("--collect-only")
        result.stdout.fnmatch_lines("*== 3 tests collected, 1 error in * ==*")
        result = pytester.runpytest("--collect-only", "-k", "foo")
        result.stdout.fnmatch_lines(
            "*== 2/3 tests collected (1 deselected), 1 error in * ==*"
        )


# ensemble: every test below asserts on a `Captured stdout` section. Capture
# is process-global state an ensemble deliberately does not install, so no
# captured-output section is ever rendered.
class TestFixtureReporting:
    def test_setup_fixture_error(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            def setup_function(function):
                print("setup func")
                assert 0
            def test_nada():
                pass
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ERROR at setup of test_nada*",
                "*setup_function(function):*",
                "*setup func*",
                "*assert 0*",
                "*1 error*",
            ]
        )
        assert result.ret != 0

    def test_teardown_fixture_error(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            def test_nada():
                pass
            def teardown_function(function):
                print("teardown func")
                assert 0
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ERROR at teardown*",
                "*teardown_function(function):*",
                "*assert 0*",
                "*Captured stdout*",
                "*teardown func*",
                "*1 passed*1 error*",
            ]
        )

    def test_teardown_fixture_error_and_test_failure(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            def test_fail():
                assert 0, "failingfunc"

            def teardown_function(function):
                print("teardown func")
                assert False
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ERROR at teardown of test_fail*",
                "*teardown_function(function):*",
                "*assert False*",
                "*Captured stdout*",
                "*teardown func*",
                "*test_fail*",
                "*def test_fail():",
                "*failingfunc*",
                "*1 failed*1 error*",
            ]
        )

    def test_setup_teardown_output_and_test_failure(self, pytester: Pytester) -> None:
        """Test for issue #442."""
        pytester.makepyfile(
            """
            def setup_function(function):
                print("setup func")

            def test_fail():
                assert 0, "failingfunc"

            def teardown_function(function):
                print("teardown func")
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*test_fail*",
                "*def test_fail():",
                "*failingfunc*",
                "*Captured stdout setup*",
                "*setup func*",
                "*Captured stdout teardown*",
                "*teardown func*",
                "*1 failed*",
            ]
        )


class TestTerminalFunctional:
    def test_deselected(self, tmp_path: Path) -> None:
        def test_one():
            pass

        def test_two():
            pass

        def test_three():
            pass

        spec = ConfigSpec(rootpath=tmp_path, args=("-k", "test_t"))
        record = run_tests(
            test_one,
            test_two,
            test_three,
            spec=spec,
            name="test_deselected",
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            ["collected 3 items / 1 deselected / 2 selected", "*test_deselected.py ..*"]
        )
        # Stronger than the original `ret == 0`.
        record.assert_outcomes(passed=2, deselected=1)

    def test_deselected_with_hook_wrapper(self, tmp_path: Path) -> None:
        class DeselectLastPlugin:
            @pytest.hookimpl(wrapper=True)
            def pytest_collection_modifyitems(self, config, items):
                yield
                deselected = items.pop()
                config.hook.pytest_deselected(items=[deselected])

        def test_one():
            pass

        def test_two():
            pass

        def test_three():
            pass

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(DeselectLastPlugin(),))
        record = run_tests(
            test_one,
            test_two,
            test_three,
            spec=spec,
            name="test_deselected_hook",
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "collected 3 items / 1 deselected / 2 selected",
                "*= 2 passed, 1 deselected in*",
            ]
        )
        record.assert_outcomes(passed=2, deselected=1)

    def test_show_deselected_items_using_markexpr_before_test_execution(
        self, tmp_path: Path
    ) -> None:
        @unregistered_mark("foo")
        def test_foobar():
            pass

        @unregistered_mark("bar")
        def test_bar():
            pass

        def test_pass():
            pass

        spec = ConfigSpec(rootpath=tmp_path, args=("-m", "not foo"))
        record = run_tests(
            test_foobar,
            test_bar,
            test_pass,
            spec=spec,
            name="test_show_deselected",
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "collected 3 items / 1 deselected / 2 selected",
                "*test_show_deselected.py ..*",
                "*= 2 passed, 1 deselected in * =*",
            ]
        )
        record.stdout.no_fnmatch_line("*= 1 deselected =*")
        record.assert_outcomes(passed=2, deselected=1)

    # ensemble: the `! Interrupted: ... !` line and the interrupted exit code
    # come from `wrap_session`, and the collection error needs a module that
    # blows up on import.
    def test_selected_count_with_error(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_selected_count_3="""
                def test_one():
                    pass
                def test_two():
                    pass
                def test_three():
                    pass
            """,
            test_selected_count_error="""
                5/0
                def test_foo():
                    pass
                def test_bar():
                    pass
            """,
        )
        result = pytester.runpytest("-k", "test_t")
        result.stdout.fnmatch_lines(
            [
                "collected 3 items / 1 error / 1 deselected / 2 selected",
                "* ERROR collecting test_selected_count_error.py *",
            ]
        )
        assert result.ret == ExitCode.INTERRUPTED

    def test_no_skip_summary_if_failure(self, tmp_path: Path) -> None:
        def test_ok():
            pass

        def test_fail():
            assert 0

        def test_skip():
            pytest.skip("dontshow")

        record = run_tests(
            test_ok,
            test_fail,
            test_skip,
            rootpath=tmp_path,
            name="test_no_skip_summary_if_failure",
            capture_output=True,
        )
        assert record.output.find("skip test summary") == -1
        # Stronger than the original `ret == 1`.
        record.assert_outcomes(passed=1, failed=1, skipped=1)

    def test_passes(self, tmp_path: Path) -> None:
        def test_passes():
            pass

        class TestClass:
            def test_method(self):
                pass

        record = run_tests(
            test_passes,
            TestClass,
            rootpath=tmp_path,
            name="test_passes",
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["test_passes.py ..*", "* 2 pass*"])
        record.assert_outcomes(passed=2)

    # ensemble: the header block differs - an ensemble has no `plugins:` line
    # and its rootdir is a throwaway tmp path.
    def test_header_trailer_info(
        self, monkeypatch: MonkeyPatch, pytester: Pytester, request
    ) -> None:
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
        monkeypatch.delenv("PYTEST_PLUGINS", raising=False)
        pytester.makepyfile(
            """
            def test_passes():
                pass
        """
        )
        result = pytester.runpytest()
        verinfo = ".".join(map(str, sys.version_info[:3]))
        result.stdout.fnmatch_lines(
            [
                "*===== test session starts ====*",
                f"platform {sys.platform} -- Python {verinfo}*pytest-{pytest.__version__}**pluggy-{pluggy.__version__}",
                "*test_header_trailer_info.py .*",
                "=* 1 passed*in *.[0-9][0-9]s *=",
            ]
        )
        if request.config.pluginmanager.list_plugin_distinfo():
            result.stdout.fnmatch_lines(["plugins: *"])

    # ensemble: an ensemble never renders a `plugins:` line, so the
    # `no_fnmatch_line("plugins: *")` half would hold for the wrong reason.
    def test_no_header_trailer_info(
        self, monkeypatch: MonkeyPatch, pytester: Pytester, request
    ) -> None:
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
        pytester.makepyfile(
            """
            def test_passes():
                pass
        """
        )
        result = pytester.runpytest("--no-header")
        verinfo = ".".join(map(str, sys.version_info[:3]))
        result.stdout.no_fnmatch_line(
            f"platform {sys.platform} -- Python {verinfo}*pytest-{pytest.__version__}**pluggy-{pluggy.__version__}"
        )
        if request.config.pluginmanager.list_plugin_distinfo():
            result.stdout.no_fnmatch_line("plugins: *")

    # ensemble: asserts `configfile:`/`testpaths:` header lines; ensemble
    # configs are built from data and never read a config file.
    def test_header(self, pytester: Pytester) -> None:
        pytester.path.joinpath("tests").mkdir()
        pytester.path.joinpath("gui").mkdir()

        # no configuration file
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["rootdir: *test_header0"])

        # with configfile
        pytester.makeini("""[pytest]""")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["rootdir: *test_header0", "configfile: tox.ini"])

        # with testpaths option, and not passing anything in the command-line
        pytester.makeini(
            """
            [pytest]
            testpaths = tests gui
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            ["rootdir: *test_header0", "configfile: tox.ini", "testpaths: tests, gui"]
        )

        # with testpaths option, passing directory in command-line: do not show testpaths then
        result = pytester.runpytest("tests")
        result.stdout.fnmatch_lines(["rootdir: *test_header0", "configfile: tox.ini"])

    # ensemble: asserts `configfile:`/`testpaths:` header lines.
    def test_header_absolute_testpath(
        self, pytester: Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        """Regression test for #7814."""
        tests = pytester.path.joinpath("tests")
        tests.mkdir()
        pytester.makepyprojecttoml(
            f"""
            [tool.pytest.ini_options]
            testpaths = ['{tests}']
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "rootdir: *absolute_testpath0",
                "configfile: pyproject.toml",
                f"testpaths: {tests}",
            ]
        )

    # ensemble: asserts on `inifile:`/`testpaths:` header lines, which an
    # ensemble never renders in the first place.
    def test_no_header(self, pytester: Pytester) -> None:
        pytester.path.joinpath("tests").mkdir()
        pytester.path.joinpath("gui").mkdir()

        # with testpaths option, and not passing anything in the command-line
        pytester.makeini(
            """
            [pytest]
            testpaths = tests gui
        """
        )
        result = pytester.runpytest("--no-header")
        result.stdout.no_fnmatch_line(
            "rootdir: *test_header0, inifile: tox.ini, testpaths: tests, gui"
        )

        # with testpaths option, passing directory in command-line: do not show testpaths then
        result = pytester.runpytest("tests", "--no-header")
        result.stdout.no_fnmatch_line("rootdir: *test_header0, inifile: tox.ini")

    def test_no_summary(self, tmp_path: Path) -> None:
        def test_no_summary():
            # Deliberately undefined, as in the original: the test only cares
            # that the test fails and that no FAILURES section is rendered.
            assert false  # type: ignore[name-defined]  # noqa: F821

        spec = ConfigSpec(rootpath=tmp_path, args=("--no-summary",))
        record = run_tests(
            test_no_summary, spec=spec, name="test_no_summary", capture_output=True
        )
        record.stdout.no_fnmatch_line("*= FAILURES =*")
        record.assert_outcomes(failed=1)

    def test_no_summary_still_runs_terminal_summary_hook(self, tmp_path: Path) -> None:
        """--no-summary must not skip pytest_terminal_summary for plugins (#14724)."""

        class SummaryPlugin:
            def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
                terminalreporter.write_line("PLUGIN_TERMINAL_SUMMARY_RAN")

        def test_ok():
            assert True

        spec = ConfigSpec(
            rootpath=tmp_path,
            args=("--no-summary",),
            extra_plugins=(SummaryPlugin(),),
        )
        record = run_tests(test_ok, spec=spec, capture_output=True)
        record.stdout.fnmatch_lines(["PLUGIN_TERMINAL_SUMMARY_RAN"])
        record.stdout.no_fnmatch_line("*= FAILURES =*")

    def test_showlocals(self, tmp_path: Path) -> None:
        def test_showlocals():
            x = 3  # noqa: F841
            y = "x" * 5000  # noqa: F841
            assert 0

        spec = ConfigSpec(rootpath=tmp_path, args=("-l",))
        record = run_tests(
            test_showlocals, spec=spec, name="test_showlocals", capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                # "_ _ * Locals *",
                "x* = 3",
                "y* = 'xxxxxx*",
            ]
        )

    def test_noshowlocals_addopts_override(self, tmp_path: Path) -> None:
        def test_noshowlocals():
            x = 3  # noqa: F841
            y = "x" * 5000  # noqa: F841
            assert 0

        # Override global --showlocals for py.test via arg
        spec = ConfigSpec(
            rootpath=tmp_path,
            inicfg={"addopts": "--showlocals"},
            args=("--no-showlocals",),
        )
        record = run_tests(
            test_noshowlocals,
            spec=spec,
            name="test_noshowlocals",
            capture_output=True,
        )
        record.stdout.no_fnmatch_line("x* = 3")
        record.stdout.no_fnmatch_line("y* = 'xxxxxx*")

    # ensemble: `--tb=short` renders the crash location as `<file>:<lineno>`,
    # and an ensemble item's file is the host `test_terminal.py`.
    def test_showlocals_short(self, pytester: Pytester) -> None:
        p1 = pytester.makepyfile(
            """
            def test_showlocals_short():
                x = 3
                y = "xxxx"
                assert 0
        """
        )
        result = pytester.runpytest(p1, "-l", "--tb=short")
        result.stdout.fnmatch_lines(
            [
                "test_showlocals_short.py:*",
                "    assert 0",
                "E   assert 0",
                "        x          = 3",
                "        y          = 'xxxx'",
            ]
        )

    @pytest.fixture
    def verbose_testfile(self, pytester: Pytester) -> Path:
        return pytester.makepyfile(
            """
            import pytest
            def test_fail():
                raise ValueError()
            def test_pass():
                pass
            class TestClass(object):
                def test_skip(self):
                    pytest.skip("hello")
        """
        )

    def test_verbose_reporting(self, tmp_path: Path) -> None:
        def test_fail():
            raise ValueError

        def test_pass():
            pass

        class TestClass:
            def test_skip(self):
                pytest.skip("hello")

        spec = ConfigSpec(
            rootpath=tmp_path, args=("-v", "-Walways::pytest.PytestWarning")
        )
        record = run_tests(
            test_fail,
            test_pass,
            TestClass,
            spec=spec,
            name="test_verbose_reporting",
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "*test_verbose_reporting.py::test_fail *FAIL*",
                "*test_verbose_reporting.py::test_pass *PASS*",
                "*test_verbose_reporting.py::TestClass::test_skip *SKIP*",
            ]
        )
        # Stronger than the original `ret == 1`.
        record.assert_outcomes(passed=1, failed=1, skipped=1)

    # ensemble: runs the ensemble's tests through xdist workers.
    def test_verbose_reporting_xdist(
        self,
        verbose_testfile,
        monkeypatch: MonkeyPatch,
        pytester: Pytester,
        pytestconfig,
    ) -> None:
        if not pytestconfig.pluginmanager.get_plugin("xdist"):
            pytest.skip("xdist plugin not installed")

        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
        result = pytester.runpytest(
            verbose_testfile, "-v", "-n 1", "-Walways::pytest.PytestWarning"
        )
        result.stdout.fnmatch_lines(
            ["*FAIL*test_verbose_reporting_xdist.py::test_fail*"]
        )
        assert result.ret == 1

    def test_quiet_reporting(self, tmp_path: Path) -> None:
        def test_pass():
            pass

        spec = ConfigSpec(rootpath=tmp_path, args=("-q",))
        record = run_tests(
            test_pass, spec=spec, name="test_quiet_reporting", capture_output=True
        )
        s = record.output
        assert "test session starts" not in s
        assert "test_quiet_reporting.py" not in s
        assert "===" not in s
        assert "passed" in s

    def test_more_quiet_reporting(self, tmp_path: Path) -> None:
        def test_pass():
            pass

        spec = ConfigSpec(rootpath=tmp_path, args=("-qq",))
        record = run_tests(
            test_pass, spec=spec, name="test_more_quiet_reporting", capture_output=True
        )
        s = record.output
        assert "test session starts" not in s
        assert "test_more_quiet_reporting.py" not in s
        assert "===" not in s
        assert "passed" not in s

    @pytest.mark.parametrize(
        "params", [(), ("--collect-only",)], ids=["no-params", "collect-only"]
    )
    def test_report_collectionfinish_hook(self, tmp_path: Path, params) -> None:
        class CollectionFinishPlugin:
            def pytest_report_collectionfinish(self, config, start_path, items):
                return [f"hello from hook: {len(items)} items"]

        @pytest.mark.parametrize("i", range(3))
        def test(i):
            pass

        spec = ConfigSpec(
            rootpath=tmp_path,
            args=params,
            extra_plugins=(CollectionFinishPlugin(),),
        )
        record = run_tests(test, spec=spec, capture_output=True)
        record.stdout.fnmatch_lines(["collected 3 items", "hello from hook: 3 items"])

    def test_summary_f_alias(self, tmp_path: Path) -> None:
        """Test that 'f' and 'F' report chars are aliases and don't show up twice in the summary (#6334)"""

        def test():
            assert False

        spec = ConfigSpec(rootpath=tmp_path, args=("-rfF",))
        record = run_tests(
            test, spec=spec, name="test_summary_f_alias", capture_output=True
        )
        expected = "FAILED test_summary_f_alias.py::test - assert False"
        record.stdout.fnmatch_lines([expected])
        assert record.output.splitlines().count(expected) == 1

    # ensemble: the folded skip line is `<file>:<lineno>`, and an ensemble
    # item's file:line is the host `test_terminal.py`.
    def test_summary_s_alias(self, pytester: Pytester) -> None:
        """Test that 's' and 'S' report chars are aliases and don't show up twice in the summary"""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.skip
            def test():
                pass
            """
        )
        result = pytester.runpytest("-rsS")
        expected = "SKIPPED [1] test_summary_s_alias.py:3: unconditional skip"
        result.stdout.fnmatch_lines([expected])
        assert result.stdout.lines.count(expected) == 1

    # ensemble: the folded skip line is `<file>:<lineno>`, host-anchored.
    def test_summary_s_folded(self, pytester: Pytester) -> None:
        """Test that skipped tests are correctly folded"""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.parametrize("param", [True, False])
            @pytest.mark.skip("Some reason")
            def test(param):
                pass
            """
        )
        result = pytester.runpytest("-rs")
        expected = "SKIPPED [2] test_summary_s_folded.py:3: Some reason"
        result.stdout.fnmatch_lines([expected])
        assert result.stdout.lines.count(expected) == 1

    def test_summary_s_unfolded(self, tmp_path: Path) -> None:
        """Test that skipped tests are not folded if --no-fold-skipped is set"""

        @pytest.mark.parametrize("param", [True, False])
        @pytest.mark.skip("Some reason")
        def test(param):
            pass

        spec = ConfigSpec(rootpath=tmp_path, args=("-rs", "--no-fold-skipped"))
        record = run_tests(
            test, spec=spec, name="test_summary_s_unfolded", capture_output=True
        )
        expected = [
            "SKIPPED test_summary_s_unfolded.py::test[True] - Skipped: Some reason",
            "SKIPPED test_summary_s_unfolded.py::test[False] - Skipped: Some reason",
        ]
        record.stdout.fnmatch_lines(expected)
        lines = record.output.splitlines()
        assert lines.count(expected[0]) == 1
        assert lines.count(expected[1]) == 1


@pytest.mark.parametrize(
    ("use_ci", "expected_message"),
    (
        (True, f"- AssertionError: {'this_failed' * 100}"),
        (False, "- AssertionError: this_failedt..."),
    ),
    ids=("on CI", "not on CI"),
)
def test_fail_extra_reporting(
    tmp_path: Path, monkeypatch, use_ci: bool, expected_message: str
) -> None:
    if use_ci:
        monkeypatch.setenv("CI", "true")
    else:
        monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("COLUMNS", "80")

    def test_this():
        assert 0, "this_failed" * 100

    record = run_tests(
        test_this,
        spec=ConfigSpec(rootpath=tmp_path, args=("-rN",)),
        name="test_fail_extra_reporting",
        capture_output=True,
    )
    record.stdout.no_fnmatch_line("*short test summary*")
    record = run_tests(
        test_this,
        rootpath=tmp_path,
        name="test_fail_extra_reporting",
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "*test summary*",
            f"FAILED test_fail_extra_reporting.py::test_this {expected_message}",
        ]
    )


def test_fail_reporting_on_pass(tmp_path: Path) -> None:
    def test_this():
        assert 1

    record = run_tests(
        test_this,
        spec=ConfigSpec(rootpath=tmp_path, args=("-rf",)),
        capture_output=True,
    )
    record.stdout.no_fnmatch_line("*short test summary*")


def test_pass_extra_reporting(tmp_path: Path) -> None:
    def test_this():
        assert 1

    record = run_tests(
        test_this,
        rootpath=tmp_path,
        name="test_pass_extra_reporting",
        capture_output=True,
    )
    record.stdout.no_fnmatch_line("*short test summary*")
    record = run_tests(
        test_this,
        spec=ConfigSpec(rootpath=tmp_path, args=("-rp",)),
        name="test_pass_extra_reporting",
        capture_output=True,
    )
    record.stdout.fnmatch_lines(["*test summary*", "PASS*test_pass_extra_reporting*"])


def test_pass_reporting_on_fail(tmp_path: Path) -> None:
    def test_this():
        assert 0

    record = run_tests(
        test_this,
        spec=ConfigSpec(rootpath=tmp_path, args=("-rp",)),
        capture_output=True,
    )
    record.stdout.no_fnmatch_line("*short test summary*")


# ensemble: asserts on `Captured stdout` sections, which need capture.
def test_pass_output_reporting(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        def setup_module():
            print("setup_module")

        def teardown_module():
            print("teardown_module")

        def test_pass_has_output():
            print("Four score and seven years ago...")

        def test_pass_no_output():
            pass
    """
    )
    result = pytester.runpytest()
    s = result.stdout.str()
    assert "test_pass_has_output" not in s
    assert "Four score and seven years ago..." not in s
    assert "test_pass_no_output" not in s
    result = pytester.runpytest("-rPp")
    result.stdout.fnmatch_lines(
        [
            "*= PASSES =*",
            "*_ test_pass_has_output _*",
            "*- Captured stdout setup -*",
            "setup_module",
            "*- Captured stdout call -*",
            "Four score and seven years ago...",
            "*- Captured stdout teardown -*",
            "teardown_module",
            "*= short test summary info =*",
            "PASSED test_pass_output_reporting.py::test_pass_has_output",
            "PASSED test_pass_output_reporting.py::test_pass_no_output",
            "*= 2 passed in *",
        ]
    )


# ensemble: asserts the `test_color_yes.py:5:` crash lines, which for an
# ensemble source are host-anchored. `module_from_path` would make them real,
# but the imported module would not be assertion-rewritten and the
# `E       assert 0` explanation would go with it.
def test_color_yes(pytester: Pytester, color_mapping) -> None:
    p1 = pytester.makepyfile(
        """
        def fail():
            assert 0

        def test_this():
            fail()
        """
    )
    result = pytester.runpytest("--color=yes", str(p1))
    result.stdout.fnmatch_lines(
        color_mapping.format_for_fnmatch(
            [
                "{bold}=*= test session starts =*={reset}",
                "collected 1 item",
                "",
                "test_color_yes.py {red}F{reset}{red} * [100%]{reset}",
                "",
                "=*= FAILURES =*=",
                "{red}{bold}_*_ test_this _*_{reset}",
                "",
                "    {reset}{kw}def{hl-reset}{kwspace}{function}test_this{hl-reset}():{endline}",
                ">       fail(){endline}",
                "",
                "{bold}{red}test_color_yes.py{reset}:5: ",
                "_ _ * _ _*",
                "",
                "    {reset}{kw}def{hl-reset}{kwspace}{function}fail{hl-reset}():{endline}",
                ">       {kw}assert{hl-reset} {number}0{hl-reset}{endline}",
                "{bold}{red}E       assert 0{reset}",
                "",
                "{bold}{red}test_color_yes.py{reset}:2: AssertionError",
                "{red}=*= {red}{bold}1 failed{reset}{red} in *s{reset}{red} =*={reset}",
            ]
        )
    )
    result = pytester.runpytest("--color=yes", "--tb=short", str(p1))
    result.stdout.fnmatch_lines(
        color_mapping.format_for_fnmatch(
            [
                "{bold}=*= test session starts =*={reset}",
                "collected 1 item",
                "",
                "test_color_yes.py {red}F{reset}{red} * [100%]{reset}",
                "",
                "=*= FAILURES =*=",
                "{red}{bold}_*_ test_this _*_{reset}",
                "{bold}{red}test_color_yes.py{reset}:5: in test_this",
                "    {reset}fail(){endline}",
                "{bold}{red}test_color_yes.py{reset}:2: in fail",
                "    {reset}{kw}assert{hl-reset} {number}0{hl-reset}{endline}",
                "{bold}{red}E   assert 0{reset}",
                "{red}=*= {red}{bold}1 failed{reset}{red} in *s{reset}{red} =*={reset}",
            ]
        )
    )


def test_color_no(tmp_path: Path) -> None:
    def test_this():
        assert 1

    record = run_tests(
        test_this,
        spec=ConfigSpec(rootpath=tmp_path, args=("--color=no",)),
        capture_output=True,
    )
    assert "test session starts" in record.output
    record.stdout.no_fnmatch_line("*\x1b[1m*")


@pytest.mark.parametrize("verbose", [True, False])
def test_color_yes_collection_on_non_atty(tmp_path: Path, verbose) -> None:
    """#1397: Skip collect progress report when working on non-terminals."""

    @pytest.mark.parametrize("i", range(10))
    def test_this(i):
        assert 1

    args = ["--color=yes"]
    if verbose:
        args.append("-vv")
    record = run_tests(
        test_this,
        spec=ConfigSpec(rootpath=tmp_path, args=tuple(args)),
        capture_output=True,
    )
    assert "test session starts" in record.output
    assert "\x1b[1m" in record.output
    record.stdout.no_fnmatch_line("*collecting 10 items*")
    if verbose:
        assert "collecting ..." in record.output
    assert "collected 10 items" in record.output


def test_getreportopt() -> None:
    from _pytest.terminal import _REPORTCHARS_DEFAULT

    class FakeConfig:
        class Option:
            reportchars = _REPORTCHARS_DEFAULT
            disable_warnings = False

        option = Option()

    config = cast(Config, FakeConfig())

    assert _REPORTCHARS_DEFAULT == "fE"

    # Default.
    assert getreportopt(config) == "wfE"

    config.option.reportchars = "sf"
    assert getreportopt(config) == "wsf"

    config.option.reportchars = "sfxw"
    assert getreportopt(config) == "sfxw"

    config.option.reportchars = "a"
    assert getreportopt(config) == "wsxXEf"

    config.option.reportchars = "N"
    assert getreportopt(config) == "w"

    config.option.reportchars = "NwfE"
    assert getreportopt(config) == "wfE"

    config.option.reportchars = "NfENx"
    assert getreportopt(config) == "wx"

    # Now with --disable-warnings.
    config.option.disable_warnings = True
    config.option.reportchars = "a"
    assert getreportopt(config) == "sxXEf"

    config.option.reportchars = "sfx"
    assert getreportopt(config) == "sfx"

    config.option.reportchars = "sfxw"
    assert getreportopt(config) == "sfx"

    config.option.reportchars = "a"
    assert getreportopt(config) == "sxXEf"

    config.option.reportchars = "A"
    assert getreportopt(config) == "PpsxXEf"

    config.option.reportchars = "AN"
    assert getreportopt(config) == ""

    config.option.reportchars = "NwfE"
    assert getreportopt(config) == "fE"


def test_terminalreporter_reportopt_addopts(tmp_path: Path) -> None:
    @pytest.fixture
    def tr(request):
        tr = request.config.pluginmanager.getplugin("terminalreporter")
        return tr

    def test_opt(tr):
        assert tr.hasopt("skipped")
        assert not tr.hasopt("qwe")

    spec = ConfigSpec(rootpath=tmp_path, inicfg={"addopts": "-rs"})
    record = run_tests(
        build_module("test_reportopt_addopts", tr=tr, test_opt=test_opt),
        spec=spec,
        capture_output=True,
    )
    record.stdout.fnmatch_lines(["*1 passed*"])
    record.assert_outcomes(passed=1)


# ensemble: asserts `*<file>:8*`; an ensemble item's file:line is host-anchored.
def test_tbstyle_short(pytester: Pytester) -> None:
    p = pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def arg(request):
            return 42
        def test_opt(arg):
            x = 0
            assert x
    """
    )
    result = pytester.runpytest("--tb=short")
    s = result.stdout.str()
    assert "arg = 42" not in s
    assert "x = 0" not in s
    result.stdout.fnmatch_lines([f"*{p.name}:8*", "    assert x", "E   assert*"])
    result = pytester.runpytest()
    s = result.stdout.str()
    assert "x = 0" in s
    assert "assert x" in s


# ensemble: asserts the NO_TESTS_COLLECTED exit code, which comes from
# `wrap_session`.
def test_traceconfig(pytester: Pytester) -> None:
    result = pytester.runpytest("--traceconfig")
    result.stdout.fnmatch_lines(["*active plugins*"])
    assert result.ret == ExitCode.NO_TESTS_COLLECTED


class TestGenericReporting:
    """Test class which can be subclassed with a different option provider to
    run e.g. distributed tests."""

    # ensemble: needs a module that raises ImportError at import time;
    # ensemble sources are already-imported objects.
    def test_collect_fail(self, pytester: Pytester, option) -> None:
        pytester.makepyfile("import xyz\n")
        result = pytester.runpytest(*option.args)
        result.stdout.fnmatch_lines(
            ["ImportError while importing*", "*No module named *xyz*", "*1 error*"]
        )

    # ensemble: `! stopping after N failures !` is emitted from the run loop
    # in `_pytest.main`, which an ensemble bypasses.
    def test_maxfailures(self, pytester: Pytester, option) -> None:
        pytester.makepyfile(
            """
            def test_1():
                assert 0
            def test_2():
                assert 0
            def test_3():
                assert 0
        """
        )
        result = pytester.runpytest("--maxfail=2", *option.args)
        result.stdout.fnmatch_lines(
            [
                "*def test_1():*",
                "*def test_2():*",
                "*! stopping after 2 failures !*",
                "*2 failed*",
            ]
        )

    # ensemble: `! session_interrupted !` is emitted from the run loop.
    def test_maxfailures_with_interrupted(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            def test(request):
                request.session.shouldstop = "session_interrupted"
                assert 0
        """
        )
        result = pytester.runpytest("--maxfail=1", "-ra")
        result.stdout.fnmatch_lines(
            [
                "*= short test summary info =*",
                "FAILED *",
                "*! stopping after 1 failures !*",
                "*! session_interrupted !*",
                "*= 1 failed in*",
            ]
        )

    def test_tb_option(self, tmp_path: Path, option) -> None:
        def g():
            raise IndexError

        def test_func():
            print(6 * 7)
            g()  # --calling--

        module = build_module("test_tb_option", g=g, test_func=test_func)
        for tbopt in ["long", "short", "no"]:
            print(f"testing --tb={tbopt}...")
            spec = ConfigSpec(rootpath=tmp_path, args=("-rN", f"--tb={tbopt}"))
            s = run_tests(module, spec=spec, capture_output=True).output
            if tbopt == "long":
                assert "print(6 * 7)" in s
            else:
                assert "print(6 * 7)" not in s
            if tbopt != "no":
                assert "--calling--" in s
                assert "IndexError" in s
            else:
                assert "FAILURES" not in s
                assert "--calling--" not in s
                assert "IndexError" not in s

    # ensemble: asserts a `Captured stdout call` section, which needs capture.
    def test_tb_line_show_capture(self, pytester: Pytester, option) -> None:
        output_to_capture = "help! let me out!"
        pytester.makepyfile(
            f"""
            import pytest
            def test_fail():
                print('{output_to_capture}')
                assert False
            """
        )
        result = pytester.runpytest("--tb=line")
        result.stdout.fnmatch_lines(["*- Captured stdout call -*", output_to_capture])

    # ensemble: asserts `<file>:<lineno>` crash lines, host-anchored.
    def test_tb_crashline(self, pytester: Pytester, option) -> None:
        p = pytester.makepyfile(
            """
            import pytest
            def g():
                raise IndexError
            def test_func1():
                print(6*7)
                g()  # --calling--
            def test_func2():
                assert 0, "hello"
        """
        )
        result = pytester.runpytest("--tb=line")
        bn = p.name
        result.stdout.fnmatch_lines(
            [f"*{bn}:3: IndexError*", f"*{bn}:8: AssertionError: hello*"]
        )
        s = result.stdout.str()
        assert "def test_func2" not in s

    # ensemble: asserts a `<file>:<lineno>` crash line, host-anchored.
    def test_tb_crashline_pytrace_false(self, pytester: Pytester, option) -> None:
        p = pytester.makepyfile(
            """
            import pytest
            def test_func1():
                pytest.fail('test_func1', pytrace=False)
        """
        )
        result = pytester.runpytest("--tb=line")
        result.stdout.str()
        bn = p.name
        result.stdout.fnmatch_lines([f"*{bn}:3: Failed: test_func1"])

    # ensemble: the point is that a subdirectory conftest contributes header
    # lines; ensembles have no directory tree to scope conftests to.
    def test_pytest_report_header(self, pytester: Pytester, option) -> None:
        pytester.makeconftest(
            """
            def pytest_sessionstart(session):
                session.config._somevalue = 42
            def pytest_report_header(config):
                return "hello: %s" % config._somevalue
        """
        )
        pytester.mkdir("a").joinpath("conftest.py").write_text(
            """
def pytest_report_header(config, start_path):
    return ["line1", str(start_path)]
""",
            encoding="utf-8",
        )
        result = pytester.runpytest("a")
        result.stdout.fnmatch_lines(["*hello: 42*", "line1", str(pytester.path)])

    # ensemble: `--show-capture` is entirely about captured output sections.
    def test_show_capture(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import sys
            import logging
            def test_one():
                sys.stdout.write('!This is stdout!')
                sys.stderr.write('!This is stderr!')
                logging.warning('!This is a warning log msg!')
                assert False, 'Something failed'
        """
        )

        result = pytester.runpytest("--tb=short")
        result.stdout.fnmatch_lines(
            [
                "!This is stdout!",
                "!This is stderr!",
                "*WARNING*!This is a warning log msg!",
            ]
        )

        result = pytester.runpytest("--show-capture=all", "--tb=short")
        result.stdout.fnmatch_lines(
            [
                "!This is stdout!",
                "!This is stderr!",
                "*WARNING*!This is a warning log msg!",
            ]
        )

        stdout = pytester.runpytest("--show-capture=stdout", "--tb=short").stdout.str()
        assert "!This is stderr!" not in stdout
        assert "!This is stdout!" in stdout
        assert "!This is a warning log msg!" not in stdout

        stdout = pytester.runpytest("--show-capture=stderr", "--tb=short").stdout.str()
        assert "!This is stdout!" not in stdout
        assert "!This is stderr!" in stdout
        assert "!This is a warning log msg!" not in stdout

        stdout = pytester.runpytest("--show-capture=log", "--tb=short").stdout.str()
        assert "!This is stdout!" not in stdout
        assert "!This is stderr!" not in stdout
        assert "!This is a warning log msg!" in stdout

        stdout = pytester.runpytest("--show-capture=no", "--tb=short").stdout.str()
        assert "!This is stdout!" not in stdout
        assert "!This is stderr!" not in stdout
        assert "!This is a warning log msg!" not in stdout

    # ensemble: `--show-capture` is entirely about captured output sections.
    def test_show_capture_with_teardown_logs(self, pytester: Pytester) -> None:
        """Ensure that the capturing of teardown logs honor --show-capture setting"""
        pytester.makepyfile(
            """
            import logging
            import sys
            import pytest

            @pytest.fixture(scope="function", autouse="True")
            def hook_each_test(request):
                yield
                sys.stdout.write("!stdout!")
                sys.stderr.write("!stderr!")
                logging.warning("!log!")

            def test_func():
                assert False
        """
        )

        result = pytester.runpytest("--show-capture=stdout", "--tb=short").stdout.str()
        assert "!stdout!" in result
        assert "!stderr!" not in result
        assert "!log!" not in result

        result = pytester.runpytest("--show-capture=stderr", "--tb=short").stdout.str()
        assert "!stdout!" not in result
        assert "!stderr!" in result
        assert "!log!" not in result

        result = pytester.runpytest("--show-capture=log", "--tb=short").stdout.str()
        assert "!stdout!" not in result
        assert "!stderr!" not in result
        assert "!log!" in result

        result = pytester.runpytest("--show-capture=no", "--tb=short").stdout.str()
        assert "!stdout!" not in result
        assert "!stderr!" not in result
        assert "!log!" not in result


# ensemble: uses the `capfd` fixture, which needs capture.
@pytest.mark.xfail("not hasattr(os, 'dup')")
def test_fdopen_kept_alive_issue124(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import os, sys
        k = []
        def test_open_file_and_keep_alive(capfd):
            stdout = os.fdopen(1, 'w', buffering=1, encoding='utf-8')
            k.append(stdout)

        def test_close_kept_alive_file():
            stdout = k.pop()
            stdout.close()
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*2 passed*"])


# ensemble: asserts the file name in a native traceback frame, which for an
# ensemble source is the host `test_terminal.py`.
def test_tbstyle_native_setup_error(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest
        @pytest.fixture
        def setup_error_fixture():
            raise Exception("error in exception")

        def test_error_fixture(setup_error_fixture):
            pass
    """
    )
    result = pytester.runpytest("--tb=native")
    result.stdout.fnmatch_lines(
        ['*File *test_tbstyle_native_setup_error.py", line *, in setup_error_fixture*']
    )


# ensemble: asserts `exitstatus: 5` (NO_TESTS_COLLECTED); the exit status an
# ensemble hands to `pytest_sessionfinish` is not computed by `wrap_session`.
def test_terminal_summary(pytester: Pytester) -> None:
    pytester.makeconftest(
        """
        def pytest_terminal_summary(terminalreporter, exitstatus):
            w = terminalreporter
            w.section("hello")
            w.line("world")
            w.line("exitstatus: {0}".format(exitstatus))
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        """
        *==== hello ====*
        world
        exitstatus: 5
    """
    )


# ensemble: asserts `*conftest.py:3:*internal warning`, i.e. the file and line
# of the warning's origin, which for an ensemble plugin object is the host file.
@pytest.mark.filterwarnings("default::UserWarning")
def test_terminal_summary_warnings_are_displayed(pytester: Pytester) -> None:
    """Test that warnings emitted during pytest_terminal_summary are displayed.
    (#1305).
    """
    pytester.makeconftest(
        """
        import warnings
        def pytest_terminal_summary(terminalreporter):
            warnings.warn(UserWarning('internal warning'))
    """
    )
    pytester.makepyfile(
        """
        def test_failure():
            import warnings
            warnings.warn("warning_from_" + "test")
            assert 0
    """
    )
    result = pytester.runpytest("-ra")
    result.stdout.fnmatch_lines(
        [
            "*= warnings summary =*",
            "*warning_from_test*",
            "*= short test summary info =*",
            "*= warnings summary (final) =*",
            "*conftest.py:3:*internal warning",
            "*== 1 failed, 2 warnings in *",
        ]
    )
    result.stdout.no_fnmatch_line("*None*")
    stdout = result.stdout.str()
    assert stdout.count("warning_from_test") == 1
    assert stdout.count("=== warnings summary ") == 2


def test_terminal_summary_warnings_header_once(tmp_path: Path) -> None:
    def test_failure():
        import warnings

        warnings.warn("warning_from_" + "test")
        assert 0

    # The host suite runs with `filterwarnings = error`; the ensemble needs
    # the warning to be shown rather than raised.
    spec = ConfigSpec(
        rootpath=tmp_path,
        args=("-ra",),
        inicfg={"filterwarnings": ["default::UserWarning"]},
    )
    record = run_tests(
        test_failure,
        spec=spec,
        name="test_terminal_summary_warnings_header_once",
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "*= warnings summary =*",
            "*warning_from_test*",
            "*= short test summary info =*",
            "*== 1 failed, 1 warning in *",
        ]
    )
    record.stdout.no_fnmatch_line("*None*")
    stdout = record.output
    assert stdout.count("warning_from_test") == 1
    assert stdout.count("=== warnings summary ") == 1


def test_terminal_no_summary_warnings_header_once(tmp_path: Path) -> None:
    def test_failure():
        import warnings

        warnings.warn("warning_from_" + "test")
        assert 0

    spec = ConfigSpec(
        rootpath=tmp_path,
        args=("--no-summary",),
        inicfg={"filterwarnings": ["default"]},
    )
    record = run_tests(
        test_failure,
        spec=spec,
        name="test_terminal_no_summary_warnings_header_once",
        capture_output=True,
    )
    record.stdout.no_fnmatch_line("*= warnings summary =*")
    record.stdout.no_fnmatch_line("*= short test summary info =*")


@pytest.fixture(scope="session")
def tr() -> TerminalReporter:
    config = _pytest.config._prepareconfig([])
    return TerminalReporter(config)


@pytest.mark.parametrize(
    "exp_color, exp_line, stats_arg",
    [
        # The method under test only cares about the length of each
        # dict value, not the actual contents, so tuples of anything
        # suffice
        # Important statuses -- the highest priority of these always wins
        ("red", [("1 failed", {"bold": True, "red": True})], {"failed": [1]}),
        (
            "red",
            [
                ("1 failed", {"bold": True, "red": True}),
                ("1 passed", {"bold": False, "green": True}),
            ],
            {"failed": [1], "passed": [1]},
        ),
        ("red", [("1 error", {"bold": True, "red": True})], {"error": [1]}),
        ("red", [("2 errors", {"bold": True, "red": True})], {"error": [1, 2]}),
        (
            "red",
            [
                ("1 passed", {"bold": False, "green": True}),
                ("1 error", {"bold": True, "red": True}),
            ],
            {"error": [1], "passed": [1]},
        ),
        # (a status that's not known to the code)
        ("yellow", [("1 weird", {"bold": True, "yellow": True})], {"weird": [1]}),
        (
            "yellow",
            [
                ("1 passed", {"bold": False, "green": True}),
                ("1 weird", {"bold": True, "yellow": True}),
            ],
            {"weird": [1], "passed": [1]},
        ),
        ("yellow", [("1 warning", {"bold": True, "yellow": True})], {"warnings": [1]}),
        (
            "yellow",
            [
                ("1 passed", {"bold": False, "green": True}),
                ("1 warning", {"bold": True, "yellow": True}),
            ],
            {"warnings": [1], "passed": [1]},
        ),
        (
            "green",
            [("5 passed", {"bold": True, "green": True})],
            {"passed": [1, 2, 3, 4, 5]},
        ),
        # "Boring" statuses.  These have no effect on the color of the summary
        # line.  Thus, if *every* test has a boring status, the summary line stays
        # at its default color, i.e. yellow, to warn the user that the test run
        # produced no useful information
        ("yellow", [("1 skipped", {"bold": True, "yellow": True})], {"skipped": [1]}),
        (
            "green",
            [
                ("1 passed", {"bold": True, "green": True}),
                ("1 skipped", {"bold": False, "yellow": True}),
            ],
            {"skipped": [1], "passed": [1]},
        ),
        (
            "yellow",
            [("1 deselected", {"bold": True, "yellow": True})],
            {"deselected": [1]},
        ),
        (
            "green",
            [
                ("1 passed", {"bold": True, "green": True}),
                ("1 deselected", {"bold": False, "yellow": True}),
            ],
            {"deselected": [1], "passed": [1]},
        ),
        ("yellow", [("1 xfailed", {"bold": True, "yellow": True})], {"xfailed": [1]}),
        (
            "green",
            [
                ("1 passed", {"bold": True, "green": True}),
                ("1 xfailed", {"bold": False, "yellow": True}),
            ],
            {"xfailed": [1], "passed": [1]},
        ),
        ("yellow", [("1 xpassed", {"bold": True, "yellow": True})], {"xpassed": [1]}),
        (
            "yellow",
            [
                ("1 passed", {"bold": False, "green": True}),
                ("1 xpassed", {"bold": True, "yellow": True}),
            ],
            {"xpassed": [1], "passed": [1]},
        ),
        # Likewise if no tests were found at all
        ("yellow", [("no tests ran", {"yellow": True})], {}),
        # Test the empty-key special case
        ("yellow", [("no tests ran", {"yellow": True})], {"": [1]}),
        (
            "green",
            [("1 passed", {"bold": True, "green": True})],
            {"": [1], "passed": [1]},
        ),
        # A couple more complex combinations
        (
            "red",
            [
                ("1 failed", {"bold": True, "red": True}),
                ("2 passed", {"bold": False, "green": True}),
                ("3 xfailed", {"bold": False, "yellow": True}),
            ],
            {"passed": [1, 2], "failed": [1], "xfailed": [1, 2, 3]},
        ),
        (
            "green",
            [
                ("1 passed", {"bold": True, "green": True}),
                ("2 skipped", {"bold": False, "yellow": True}),
                ("3 deselected", {"bold": False, "yellow": True}),
                ("2 xfailed", {"bold": False, "yellow": True}),
            ],
            {
                "passed": [1],
                "skipped": [1, 2],
                "deselected": [1, 2, 3],
                "xfailed": [1, 2],
            },
        ),
    ],
)
def test_summary_stats(
    tr: TerminalReporter,
    exp_line: list[tuple[str, dict[str, bool]]],
    exp_color: str,
    stats_arg: dict[str, list[object]],
) -> None:
    tr.stats = stats_arg

    # Fake "_is_last_item" to be True.
    class fake_session:
        testscollected = 0

    tr._session = fake_session  # type: ignore[assignment]
    assert tr._is_last_item

    # Reset cache.
    tr._main_color = None

    print(f"Based on stats: {stats_arg}")
    print(f'Expect summary: "{exp_line}"; with color "{exp_color}"')
    (line, color) = tr.build_summary_stats_line()
    print(f'Actually got:   "{line}"; with color "{color}"')
    assert line == exp_line
    assert color == exp_color


def test_skip_counting_towards_summary(tr):
    class DummyReport(BaseReport):
        count_towards_summary = True

    r1 = DummyReport()
    r2 = DummyReport()
    tr.stats = {"failed": (r1, r2)}
    tr._main_color = None
    res = tr.build_summary_stats_line()
    assert res == ([("2 failed", {"bold": True, "red": True})], "red")

    r1.count_towards_summary = False
    tr.stats = {"failed": (r1, r2)}
    tr._main_color = None
    res = tr.build_summary_stats_line()
    assert res == ([("1 failed", {"bold": True, "red": True})], "red")


class TestClassicOutputStyle:
    """Ensure classic output style works as expected (#3883)"""

    @pytest.fixture
    def test_files(self) -> tuple[object, ...]:
        def test_one():
            pass

        def test_two():
            assert 0

        def test_three_1():
            pass

        def test_three_2():
            assert 0

        def test_three_3():
            pass

        # Collected in the order a filesystem walk would produce them.
        return (
            build_module(
                "sub/test_three",
                test_three_1=test_three_1,
                test_three_2=test_three_2,
                test_three_3=test_three_3,
            ),
            build_module("test_one", test_one=test_one),
            build_module("test_two", test_two=test_two),
        )

    @staticmethod
    def _run(tmp_path: Path, test_files, *args: str):
        spec = ConfigSpec(
            rootpath=tmp_path, args=("-o", "console_output_style=classic", *args)
        )
        return run_tests(*test_files, spec=spec, capture_output=True)

    def test_normal_verbosity(self, tmp_path: Path, test_files) -> None:
        record = self._run(tmp_path, test_files)
        record.stdout.fnmatch_lines(
            [
                f"sub{os.sep}test_three.py .F.",
                "test_one.py .",
                "test_two.py F",
                "*2 failed, 3 passed in*",
            ]
        )
        record.assert_outcomes(passed=3, failed=2)

    def test_verbose(self, tmp_path: Path, test_files) -> None:
        record = self._run(tmp_path, test_files, "-v")
        record.stdout.fnmatch_lines(
            [
                f"sub{os.sep}test_three.py::test_three_1 PASSED",
                f"sub{os.sep}test_three.py::test_three_2 FAILED",
                f"sub{os.sep}test_three.py::test_three_3 PASSED",
                "test_one.py::test_one PASSED",
                "test_two.py::test_two FAILED",
                "*2 failed, 3 passed in*",
            ]
        )
        record.assert_outcomes(passed=3, failed=2)

    def test_quiet(self, tmp_path: Path, test_files) -> None:
        record = self._run(tmp_path, test_files, "-q")
        record.stdout.fnmatch_lines([".F..F", "*2 failed, 3 passed in*"])
        record.assert_outcomes(passed=3, failed=2)


# ensemble: asserts a usage error written to stderr by the command line parser.
def test_console_output_style_invalid(pytester: Pytester) -> None:
    """An invalid console_output_style fails with a clean usage error."""
    result = pytester.runpytest("-o", "console_output_style=fancy")
    assert result.ret == ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(
        [
            "*ERROR: *config option 'console_output_style' expects one of "
            "'classic' | 'progress' | 'count' | 'times' | "
            "'progress-even-when-capture-no', got 'fancy'"
        ]
    )


class TestProgressOutputStyle:
    @pytest.fixture
    def many_tests_sources(self) -> tuple[ModuleType, ...]:
        @pytest.mark.parametrize("i", range(10))
        def test_bar(i):
            pass

        @pytest.mark.parametrize("i", range(5))
        def test_foo(i):
            pass

        @pytest.mark.parametrize("i", range(5))
        def test_foobar(i):
            pass

        return (
            build_module("test_bar", test_bar),
            build_module("test_foo", test_foo),
            build_module("test_foobar", test_foobar),
        )

    @staticmethod
    def _run(
        tmp_path: Path,
        sources: tuple[ModuleType, ...],
        *args: str,
        inicfg: dict[str, object] | None = None,
    ) -> RunRecord:
        spec = ConfigSpec(rootpath=tmp_path, args=args, inicfg=inicfg or {})
        return run_tests(*sources, spec=spec, capture_output=True)

    @pytest.fixture
    def many_tests_files(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_bar="""
                import pytest
                @pytest.mark.parametrize('i', range(10))
                def test_bar(i): pass
            """,
            test_foo="""
                import pytest
                @pytest.mark.parametrize('i', range(5))
                def test_foo(i): pass
            """,
            test_foobar="""
                import pytest
                @pytest.mark.parametrize('i', range(5))
                def test_foobar(i): pass
            """,
        )

    @pytest.fixture
    def more_tests_files(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_bar="""
                import pytest
                @pytest.mark.parametrize('i', range(30))
                def test_bar(i): pass
            """,
            test_foo="""
                import pytest
                @pytest.mark.parametrize('i', range(5))
                def test_foo(i): pass
            """,
        )

    def test_zero_tests_collected(self, tmp_path: Path) -> None:
        """Some plugins (testmon for example) might issue pytest_runtest_logreport without any tests being
        actually collected (#2971)."""

        class LogReportsWithoutItems:
            def pytest_collection_modifyitems(self, items, config):
                for node_id in ("nodeid1", "nodeid2"):
                    rep = CollectReport(node_id, "passed", None, None)
                    rep.when = "passed"
                    rep.duration = 0.1  # type: ignore[attr-defined]
                    config.hook.pytest_runtest_logreport(report=rep)

        record = run_tests(
            spec=ConfigSpec(
                rootpath=tmp_path, extra_plugins=(LogReportsWithoutItems(),)
            ),
            capture_output=True,
        )
        record.stdout.no_fnmatch_line("*ZeroDivisionError*")
        record.stdout.fnmatch_lines(["=* 2 passed in *="])

    def test_normal(self, tmp_path: Path, many_tests_sources) -> None:
        record = self._run(tmp_path, many_tests_sources)
        record.stdout.re_match_lines(
            [
                r"test_bar.py \.{10} \s+ \[ 50%\]",
                r"test_foo.py \.{5} \s+ \[ 75%\]",
                r"test_foobar.py \.{5} \s+ \[100%\]",
            ]
        )
        record.assert_outcomes(passed=20)

    def test_colored_progress(self, tmp_path: Path, monkeypatch, color_mapping) -> None:
        monkeypatch.setenv("PY_COLORS", "1")

        @pytest.mark.xfail
        def test_axfail():
            assert 0

        @pytest.mark.parametrize("i", range(10))
        def test_bar(i):
            pass

        @pytest.mark.parametrize("i", range(5))
        def test_foo(i):
            import warnings

            warnings.warn(DeprecationWarning("collection"))

        @pytest.mark.parametrize("i", range(5))
        def test_foobar(i):
            raise ValueError

        axfail_module = build_module("test_axfail", test_axfail)
        sources = (
            axfail_module,
            build_module("test_bar", test_bar),
            build_module("test_foo", test_foo),
            build_module("test_foobar", test_foobar),
        )
        # The host suite turns warnings into errors; the point here is the
        # yellow progress indicator a *recorded* warning produces.
        inicfg: dict[str, object] = {"filterwarnings": ["always"]}
        record = self._run(tmp_path, sources, inicfg=inicfg)
        record.stdout.re_match_lines(
            color_mapping.format_for_rematch(
                [
                    r"test_axfail.py {yellow}x{reset}{green} \s+ \[  4%\]{reset}",
                    r"test_bar.py ({green}\.{reset}){{10}}{green} \s+ \[ 52%\]{reset}",
                    r"test_foo.py ({green}\.{reset}){{5}}{yellow} \s+ \[ 76%\]{reset}",
                    r"test_foobar.py ({red}F{reset}){{5}}{red} \s+ \[100%\]{reset}",
                ]
            )
        )
        record.assert_outcomes(passed=15, failed=5, xfailed=1, warnings=5)

        # Only xfail should have yellow progress indicator.
        record = self._run(tmp_path, (axfail_module,))
        record.stdout.re_match_lines(
            color_mapping.format_for_rematch(
                [
                    r"test_axfail.py {yellow}x{reset}{yellow} \s+ \[100%\]{reset}",
                    r"^{yellow}=+ ({yellow}{bold}|{bold}{yellow})1 xfailed{reset}{yellow} in ",
                ]
            )
        )
        record.assert_outcomes(xfailed=1)

    def test_count(self, tmp_path: Path, many_tests_sources) -> None:
        record = self._run(
            tmp_path, many_tests_sources, inicfg={"console_output_style": "count"}
        )
        record.stdout.re_match_lines(
            [
                r"test_bar.py \.{10} \s+ \[10/20\]",
                r"test_foo.py \.{5} \s+ \[15/20\]",
                r"test_foobar.py \.{5} \s+ \[20/20\]",
            ]
        )
        record.assert_outcomes(passed=20)

    # ensemble: `console_output_style=times` groups reports by
    # ``report.location[0]``, which for a synthesized ensemble module is the
    # *host* file - every item then looks like it belongs to one giant module
    # and only the very last line gets a duration. See test_times_none_collected.
    def test_times(self, many_tests_files, pytester: Pytester) -> None:
        pytester.makeini(
            """
            [pytest]
            console_output_style = times
        """
        )
        output = pytester.runpytest()
        output.stdout.re_match_lines(
            [
                r"test_bar.py \.{10} \s+ \d{1,3}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2}$",
                r"test_foo.py \.{5} \s+ \d{1,3}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2}$",
                r"test_foobar.py \.{5} \s+ \d{1,3}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2}$",
            ]
        )

    # ensemble: see test_times.
    def test_times_multiline(
        self, more_tests_files, monkeypatch, pytester: Pytester
    ) -> None:
        monkeypatch.setenv("COLUMNS", "40")
        pytester.makeini(
            """
            [pytest]
            console_output_style = times
        """
        )
        output = pytester.runpytest()
        output.stdout.re_match_lines(
            [
                r"test_bar.py ...................",
                r"........... \s+ \d{1,4}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2}$",
                r"test_foo.py \.{5} \s+ \d{1,4}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2}$",
            ],
            consecutive=True,
        )

    # ensemble: asserts the NO_TESTS_COLLECTED exit code, which comes from
    # `wrap_session`; an ensemble has no session wrapper and no exit code.
    def test_times_none_collected(self, pytester: Pytester) -> None:
        pytester.makeini(
            """
            [pytest]
            console_output_style = times
        """
        )
        output = pytester.runpytest()
        assert output.ret == ExitCode.NO_TESTS_COLLECTED

    def test_verbose(self, tmp_path: Path, many_tests_sources) -> None:
        record = self._run(tmp_path, many_tests_sources, "-v")
        record.stdout.re_match_lines(
            [
                r"test_bar.py::test_bar\[0\] PASSED \s+ \[  5%\]",
                r"test_foo.py::test_foo\[4\] PASSED \s+ \[ 75%\]",
                r"test_foobar.py::test_foobar\[4\] PASSED \s+ \[100%\]",
            ]
        )
        record.assert_outcomes(passed=20)

    def test_verbose_count(self, tmp_path: Path, many_tests_sources) -> None:
        record = self._run(
            tmp_path, many_tests_sources, "-v", inicfg={"console_output_style": "count"}
        )
        record.stdout.re_match_lines(
            [
                r"test_bar.py::test_bar\[0\] PASSED \s+ \[ 1/20\]",
                r"test_foo.py::test_foo\[4\] PASSED \s+ \[15/20\]",
                r"test_foobar.py::test_foobar\[4\] PASSED \s+ \[20/20\]",
            ]
        )
        record.assert_outcomes(passed=20)

    # ensemble: see test_times.
    def test_verbose_times(self, many_tests_files, pytester: Pytester) -> None:
        pytester.makeini(
            """
            [pytest]
            console_output_style = times
        """
        )
        output = pytester.runpytest("-v")
        output.stdout.re_match_lines(
            [
                r"test_bar.py::test_bar\[0\] PASSED \s+ \d{1,3}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2}$",
                r"test_foo.py::test_foo\[4\] PASSED \s+ \d{1,3}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2}$",
                r"test_foobar.py::test_foobar\[4\] PASSED \s+ \d{1,3}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2}$",
            ]
        )

    # ensemble: the four xdist tests below run the tests through xdist
    # workers, i.e. subprocesses.
    def test_xdist_normal(
        self, many_tests_files, pytester: Pytester, monkeypatch
    ) -> None:
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        output = pytester.runpytest("-n2")
        output.stdout.re_match_lines([r"\.{20} \s+ \[100%\]"])

    def test_xdist_normal_count(
        self, many_tests_files, pytester: Pytester, monkeypatch
    ) -> None:
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        pytester.makeini(
            """
            [pytest]
            console_output_style = count
        """
        )
        output = pytester.runpytest("-n2")
        output.stdout.re_match_lines([r"\.{20} \s+ \[20/20\]"])

    def test_xdist_verbose(
        self, many_tests_files, pytester: Pytester, monkeypatch
    ) -> None:
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        output = pytester.runpytest("-n2", "-v")
        output.stdout.re_match_lines_random(
            [
                r"\[gw\d\] \[\s*\d+%\] PASSED test_bar.py::test_bar\[1\]",
                r"\[gw\d\] \[\s*\d+%\] PASSED test_foo.py::test_foo\[1\]",
                r"\[gw\d\] \[\s*\d+%\] PASSED test_foobar.py::test_foobar\[1\]",
            ]
        )
        output.stdout.fnmatch_lines_random(
            [
                line.translate(TRANS_FNMATCH)
                for line in [
                    "test_bar.py::test_bar[0] ",
                    "test_foo.py::test_foo[0] ",
                    "test_foobar.py::test_foobar[0] ",
                    "[gw?] [  5%] PASSED test_*[?] ",
                    "[gw?] [ 10%] PASSED test_*[?] ",
                    "[gw?] [ 55%] PASSED test_*[?] ",
                    "[gw?] [ 60%] PASSED test_*[?] ",
                    "[gw?] [ 95%] PASSED test_*[?] ",
                    "[gw?] [100%] PASSED test_*[?] ",
                ]
            ]
        )

    def test_xdist_times(
        self, many_tests_files, pytester: Pytester, monkeypatch
    ) -> None:
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        pytester.makeini(
            """
            [pytest]
            console_output_style = times
        """
        )
        output = pytester.runpytest("-n2", "-v")
        output.stdout.re_match_lines_random(
            [
                r"\[gw\d\] \d{1,3}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2} PASSED test_bar.py::test_bar\[1\]",
                r"\[gw\d\] \d{1,3}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2} PASSED test_foo.py::test_foo\[1\]",
                r"\[gw\d\] \d{1,3}[\.[a-z\ ]{1,2}\d{0,3}\w{1,2} PASSED test_foobar.py::test_foobar\[1\]",
            ]
        )

    # ensemble: the point is that `--capture=no` suppresses the progress
    # column. An ensemble reporter writes to its own private stream, which
    # counts as captured no matter what `--capture` says, so the column stays
    # on and the `no_fnmatch_line("*%]*")` half could never hold.
    def test_capture_no(self, many_tests_files, pytester: Pytester) -> None:
        output = pytester.runpytest("-s")
        output.stdout.re_match_lines(
            [r"test_bar.py \.{10}", r"test_foo.py \.{5}", r"test_foobar.py \.{5}"]
        )

        output = pytester.runpytest("--capture=no")
        output.stdout.no_fnmatch_line("*%]*")

    # ensemble: the ini value under test only does anything when
    # `--capture=no` would otherwise suppress the column; in an ensemble the
    # column is on regardless, so this would assert nothing. See
    # test_capture_no.
    def test_capture_no_progress_enabled(
        self, many_tests_files, pytester: Pytester
    ) -> None:
        pytester.makeini(
            """
            [pytest]
            console_output_style = progress-even-when-capture-no
        """
        )
        output = pytester.runpytest("-s")
        output.stdout.re_match_lines(
            [
                r"test_bar.py \.{10} \s+ \[ 50%\]",
                r"test_foo.py \.{5} \s+ \[ 75%\]",
                r"test_foobar.py \.{5} \s+ \[100%\]",
            ]
        )


class TestProgressWithTeardown:
    """Ensure we show the correct percentages for tests that fail during teardown (#3088)"""

    @pytest.fixture
    def teardown_fixture_plugin(self) -> object:
        """The ensemble equivalent of a conftest at the rootdir."""

        class TeardownFixturePlugin:
            @pytest.fixture
            def fail_teardown(self):
                yield
                assert False

        return TeardownFixturePlugin()

    @pytest.fixture
    def many_sources(self) -> tuple[ModuleType, ...]:
        @pytest.mark.parametrize("i", range(5))
        def test_bar(fail_teardown, i):
            pass

        @pytest.mark.parametrize("i", range(15))
        def test_foo(fail_teardown, i):
            pass

        return (
            build_module("test_bar", test_bar),
            build_module("test_foo", test_foo),
        )

    @pytest.fixture
    def contest_with_teardown_fixture(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture
            def fail_teardown():
                yield
                assert False
        """
        )

    @pytest.fixture
    def many_files(self, pytester: Pytester, contest_with_teardown_fixture) -> None:
        pytester.makepyfile(
            test_bar="""
                import pytest
                @pytest.mark.parametrize('i', range(5))
                def test_bar(fail_teardown, i):
                    pass
            """,
            test_foo="""
                import pytest
                @pytest.mark.parametrize('i', range(15))
                def test_foo(fail_teardown, i):
                    pass
            """,
        )

    def test_teardown_simple(self, tmp_path: Path, teardown_fixture_plugin) -> None:
        def test_foo(fail_teardown):
            pass

        record = run_tests(
            test_foo,
            spec=ConfigSpec(
                rootpath=tmp_path, extra_plugins=(teardown_fixture_plugin,)
            ),
            name="test_teardown_simple",
            capture_output=True,
        )
        record.stdout.re_match_lines([r"test_teardown_simple.py \.E\s+\[100%\]"])
        # `assertoutcome`-style categories: the teardown failure is an error.
        record.assert_outcomes(passed=1, errors=1)

    def test_teardown_with_test_also_failing(
        self, tmp_path: Path, teardown_fixture_plugin
    ) -> None:
        def test_foo(fail_teardown):
            assert 0

        record = run_tests(
            test_foo,
            spec=ConfigSpec(
                rootpath=tmp_path,
                args=("-rfE",),
                extra_plugins=(teardown_fixture_plugin,),
            ),
            name="test_teardown_with_test_also_failing",
            capture_output=True,
        )
        record.stdout.re_match_lines(
            [
                r"test_teardown_with_test_also_failing.py FE\s+\[100%\]",
                "FAILED test_teardown_with_test_also_failing.py::test_foo - assert 0",
                "ERROR test_teardown_with_test_also_failing.py::test_foo - assert False",
            ]
        )
        record.assert_outcomes(failed=1, errors=1)

    def test_teardown_many(
        self, tmp_path: Path, many_sources, teardown_fixture_plugin
    ) -> None:
        record = run_tests(
            *many_sources,
            spec=ConfigSpec(
                rootpath=tmp_path, extra_plugins=(teardown_fixture_plugin,)
            ),
            capture_output=True,
        )
        record.stdout.re_match_lines(
            [r"test_bar.py (\.E){5}\s+\[ 25%\]", r"test_foo.py (\.E){15}\s+\[100%\]"]
        )
        record.assert_outcomes(passed=20, errors=20)

    def test_teardown_many_verbose(
        self, tmp_path: Path, many_sources, teardown_fixture_plugin, color_mapping
    ) -> None:
        record = run_tests(
            *many_sources,
            spec=ConfigSpec(
                rootpath=tmp_path,
                args=("-v",),
                extra_plugins=(teardown_fixture_plugin,),
            ),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            color_mapping.format_for_fnmatch(
                [
                    "test_bar.py::test_bar[0] PASSED  * [  5%]",
                    "test_bar.py::test_bar[0] ERROR   * [  5%]",
                    "test_bar.py::test_bar[4] PASSED  * [ 25%]",
                    "test_foo.py::test_foo[14] PASSED * [100%]",
                    "test_foo.py::test_foo[14] ERROR  * [100%]",
                    "=* 20 passed, 20 errors in *",
                ]
            )
        )
        record.assert_outcomes(passed=20, errors=20)

    # ensemble: runs the tests through xdist workers, i.e. subprocesses.
    def test_xdist_normal(self, many_files, pytester: Pytester, monkeypatch) -> None:
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        output = pytester.runpytest("-n2")
        output.stdout.re_match_lines([r"[\.E]{40} \s+ \[100%\]"])


def test_skip_reasons_folding() -> None:
    path = "xyz"
    lineno = 3
    message = "justso"
    longrepr = (path, lineno, message)

    class X:
        pass

    ev1 = cast(CollectReport, X())
    ev1.when = "execute"
    ev1.skipped = True  # type: ignore[misc]
    ev1.longrepr = longrepr

    ev2 = cast(CollectReport, X())
    ev2.when = "execute"
    ev2.longrepr = longrepr
    ev2.skipped = True  # type: ignore[misc]

    # ev3 might be a collection report
    ev3 = cast(CollectReport, X())
    ev3.when = "collect"
    ev3.longrepr = longrepr
    ev3.skipped = True  # type: ignore[misc]

    values = _folded_skips(Path.cwd(), [ev1, ev2, ev3])
    assert len(values) == 1
    num, fspath, lineno_, reason = values[0]
    assert num == 3
    assert fspath == path
    assert lineno_ == lineno
    assert reason == message


def test_line_with_reprcrash(monkeypatch: MonkeyPatch) -> None:
    mocked_verbose_word = "FAILED"

    mocked_pos = "some::nodeid"

    def mock_get_pos(*args):
        return mocked_pos

    monkeypatch.setattr(_pytest.terminal, "_get_node_id_with_markup", mock_get_pos)

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class config:
        def __init__(self):
            self.option = Namespace(verbose=0)

    class rep:
        def _get_verbose_word_with_markup(self, *args):
            return mocked_verbose_word, {}

        class longrepr:
            class reprcrash:
                pass

    def check(msg, width, expected):
        class DummyTerminalWriter:
            fullwidth = width

            def markup(self, word: str, **markup: str):
                return word

        __tracebackhide__ = True
        if msg:
            rep.longrepr.reprcrash.message = msg  # type: ignore
        actual = _get_line_with_reprcrash_message(
            config(),  # type: ignore[arg-type]
            rep(),  # type: ignore[arg-type]
            DummyTerminalWriter(),  # type: ignore[arg-type]
            {},
        )

        assert actual == expected
        if actual != f"{mocked_verbose_word} {mocked_pos}":
            assert len(actual) <= width
            assert wcswidth(actual) <= width

    # AttributeError with message
    check(None, 80, "FAILED some::nodeid")

    check("msg", 80, "FAILED some::nodeid - msg")
    check("msg", 3, "FAILED some::nodeid")

    check("msg", 24, "FAILED some::nodeid")
    check("msg", 25, "FAILED some::nodeid - msg")

    check("some longer msg", 24, "FAILED some::nodeid")
    check("some longer msg", 25, "FAILED some::nodeid - ...")
    check("some longer msg", 26, "FAILED some::nodeid - s...")

    check("some\nmessage", 25, "FAILED some::nodeid - ...")
    check("some\nmessage", 26, "FAILED some::nodeid - some")
    check("some\nmessage", 80, "FAILED some::nodeid - some")

    # Test unicode safety.
    check("🉐🉐🉐🉐🉐\n2nd line", 25, "FAILED some::nodeid - ...")
    check("🉐🉐🉐🉐🉐\n2nd line", 26, "FAILED some::nodeid - ...")
    check("🉐🉐🉐🉐🉐\n2nd line", 27, "FAILED some::nodeid - 🉐...")
    check("🉐🉐🉐🉐🉐\n2nd line", 28, "FAILED some::nodeid - 🉐...")
    check("🉐🉐🉐🉐🉐\n2nd line", 29, "FAILED some::nodeid - 🉐🉐...")

    # NOTE: constructed, not sure if this is supported.
    mocked_pos = "nodeid::🉐::withunicode"
    check("🉐🉐🉐🉐🉐\n2nd line", 29, "FAILED nodeid::🉐::withunicode")
    check("🉐🉐🉐🉐🉐\n2nd line", 40, "FAILED nodeid::🉐::withunicode - 🉐🉐...")
    check("🉐🉐🉐🉐🉐\n2nd line", 41, "FAILED nodeid::🉐::withunicode - 🉐🉐...")
    check("🉐🉐🉐🉐🉐\n2nd line", 42, "FAILED nodeid::🉐::withunicode - 🉐🉐🉐...")
    check("🉐🉐🉐🉐🉐\n2nd line", 80, "FAILED nodeid::🉐::withunicode - 🉐🉐🉐🉐🉐")


# ensemble: the assertion explanation is produced by the *host* assertion
# plugin (sources in this file are rewritten by the host, and
# `_pytest.assertion.util` keeps its verbosity in a module global bound at
# host configure time), so `-vv` inside an ensemble does not un-truncate it.
def test_short_summary_with_verbose(
    monkeypatch: MonkeyPatch, pytester: Pytester
) -> None:
    """With -vv do not truncate the summary info (#11777)."""
    # On CI we also do not truncate the summary info, monkeypatch it to ensure we
    # are testing against the -vv flag on CI.
    monkeypatch.setattr(_pytest.terminal, "running_on_ci", lambda: False)

    string_length = 200
    pytester.makepyfile(
        f"""
        def test():
            s1 = "A" * {string_length}
            s2 = "B" * {string_length}
            assert s1 == s2
        """
    )

    # No -vv, summary info should be truncated.
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        [
            "*short test summary info*",
            "* assert 'AAA...",
        ],
    )

    # No truncation with -vv.
    result = pytester.runpytest("-vv")
    result.stdout.fnmatch_lines(
        [
            "*short test summary info*",
            f"*{'A' * string_length}*{'B' * string_length}'",
        ]
    )


# ensemble: assertion verbosity is host-global; see
# test_short_summary_with_verbose.
def test_full_sequence_print_with_vv(
    monkeypatch: MonkeyPatch, pytester: Pytester
) -> None:
    """Do not truncate sequences in summaries with -vv (#11777)."""
    monkeypatch.setattr(_pytest.terminal, "running_on_ci", lambda: False)

    pytester.makepyfile(
        """
        def test_len_list():
            l = list(range(10))
            assert len(l) == 9

        def test_len_dict():
            d = dict(zip(range(10), range(10)))
            assert len(d) == 9
        """
    )

    result = pytester.runpytest("-vv")
    assert result.ret == 1
    result.stdout.fnmatch_lines(
        [
            "*short test summary info*",
            f"*{list(range(10))}*",
            f"*{dict(zip(range(10), range(10), strict=True))}*",
        ]
    )


# ensemble: assertion verbosity is host-global; see
# test_short_summary_with_verbose.
def test_force_short_summary(monkeypatch: MonkeyPatch, pytester: Pytester) -> None:
    monkeypatch.setattr(_pytest.terminal, "running_on_ci", lambda: False)

    pytester.makepyfile(
        """
        def test():
            assert "a\\n" * 10 == ""
        """
    )

    result = pytester.runpytest("-vv", "--force-short-summary")
    assert result.ret == 1
    result.stdout.fnmatch_lines(
        ["*short test summary info*", "*AssertionError: assert 'a\\na\\na\\na..."]
    )


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (10.0, "10.00s"),
        (10.34, "10.34s"),
        (59.99, "59.99s"),
        (60.55, "60.55s (0:01:00)"),
        (123.55, "123.55s (0:02:03)"),
        (60 * 60 + 0.5, "3600.50s (1:00:00)"),
    ],
)
def test_format_session_duration(seconds, expected):
    from _pytest.terminal import format_session_duration

    assert format_session_duration(seconds) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (3600 * 100 - 60, " 99h 59m"),
        (31 * 60 - 1, " 30m 59s"),
        (10.1236, " 10.124s"),
        (9.1236, " 9.124s"),
        (0.1236, " 123.6ms"),
        (0.01236, " 12.36ms"),
        (0.001236, " 1.236ms"),
        (0.0001236, " 123.6us"),
        (0.00001236, " 12.36us"),
        (0.000001236, " 1.236us"),
    ],
)
def test_format_node_duration(seconds: float, expected: str) -> None:
    from _pytest.terminal import format_node_duration

    assert format_node_duration(seconds) == expected


# ensemble: asserts `! Interrupted: 1 error during collection !`, produced by
# `wrap_session`, and needs a module that fails at import.
def test_collecterror(pytester: Pytester) -> None:
    p1 = pytester.makepyfile("raise SyntaxError()")
    result = pytester.runpytest("-ra", str(p1))
    result.stdout.fnmatch_lines(
        [
            "collected 0 items / 1 error",
            "*= ERRORS =*",
            "*_ ERROR collecting test_collecterror.py _*",
            "E   SyntaxError: *",
            "*= short test summary info =*",
            "ERROR test_collecterror.py",
            "*! Interrupted: 1 error during collection !*",
            "*= 1 error in *",
        ]
    )


# ensemble: needs a module that fails at import.
def test_no_summary_collecterror(pytester: Pytester) -> None:
    p1 = pytester.makepyfile("raise SyntaxError()")
    result = pytester.runpytest("-ra", "--no-summary", str(p1))
    result.stdout.no_fnmatch_line("*= ERRORS =*")


# ensemble: asserts the `<- <string>` suffix at -vv; ensemble items always
# carry a `<- .../test_terminal.py` suffix instead.
def test_via_exec(pytester: Pytester) -> None:
    p1 = pytester.makepyfile("exec('def test_via_exec(): pass')")
    result = pytester.runpytest(str(p1), "-vv")
    result.stdout.fnmatch_lines(
        ["test_via_exec.py::test_via_exec <- <string> PASSED*", "*= 1 passed in *"]
    )


class TestCodeHighlight:
    # ensemble: the rendering asserted here is of the source line
    # `assert 1 == 10`, character for character. As real code in this file that
    # line needs a `# type: ignore[comparison-overlap]` (the suite runs mypy
    # with strict_equality), and the trailing comment becomes part of the
    # highlighted line - so the port would have to weaken the pattern.
    def test_code_highlight_simple(self, pytester: Pytester, color_mapping) -> None:
        pytester.makepyfile(
            """
            def test_foo():
                assert 1 == 10
        """
        )
        result = pytester.runpytest("--color=yes")
        result.stdout.fnmatch_lines(
            color_mapping.format_for_fnmatch(
                [
                    "    {reset}{kw}def{hl-reset}{kwspace}{function}test_foo{hl-reset}():{endline}",
                    ">       {kw}assert{hl-reset} {number}1{hl-reset} == {number}10{hl-reset}{endline}",
                    "{bold}{red}E       assert 1 == 10{reset}",
                ]
            )
        )

    # ensemble: the source under test is a `print('''...'''); assert 0`
    # one-liner whose exact layout is the point; as real code in this file the
    # formatter would rewrite it and the expected highlighting with it.
    def test_code_highlight_continuation(
        self, pytester: Pytester, color_mapping
    ) -> None:
        pytester.makepyfile(
            """
            def test_foo():
                print('''
                '''); assert 0
        """
        )
        result = pytester.runpytest("--color=yes")

        result.stdout.fnmatch_lines(
            color_mapping.format_for_fnmatch(
                [
                    "    {reset}{kw}def{hl-reset}{kwspace}{function}test_foo{hl-reset}():{endline}",
                    "        {print}print{hl-reset}({str}'''{hl-reset}{str}{hl-reset}",
                    ">   {str}    {hl-reset}{str}'''{hl-reset}); {kw}assert{hl-reset} {number}0{hl-reset}{endline}",
                    "{bold}{red}E       assert 0{reset}",
                ]
            )
        )

    # ensemble: see test_code_highlight_simple.
    def test_code_highlight_custom_theme(
        self, pytester: Pytester, color_mapping, monkeypatch: MonkeyPatch
    ) -> None:
        pytester.makepyfile(
            """
            def test_foo():
                assert 1 == 10
        """
        )
        monkeypatch.setenv("PYTEST_THEME", "solarized-dark")
        monkeypatch.setenv("PYTEST_THEME_MODE", "dark")
        result = pytester.runpytest("--color=yes")
        result.stdout.fnmatch_lines(
            color_mapping.format_for_fnmatch(
                [
                    "    {reset}{kw}def{hl-reset}{kwspace}{function}test_foo{hl-reset}():{endline}",
                    ">       {kw}assert{hl-reset} {number}1{hl-reset} == {number}10{hl-reset}{endline}",
                    "{bold}{red}E       assert 1 == 10{reset}",
                ]
            )
        )

    # ensemble: asserts a startup error written to stderr by a subprocess.
    def test_code_highlight_invalid_theme(
        self, pytester: Pytester, color_mapping, monkeypatch: MonkeyPatch
    ) -> None:
        pytester.makepyfile(
            """
            def test_foo():
                assert 1 == 10
        """
        )
        monkeypatch.setenv("PYTEST_THEME", "invalid")
        result = pytester.runpytest_subprocess("--color=yes")
        result.stderr.fnmatch_lines(
            "ERROR: PYTEST_THEME environment variable has an invalid value: 'invalid'. "
            "Hint: See available pygments styles with `pygmentize -L styles`."
        )

    # ensemble: asserts a startup error written to stderr by a subprocess.
    def test_code_highlight_invalid_theme_mode(
        self, pytester: Pytester, color_mapping, monkeypatch: MonkeyPatch
    ) -> None:
        pytester.makepyfile(
            """
            def test_foo():
                assert 1 == 10
        """
        )
        monkeypatch.setenv("PYTEST_THEME_MODE", "invalid")
        result = pytester.runpytest_subprocess("--color=yes")
        result.stderr.fnmatch_lines(
            "ERROR: PYTEST_THEME_MODE environment variable has an invalid value: 'invalid'. "
            "The allowed values are 'dark' (default) and 'light'."
        )


def test_raw_skip_reason_skipped() -> None:
    report = SimpleNamespace()
    report.skipped = True
    report.longrepr = ("xyz", 3, "Skipped: Just so")

    reason = _get_raw_skip_reason(cast(TestReport, report))
    assert reason == "Just so"


def test_raw_skip_reason_xfail() -> None:
    report = SimpleNamespace()
    report.wasxfail = "reason: To everything there is a season"

    reason = _get_raw_skip_reason(cast(TestReport, report))
    assert reason == "To everything there is a season"


def test_format_trimmed() -> None:
    msg = "unconditional skip"

    assert _format_trimmed(" ({}) ", msg, len(msg) + 4) == " (unconditional skip) "
    assert _format_trimmed(" ({}) ", msg, len(msg) + 3) == " (unconditional ...) "


# ensemble: asserts a `configfile:` header line; ensemble configs never read
# a config file.
def test_warning_when_init_trumps_pyproject_toml(
    pytester: Pytester, monkeypatch: MonkeyPatch
) -> None:
    """Regression test for #7814."""
    tests = pytester.path.joinpath("tests")
    tests.mkdir()
    pytester.makepyprojecttoml(
        f"""
        [tool.pytest.ini_options]
        testpaths = ['{tests}']
    """
    )
    pytester.makefile(".ini", pytest="")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        [
            "configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)",
        ]
    )


# ensemble: asserts a `configfile:` header line.
def test_warning_when_init_trumps_multiple_files(
    pytester: Pytester, monkeypatch: MonkeyPatch
) -> None:
    """Regression test for #7814."""
    tests = pytester.path.joinpath("tests")
    tests.mkdir()
    pytester.makepyprojecttoml(
        f"""
        [tool.pytest.ini_options]
        testpaths = ['{tests}']
    """
    )
    pytester.makefile(".ini", pytest="")
    pytester.makeini(
        """
        # tox.ini
        [pytest]
        minversion = 6.0
        addopts = -ra -q
        testpaths =
            tests
            integration
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        [
            "configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml, tox.ini!)",
        ]
    )


# ensemble: asserts a `configfile:` header line.
def test_no_warning_when_init_but_pyproject_toml_has_no_entry(
    pytester: Pytester, monkeypatch: MonkeyPatch
) -> None:
    """Regression test for #7814."""
    tests = pytester.path.joinpath("tests")
    tests.mkdir()
    pytester.makepyprojecttoml(
        f"""
        [tool]
        testpaths = ['{tests}']
    """
    )
    pytester.makefile(".ini", pytest="")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        [
            "configfile: pytest.ini",
        ]
    )


# ensemble: asserts a `configfile:` header line.
def test_no_warning_on_terminal_with_a_single_config_file(
    pytester: Pytester, monkeypatch: MonkeyPatch
) -> None:
    """Regression test for #7814."""
    tests = pytester.path.joinpath("tests")
    tests.mkdir()
    pytester.makepyprojecttoml(
        f"""
        [tool.pytest.ini_options]
        testpaths = ['{tests}']
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        [
            "configfile: pyproject.toml",
        ]
    )


class TestFineGrainedTestCase:
    DEFAULT_FILE_CONTENTS = """
            import pytest

            @pytest.mark.parametrize("i", range(4))
            def test_ok(i):
                '''
                some docstring
                '''
                pass

            def test_fail():
                assert False
            """
    LONG_SKIP_FILE_CONTENTS = """
            import pytest

            @pytest.mark.skip(
              "some long skip reason that will not fit on a single line with other content that goes"
              " on and on and on and on and on"
            )
            def test_skip():
                pass
            """

    @staticmethod
    def _default_module(name: str) -> ModuleType:
        """The in-memory equivalent of DEFAULT_FILE_CONTENTS."""

        @pytest.mark.parametrize("i", range(4))
        def test_ok(i):
            """
            some docstring
            """  # noqa: D200, D403

        def test_fail():
            assert False

        return build_module(name, test_ok, test_fail)

    @staticmethod
    def _long_skip_module(name: str) -> ModuleType:
        """The in-memory equivalent of LONG_SKIP_FILE_CONTENTS."""

        @pytest.mark.skip(
            "some long skip reason that will not fit on a single line with other content that goes"
            " on and on and on and on and on"
        )
        def test_skip():
            pass

        return build_module(name, test_skip)

    @staticmethod
    def _run(
        module: ModuleType, tmp_path: Path, verbosity: int, *args: str
    ) -> RunRecord:
        """Run *module* with ``verbosity_test_cases`` set, capturing output."""
        spec = ConfigSpec(
            rootpath=tmp_path,
            args=args,
            inicfg={"verbosity_test_cases": str(verbosity)},
        )
        return run_tests(module, spec=spec, capture_output=True)

    @pytest.mark.parametrize("verbosity", [1, 2])
    def test_execute_positive(
        self, verbosity, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        # expected: one test case per line (with file name), word describing result
        # The column layout is the point, so the width must not be the host's.
        monkeypatch.setenv("COLUMNS", "80")
        name = "test_execute_positive.py"
        record = self._run(self._default_module(name[:-3]), tmp_path, verbosity)

        record.stdout.fnmatch_lines(
            [
                "collected 5 items",
                "",
                f"{name}::test_ok[0] PASSED                              [ 20%]",
                f"{name}::test_ok[1] PASSED                              [ 40%]",
                f"{name}::test_ok[2] PASSED                              [ 60%]",
                f"{name}::test_ok[3] PASSED                              [ 80%]",
                f"{name}::test_fail FAILED                               [100%]",
            ],
            consecutive=True,
        )
        record.assert_outcomes(passed=4, failed=1)

    def test_execute_0_global_1(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        # expected: one file name per line, single character describing result
        monkeypatch.setenv("COLUMNS", "80")
        name = "test_execute_0_global_1.py"
        record = self._run(self._default_module(name[:-3]), tmp_path, 0, "-v")

        record.stdout.fnmatch_lines(
            [
                "collecting ... collected 5 items",
                "",
                f"{name} ....F                                         [100%]",
            ],
            consecutive=True,
        )
        record.assert_outcomes(passed=4, failed=1)

    @pytest.mark.parametrize("verbosity", [-1, -2])
    def test_execute_negative(
        self, verbosity, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        # expected: single character describing result
        monkeypatch.setenv("COLUMNS", "80")
        name = "test_execute_negative.py"
        record = self._run(self._default_module(name[:-3]), tmp_path, verbosity)

        record.stdout.fnmatch_lines(
            [
                "collected 5 items",
                "....F                                                                    [100%]",
            ],
            consecutive=True,
        )
        record.assert_outcomes(passed=4, failed=1)

    def test_execute_skipped_positive_2(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        # expected: one test case per line (with file name), word describing result, full reason
        monkeypatch.setenv("COLUMNS", "80")
        name = "test_execute_skipped_positive_2.py"
        record = self._run(self._long_skip_module(name[:-3]), tmp_path, 2)

        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "",
                f"{name}::test_skip SKIPPED (some long skip",
                "reason that will not fit on a single line with other content that goes",
                "on and on and on and on and on)                                          [100%]",
            ],
            consecutive=True,
        )
        record.assert_outcomes(skipped=1)

    def test_execute_skipped_positive_1(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        # expected: one test case per line (with file name), word describing result, reason truncated
        monkeypatch.setenv("COLUMNS", "80")
        name = "test_execute_skipped_positive_1.py"
        record = self._run(self._long_skip_module(name[:-3]), tmp_path, 1)

        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "",
                f"{name}::test_skip SKIPPED (some long ski...) [100%]",
            ],
            consecutive=True,
        )
        record.assert_outcomes(skipped=1)

    def test_execute_skipped__0_global_1(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        # expected: one file name per line, single character describing result (no reason)
        monkeypatch.setenv("COLUMNS", "80")
        name = "test_execute_skipped__0_global_1.py"
        record = self._run(self._long_skip_module(name[:-3]), tmp_path, 0, "-v")

        record.stdout.fnmatch_lines(
            [
                "collecting ... collected 1 item",
                "",
                f"{name} s                                    [100%]",
            ],
            consecutive=True,
        )
        record.assert_outcomes(skipped=1)

    @pytest.mark.parametrize("verbosity", [-1, -2])
    def test_execute_skipped_negative(
        self, verbosity, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        # expected: single character describing result (no reason)
        monkeypatch.setenv("COLUMNS", "80")
        name = "test_execute_skipped_negative.py"
        record = self._run(self._long_skip_module(name[:-3]), tmp_path, verbosity)

        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "s                                                                        [100%]",
            ],
            consecutive=True,
        )
        record.assert_outcomes(skipped=1)

    # ensemble: every test below asserts on `--collect-only` rendering, which
    # is served from `pytest_cmdline_main`; an ensemble runs neither that hook
    # nor the `<Dir ...>` node the rendering starts from.
    @pytest.mark.parametrize("verbosity", [1, 2])
    def test__collect_only_positive(self, verbosity, pytester: Pytester) -> None:
        p = TestFineGrainedTestCase._initialize_files(pytester, verbosity=verbosity)
        result = pytester.runpytest("--collect-only", p)

        result.stdout.fnmatch_lines(
            [
                "collected 5 items",
                "",
                f"<Dir {p.parent.name}>",
                f"  <Module {p.name}>",
                "    <Function test_ok[0]>",
                "      some docstring",
                "    <Function test_ok[1]>",
                "      some docstring",
                "    <Function test_ok[2]>",
                "      some docstring",
                "    <Function test_ok[3]>",
                "      some docstring",
                "    <Function test_fail>",
            ],
            consecutive=True,
        )

    def test_collect_only_0_global_1(self, pytester: Pytester) -> None:
        p = TestFineGrainedTestCase._initialize_files(pytester, verbosity=0)
        result = pytester.runpytest("-v", "--collect-only", p)

        result.stdout.fnmatch_lines(
            [
                "collecting ... collected 5 items",
                "",
                f"<Dir {p.parent.name}>",
                f"  <Module {p.name}>",
                "    <Function test_ok[0]>",
                "    <Function test_ok[1]>",
                "    <Function test_ok[2]>",
                "    <Function test_ok[3]>",
                "    <Function test_fail>",
            ],
            consecutive=True,
        )

    def test_collect_only_negative_1(self, pytester: Pytester) -> None:
        p = TestFineGrainedTestCase._initialize_files(pytester, verbosity=-1)
        result = pytester.runpytest("--collect-only", p)

        result.stdout.fnmatch_lines(
            [
                "collected 5 items",
                "",
                f"{p.name}::test_ok[0]",
                f"{p.name}::test_ok[1]",
                f"{p.name}::test_ok[2]",
                f"{p.name}::test_ok[3]",
                f"{p.name}::test_fail",
            ],
            consecutive=True,
        )

    def test_collect_only_negative_2(self, pytester: Pytester) -> None:
        p = TestFineGrainedTestCase._initialize_files(pytester, verbosity=-2)
        result = pytester.runpytest("--collect-only", p)

        result.stdout.fnmatch_lines(
            [
                "collected 5 items",
                "",
                f"{p.name}: 5",
            ],
            consecutive=True,
        )

    @staticmethod
    def _initialize_files(
        pytester: Pytester, verbosity: int, file_contents: str = DEFAULT_FILE_CONTENTS
    ) -> Path:
        p = pytester.makepyfile(file_contents)
        pytester.makeini(
            f"""
            [pytest]
            verbosity_test_cases = {verbosity}
            """
        )
        return p


def test_summary_xfail_reason(tmp_path: Path) -> None:
    @pytest.mark.xfail
    def test_xfail():
        assert False

    @pytest.mark.xfail(reason="foo")
    def test_xfail_reason():
        assert False

    record = run_tests(
        test_xfail,
        test_xfail_reason,
        spec=ConfigSpec(rootpath=tmp_path, args=("-rx",)),
        name="test_summary_xfail_reason",
        capture_output=True,
    )
    expect1 = "XFAIL test_summary_xfail_reason.py::test_xfail"
    expect2 = "XFAIL test_summary_xfail_reason.py::test_xfail_reason - foo"
    record.stdout.fnmatch_lines([expect1, expect2])
    lines = record.output.splitlines()
    assert lines.count(expect1) == 1
    assert lines.count(expect2) == 1


@pytest.fixture()
def xfail_testsources() -> tuple[object, ...]:
    def test_fail():
        a, b = 1, 2
        assert a == b

    @pytest.mark.xfail
    def test_xfail():
        c, d = 3, 4
        assert c == d

    return (test_fail, test_xfail)


@pytest.fixture()
def xfail_testfile(pytester: Pytester) -> Path:
    return pytester.makepyfile(
        """
        import pytest

        def test_fail():
            a, b = 1, 2
            assert a == b

        @pytest.mark.xfail
        def test_xfail():
            c, d = 3, 4
            assert c == d
        """
    )


def test_xfail_tb_default(xfail_testsources, tmp_path: Path) -> None:
    record = run_tests(
        *xfail_testsources, rootpath=tmp_path, name="test_xfail_tb", capture_output=True
    )

    # test_fail, show traceback
    record.stdout.fnmatch_lines(
        [
            "*= FAILURES =*",
            "*_ test_fail _*",
            "*def test_fail():*",
            "*        a, b = 1, 2*",
            "*>       assert a == b*",
            "*E       assert 1 == 2*",
        ]
    )

    # test_xfail, don't show traceback
    record.stdout.no_fnmatch_line("*= XFAILURES =*")
    record.assert_outcomes(failed=1, xfailed=1)


def test_xfail_tb_true(xfail_testsources, tmp_path: Path) -> None:
    record = run_tests(
        *xfail_testsources,
        spec=ConfigSpec(rootpath=tmp_path, args=("--xfail-tb",)),
        name="test_xfail_tb",
        capture_output=True,
    )

    # both test_fail and test_xfail, show traceback
    record.stdout.fnmatch_lines(
        [
            "*= FAILURES =*",
            "*_ test_fail _*",
            "*def test_fail():*",
            "*        a, b = 1, 2*",
            "*>       assert a == b*",
            "*E       assert 1 == 2*",
            "*= XFAILURES =*",
            "*_ test_xfail _*",
            "*def test_xfail():*",
            "*        c, d = 3, 4*",
            "*>       assert c == d*",
            "*E       assert 3 == 4*",
            "*short test summary info*",
        ]
    )
    record.assert_outcomes(failed=1, xfailed=1)


# ensemble: `--tb=line` renders `<file>:<lineno>: <message>`, and an ensemble
# item's file:line is the host `test_terminal.py`.
def test_xfail_tb_line(xfail_testfile, pytester: Pytester) -> None:
    result = pytester.runpytest(xfail_testfile, "--xfail-tb", "--tb=line")

    # both test_fail and test_xfail, show line
    result.stdout.fnmatch_lines(
        [
            "*= FAILURES =*",
            "*test_xfail_tb_line.py:5: assert 1 == 2",
            "*= XFAILURES =*",
            "*test_xfail_tb_line.py:10: assert 3 == 4",
            "*short test summary info*",
        ]
    )


def test_summary_xpass_reason(tmp_path: Path) -> None:
    @pytest.mark.xfail
    def test_pass(): ...

    @pytest.mark.xfail(reason="foo")
    def test_reason(): ...

    record = run_tests(
        test_pass,
        test_reason,
        spec=ConfigSpec(rootpath=tmp_path, args=("-rX",)),
        name="test_summary_xpass_reason",
        capture_output=True,
    )
    expect1 = "XPASS test_summary_xpass_reason.py::test_pass"
    expect2 = "XPASS test_summary_xpass_reason.py::test_reason - foo"
    record.stdout.fnmatch_lines([expect1, expect2])
    lines = record.output.splitlines()
    assert lines.count(expect1) == 1
    assert lines.count(expect2) == 1


# ensemble: asserts a `Captured stdout call` section, which needs capture.
def test_xpass_output(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.xfail
        def test_pass():
            print('hi there')
        """
    )
    result = pytester.runpytest("-rX")
    result.stdout.fnmatch_lines(
        [
            "*= XPASSES =*",
            "*_ test_pass _*",
            "*- Captured stdout call -*",
            "*= short test summary info =*",
            "XPASS test_xpass_output.py::test_pass*",
            "*= 1 xpassed in * =*",
        ]
    )


class TestNodeIDHandling:
    # ensemble: the point is how nodeids are rendered relative to a rootdir
    # that differs from the invocation dir, which needs a real directory tree.
    def test_nodeid_handling_windows_paths(self, pytester: Pytester, tmp_path) -> None:
        """Test the correct handling of Windows-style paths with backslashes."""
        pytester.makeini("[pytest]")  # Change `config.rootpath`

        test_path = pytester.path / "tests" / "test_foo.py"
        test_path.parent.mkdir()
        os.chdir(test_path.parent)  # Change `config.invocation_params.dir`

        test_path.write_text(
            textwrap.dedent(
                """
                import pytest

                @pytest.mark.parametrize("a", ["x/y", "C:/path", "\\\\", "C:\\\\path", "a::b/"])
                def test_x(a):
                    assert False
                """
            ),
            encoding="utf-8",
        )

        result = pytester.runpytest("-v")

        result.stdout.re_match_lines(
            [
                r".*test_foo.py::test_x\[x/y\] .*FAILED.*",
                r".*test_foo.py::test_x\[C:/path\] .*FAILED.*",
                r".*test_foo.py::test_x\[\\\\\] .*FAILED.*",
                r".*test_foo.py::test_x\[C:\\\\path\] .*FAILED.*",
                r".*test_foo.py::test_x\[a::b/\] .*FAILED.*",
            ]
        )


class TestTerminalProgressPlugin:
    """Tests for the TerminalProgressPlugin."""

    @pytest.fixture
    def mock_file(self) -> StringIO:
        return StringIO()

    @pytest.fixture
    def mock_tr(self, mock_file: StringIO) -> pytest.TerminalReporter:
        tr: pytest.TerminalReporter = mock.create_autospec(pytest.TerminalReporter)

        def write_raw(content: str, *, flush: bool = False) -> None:
            mock_file.write(content)

        tr.write_raw = write_raw  # type: ignore[method-assign]
        tr._progress_nodeids_reported = set()
        return tr

    # ensemble: the plugin decides whether to register from the terminal
    # reporter's file being a tty; an ensemble's file is a private buffer, so
    # monkeypatching `sys.stdout.isatty` would not reach it.
    @pytest.mark.skipif(sys.platform != "win32", reason="#13896")
    def test_plugin_registration_enabled_by_default(
        self, pytester: pytest.Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that the plugin registration is enabled by default.

        Currently only on Windows (#13896).
        """
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        # The plugin module should be registered as a default plugin.
        config = pytester.parseconfigure()
        plugin = config.pluginmanager.get_plugin("terminalprogress")
        assert plugin is not None

    # ensemble: see test_plugin_registration_enabled_by_default.
    def test_plugin_registred_on_all_platforms_when_explicitly_requested(
        self, pytester: pytest.Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that the plugin is registered on any platform if explicitly requested."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        # The plugin module should be registered as a default plugin.
        config = pytester.parseconfigure("-p", "terminalprogress")
        plugin = config.pluginmanager.get_plugin("terminalprogress")
        assert plugin is not None

    # ensemble: see test_plugin_registration_enabled_by_default.
    def test_disabled_for_non_tty(
        self, pytester: pytest.Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that plugin is disabled for non-TTY output."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        config = pytester.parseconfigure("-p", "terminalprogress")
        plugin = config.pluginmanager.get_plugin("terminalprogress-plugin")
        assert plugin is None

    # ensemble: see test_plugin_registration_enabled_by_default.
    def test_disabled_for_dumb_terminal(
        self, pytester: pytest.Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that plugin is disabled when TERM=dumb."""
        monkeypatch.setenv("TERM", "dumb")
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        config = pytester.parseconfigure("-p", "terminalprogress")
        plugin = config.pluginmanager.get_plugin("terminalprogress-plugin")
        assert plugin is None

    @pytest.mark.parametrize(
        ["state", "progress", "expected"],
        [
            ("indeterminate", None, "\x1b]9;4;3;\x1b\\"),
            ("normal", 50, "\x1b]9;4;1;50\x1b\\"),
            ("error", 75, "\x1b]9;4;2;75\x1b\\"),
            ("paused", None, "\x1b]9;4;4;\x1b\\"),
            ("paused", 80, "\x1b]9;4;4;80\x1b\\"),
            ("remove", None, "\x1b]9;4;0;\x1b\\"),
        ],
    )
    def test_emit_progress_sequences(
        self,
        mock_file: StringIO,
        mock_tr: pytest.TerminalReporter,
        state: Literal["remove", "normal", "error", "indeterminate", "paused"],
        progress: int | None,
        expected: str,
    ) -> None:
        """Test that progress sequences are emitted correctly."""
        plugin = TerminalProgressPlugin(mock_tr)
        plugin._emit_progress(state, progress)
        assert expected in mock_file.getvalue()

    def test_session_lifecycle(
        self, mock_file: StringIO, mock_tr: pytest.TerminalReporter
    ) -> None:
        """Test progress updates during session lifecycle."""
        plugin = TerminalProgressPlugin(mock_tr)

        session = mock.create_autospec(pytest.Session)
        session.testscollected = 3

        # Session start - should emit indeterminate progress.
        plugin.pytest_sessionstart(session)
        assert "\x1b]9;4;3;\x1b\\" in mock_file.getvalue()
        mock_file.truncate(0)
        mock_file.seek(0)

        # Collection finish - should emit 0% progress.
        plugin.pytest_collection_finish()
        assert "\x1b]9;4;1;0\x1b\\" in mock_file.getvalue()
        mock_file.truncate(0)
        mock_file.seek(0)

        # First test - 33% progress.
        report1 = pytest.TestReport(
            nodeid="test_1",
            location=("test.py", 0, "test_1"),
            when="call",
            outcome="passed",
            keywords={},
            longrepr=None,
        )
        mock_tr.reported_progress = 1  # type: ignore[misc]
        plugin.pytest_runtest_logreport(report1)
        assert "\x1b]9;4;1;33\x1b\\" in mock_file.getvalue()
        mock_file.truncate(0)
        mock_file.seek(0)

        # Second test with failure - 66% in error state.
        report2 = pytest.TestReport(
            nodeid="test_2",
            location=("test.py", 1, "test_2"),
            when="call",
            outcome="failed",
            keywords={},
            longrepr=None,
        )
        mock_tr.reported_progress = 2  # type: ignore[misc]
        plugin.pytest_runtest_logreport(report2)
        assert "\x1b]9;4;2;66\x1b\\" in mock_file.getvalue()
        mock_file.truncate(0)
        mock_file.seek(0)

        # Session finish - should remove progress.
        plugin.pytest_sessionfinish()
        assert "\x1b]9;4;0;\x1b\\" in mock_file.getvalue()

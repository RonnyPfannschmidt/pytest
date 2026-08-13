from __future__ import annotations

from enum import Enum
import json
import logging
from pathlib import Path
import sys
import types
from typing import Literal
import unittest

from _pytest._io.saferepr import saferepr
from _pytest.ensemble import build_module
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import run_tests
from _pytest.ensemble import Source
from _pytest.outcomes import Exit
from _pytest.reports import TestReport
from _pytest.subtests import SubtestContext
from _pytest.subtests import SubtestReport
import pytest


IS_PY311 = sys.version_info[:2] >= (3, 11)


def _subtests_spec(tmp_path: Path, *args: str, **inicfg: str) -> ConfigSpec:
    """A :class:`ConfigSpec` for an ensemble exercising subtests.

    ``subtests`` is not one of the ensemble default plugins, so it has to be
    asked for explicitly.
    """
    return ConfigSpec(rootpath=tmp_path, args=args, inicfg=inicfg).with_plugins(
        "subtests"
    )


def _rendering_spec(tmp_path: Path, *args: str, **inicfg: str) -> ConfigSpec:
    """As :func:`_subtests_spec`, for ensembles whose *output* is the subject.

    Only usable together with ``capture_output=True``, which is what pulls in
    the terminal plugin these settings belong to. The console output style is
    bumped because the terminal reporter draws the progress percentages only
    when it believes output is being captured - which an ensemble's terminal,
    having no capture manager of its own, never is.
    """
    return _subtests_spec(
        tmp_path,
        *args,
        console_output_style="progress-even-when-capture-no",
        **inicfg,
    )


def _failure_sources() -> types.ModuleType:
    """The module under test of ``test_failures``."""

    def test_foo(subtests: pytest.Subtests) -> None:
        with subtests.test("foo subtest"):
            assert False, "foo subtest failure"

    def test_bar(subtests: pytest.Subtests) -> None:
        with subtests.test("bar subtest"):
            assert False, "bar subtest failure"
        assert False, "test_bar also failed"

    def test_zaz(subtests: pytest.Subtests) -> None:
        with subtests.test("zaz subtest"):
            pass

    return build_module("test_failures", test_foo, test_bar, test_zaz)


def test_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "120")
    summary_lines = [
        "*=== FAILURES ===*",
        #
        "*___ test_foo [[]foo subtest[]] ___*",
        "*AssertionError: foo subtest failure",
        #
        "*___ test_foo ___*",
        "contains 1 failed subtest",
        #
        "*___ test_bar [[]bar subtest[]] ___*",
        "*AssertionError: bar subtest failure",
        #
        "*___ test_bar ___*",
        "*AssertionError: test_bar also failed",
        #
        "*=== short test summary info ===*",
        "SUBFAILED[[]foo subtest[]] test_*.py::test_foo - AssertionError*",
        "FAILED test_*.py::test_foo - contains 1 failed subtest",
        "SUBFAILED[[]bar subtest[]] test_*.py::test_bar - AssertionError*",
        "FAILED test_*.py::test_bar - AssertionError*",
    ]
    record = run_tests(
        _failure_sources(), spec=_rendering_spec(tmp_path), capture_output=True
    )
    record.stdout.fnmatch_lines(
        [
            # The original also matched a trailing "[100%]" here. The terminal
            # reporter defers that final fill to ``pytest_runtestloop``, which
            # an ensemble never calls - it drives the items directly. The
            # per-test letters, which are what this test is about, are intact.
            "test_*.py uFuF.",
            *summary_lines,
            "* 4 failed, 1 passed in *",
        ]
    )
    record.assert_outcomes(failed=4, passed=1)

    record = run_tests(
        _failure_sources(), spec=_rendering_spec(tmp_path, "-v"), capture_output=True
    )
    record.stdout.fnmatch_lines(
        [
            "test_*.py::test_foo SUBFAILED[[]foo subtest[]]    *     [[] 33%[]]",
            "test_*.py::test_foo FAILED                        *     [[] 33%[]]",
            "test_*.py::test_bar SUBFAILED[[]bar subtest[]]    *     [[] 66%[]]",
            "test_*.py::test_bar FAILED                        *     [[] 66%[]]",
            "test_*.py::test_zaz SUBPASSED[[]zaz subtest[]]    *     [[]100%[]]",
            "test_*.py::test_zaz PASSED                        *     [[]100%[]]",
            *summary_lines,
            "* 4 failed, 1 passed, 1 subtests passed in *",
        ]
    )
    # "subtests passed" is a terminal category of its own, which
    # assert_outcomes() (like RunResult.assert_outcomes) does not know about.
    assert record.outcomes()["subtests passed"] == 1

    record = run_tests(
        _failure_sources(),
        spec=_rendering_spec(tmp_path, "-v", verbosity_subtests="0"),
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "test_*.py::test_foo SUBFAILED[[]foo subtest[]]    *     [[] 33%[]]",
            "test_*.py::test_foo FAILED                        *     [[] 33%[]]",
            "test_*.py::test_bar SUBFAILED[[]bar subtest[]]    *     [[] 66%[]]",
            "test_*.py::test_bar FAILED                        *     [[] 66%[]]",
            "test_*.py::test_zaz PASSED                        *     [[]100%[]]",
            *summary_lines,
            "* 4 failed, 1 passed in *",
        ]
    )
    record.stdout.no_fnmatch_line("test_*.py::test_zaz SUBPASSED[[]zaz subtest[]]*")
    assert "subtests passed" not in record.outcomes()


def _passing_sources() -> types.ModuleType:
    """The module under test of ``test_passes``."""

    def test_foo(subtests: pytest.Subtests) -> None:
        with subtests.test("foo subtest"):
            pass

    def test_bar(subtests: pytest.Subtests) -> None:
        with subtests.test("bar subtest"):
            pass

    return build_module("test_passes", test_foo, test_bar)


def test_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "120")
    record = run_tests(
        _passing_sources(), spec=_rendering_spec(tmp_path), capture_output=True
    )
    record.stdout.fnmatch_lines(
        [
            # see test_failures on the dropped "[100%]"
            "test_*.py ..",
            "* 2 passed in *",
        ]
    )
    record.assert_outcomes(passed=2)

    record = run_tests(
        _passing_sources(), spec=_rendering_spec(tmp_path, "-v"), capture_output=True
    )
    record.stdout.fnmatch_lines(
        [
            "*.py::test_foo SUBPASSED[[]foo subtest[]]      * [[] 50%[]]",
            "*.py::test_foo PASSED                          * [[] 50%[]]",
            "*.py::test_bar SUBPASSED[[]bar subtest[]]      * [[]100%[]]",
            "*.py::test_bar PASSED                          * [[]100%[]]",
            "* 2 passed, 2 subtests passed in *",
        ]
    )
    assert record.outcomes()["subtests passed"] == 2

    record = run_tests(
        _passing_sources(),
        spec=_rendering_spec(tmp_path, "-v", verbosity_subtests="0"),
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "*.py::test_foo PASSED                          * [[] 50%[]]",
            "*.py::test_bar PASSED                          * [[]100%[]]",
            "* 2 passed in *",
        ]
    )
    record.stdout.no_fnmatch_line("*.py::test_foo SUBPASSED[[]foo subtest[]]*")
    record.stdout.no_fnmatch_line("*.py::test_bar SUBPASSED[[]bar subtest[]]*")


def _skip_sources() -> types.ModuleType:
    """The module under test of ``test_skip``."""

    def test_foo(subtests: pytest.Subtests) -> None:
        with subtests.test("foo subtest"):
            pytest.skip("skip foo subtest")

    def test_bar(subtests: pytest.Subtests) -> None:
        with subtests.test("bar subtest"):
            pytest.skip("skip bar subtest")
        pytest.skip("skip test_bar")

    return build_module("test_skip", test_foo, test_bar)


def test_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "120")
    record = run_tests(
        _skip_sources(), spec=_rendering_spec(tmp_path, "-ra"), capture_output=True
    )
    record.stdout.fnmatch_lines(
        [
            # see test_failures on the dropped "[100%]"
            "test_*.py .s",
            "*=== short test summary info ===*",
            # the original spelled out "test_skip.py:9" here; the location of
            # an in-memory source is anchored in *this* file.
            "SKIPPED [[]1[]] *: skip test_bar",
            "* 1 passed, 1 skipped in *",
        ]
    )
    record.assert_outcomes(passed=1, skipped=1)

    record = run_tests(
        _skip_sources(),
        spec=_rendering_spec(tmp_path, "-v", "-ra"),
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "*.py::test_foo SUBSKIPPED[[]foo subtest[]] (skip foo subtest)  * [[] 50%[]]",
            "*.py::test_foo PASSED                                          * [[] 50%[]]",
            "*.py::test_bar SUBSKIPPED[[]bar subtest[]] (skip bar subtest)  * [[]100%[]]",
            "*.py::test_bar SKIPPED (skip test_bar)                         * [[]100%[]]",
            "*=== short test summary info ===*",
            "SUBSKIPPED[[]foo subtest[]] [[]1[]] *.py:*: skip foo subtest",
            "SUBSKIPPED[[]foo subtest[]] [[]1[]] *.py:*: skip bar subtest",
            "SUBSKIPPED[[]foo subtest[]] [[]1[]] *.py:*: skip test_bar",
            "* 1 passed, 3 skipped in *",
        ]
    )
    record.assert_outcomes(passed=1, skipped=3)

    record = run_tests(
        _skip_sources(),
        spec=_rendering_spec(tmp_path, "-v", "-ra", verbosity_subtests="0"),
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "*.py::test_foo PASSED                          * [[] 50%[]]",
            "*.py::test_bar SKIPPED (skip test_bar)         * [[]100%[]]",
            "*=== short test summary info ===*",
            "* 1 passed, 1 skipped in *",
        ]
    )
    record.stdout.no_fnmatch_line("*.py::test_foo SUBPASSED[[]foo subtest[]]*")
    record.stdout.no_fnmatch_line("*.py::test_bar SUBPASSED[[]bar subtest[]]*")
    record.stdout.no_fnmatch_line(
        "SUBSKIPPED[[]foo subtest[]] [[]1[]] *.py:*: skip foo subtest"
    )
    record.stdout.no_fnmatch_line(
        "SUBSKIPPED[[]foo subtest[]] [[]1[]] *.py:*: skip test_bar"
    )


def _xfail_sources() -> types.ModuleType:
    """The module under test of ``test_xfail``."""

    def test_foo(subtests: pytest.Subtests) -> None:
        with subtests.test("foo subtest"):
            pytest.xfail("xfail foo subtest")

    def test_bar(subtests: pytest.Subtests) -> None:
        with subtests.test("bar subtest"):
            pytest.xfail("xfail bar subtest")
        pytest.xfail("xfail test_bar")

    return build_module("test_xfail", test_foo, test_bar)


def test_xfail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "120")
    record = run_tests(
        _xfail_sources(), spec=_rendering_spec(tmp_path, "-ra"), capture_output=True
    )
    record.stdout.fnmatch_lines(
        [
            # see test_failures on the dropped "[100%]"
            "test_*.py .x",
            "*=== short test summary info ===*",
            "* 1 passed, 1 xfailed in *",
        ]
    )
    record.assert_outcomes(passed=1, xfailed=1)

    record = run_tests(
        _xfail_sources(),
        spec=_rendering_spec(tmp_path, "-v", "-ra"),
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "*.py::test_foo SUBXFAIL[[]foo subtest[]] (xfail foo subtest)    * [[] 50%[]]",
            "*.py::test_foo PASSED                                           * [[] 50%[]]",
            "*.py::test_bar SUBXFAIL[[]bar subtest[]] (xfail bar subtest)    * [[]100%[]]",
            "*.py::test_bar XFAIL (xfail test_bar)                           * [[]100%[]]",
            "*=== short test summary info ===*",
            "SUBXFAIL[[]foo subtest[]] *.py::test_foo - xfail foo subtest",
            "SUBXFAIL[[]bar subtest[]] *.py::test_bar - xfail bar subtest",
            "XFAIL *.py::test_bar - xfail test_bar",
            "* 1 passed, 3 xfailed in *",
        ]
    )
    record.assert_outcomes(passed=1, xfailed=3)

    record = run_tests(
        _xfail_sources(),
        spec=_rendering_spec(tmp_path, "-v", "-ra", verbosity_subtests="0"),
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "*.py::test_foo PASSED                          * [[] 50%[]]",
            "*.py::test_bar XFAIL (xfail test_bar)         * [[]100%[]]",
            "*=== short test summary info ===*",
            "* 1 passed, 1 xfailed in *",
        ]
    )
    record.stdout.no_fnmatch_line(
        "SUBXFAIL[[]foo subtest[]] *.py::test_foo - xfail foo subtest"
    )
    record.stdout.no_fnmatch_line(
        "SUBXFAIL[[]bar subtest[]] *.py::test_bar - xfail bar subtest"
    )


def test_typing_exported(tmp_path: Path) -> None:
    from pytest import Subtests

    def test_typing_exported(subtests: Subtests) -> None:
        assert isinstance(subtests, Subtests)

    record = run_tests(
        test_typing_exported, spec=_subtests_spec(tmp_path), name="test_typing_exported"
    )
    record.assert_outcomes(passed=1)


def _parametrized_sources() -> types.ModuleType:
    """The module under test of ``test_subtests_and_parametrization``."""

    @pytest.mark.parametrize("x", [0, 1])
    def test_foo(subtests: pytest.Subtests, x: int) -> None:
        for i in range(3):
            with subtests.test("custom", i=i):
                assert i % 2 == 0
        assert x == 0

    return build_module("test_subtests_and_parametrization", test_foo)


def test_subtests_and_parametrization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "120")
    record = run_tests(
        _parametrized_sources(),
        spec=_rendering_spec(tmp_path, "-v"),
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "*.py::test_foo[[]0[]] SUBFAILED[[]custom[]] (i=1) *[[] 50%[]]",
            "*.py::test_foo[[]0[]] FAILED                        *[[] 50%[]]",
            "*.py::test_foo[[]1[]] SUBFAILED[[]custom[]] (i=1) *[[]100%[]]",
            "*.py::test_foo[[]1[]] FAILED                        *[[]100%[]]",
            "contains 1 failed subtest",
            "* 4 failed, 4 subtests passed in *",
        ]
    )
    record.assert_outcomes(failed=4)
    assert record.outcomes()["subtests passed"] == 4

    record = run_tests(
        _parametrized_sources(),
        spec=_rendering_spec(tmp_path, "-v", verbosity_subtests="0"),
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "*.py::test_foo[[]0[]] SUBFAILED[[]custom[]] (i=1) *[[] 50%[]]",
            "*.py::test_foo[[]0[]] FAILED                        *[[] 50%[]]",
            "*.py::test_foo[[]1[]] SUBFAILED[[]custom[]] (i=1) *[[]100%[]]",
            "*.py::test_foo[[]1[]] FAILED                        *[[]100%[]]",
            "contains 1 failed subtest",
            "* 4 failed in *",
        ]
    )
    assert "subtests passed" not in record.outcomes()


def test_subtests_fail_top_level_test(tmp_path: Path) -> None:
    def test_foo(subtests: pytest.Subtests) -> None:
        for i in range(3):
            with subtests.test("custom", i=i):
                assert i % 2 == 0

    # ``-v`` is a terminal plugin option, which an ensemble only loads when it
    # is asked to capture output; the ini has the same effect on the category.
    record = run_tests(
        test_foo,
        spec=_subtests_spec(tmp_path, verbosity_subtests="1"),
        name="test_subtests_fail_top_level_test",
    )
    # the original read "* 2 failed, 2 subtests passed in *" off the summary
    record.assert_outcomes(failed=2)
    assert record.outcomes()["subtests passed"] == 2


def test_subtests_do_not_overwrite_top_level_failure(tmp_path: Path) -> None:
    def test_foo(subtests: pytest.Subtests) -> None:
        for i in range(3):
            with subtests.test("custom", i=i):
                assert i % 2 == 0
        assert False, "top-level failure"

    record = run_tests(
        test_foo,
        spec=_subtests_spec(tmp_path, verbosity_subtests="1"),
        name="test_subtests_do_not_overwrite_top_level_failure",
    )
    record.assert_outcomes(failed=2)
    assert record.outcomes()["subtests passed"] == 2
    # the top level report keeps its own failure instead of being replaced by
    # the "contains N failed subtests" one
    call = record["test_foo"].call
    assert call is not None
    assert "AssertionError: top-level failure" in call.longreprtext


def test_msg_not_a_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Using a non-string in subtests.test() should still show it in the terminal (#14195).

    Note: this was not a problem originally with the subtests fixture, only with TestCase.subTest; this test
    was added for symmetry.
    """
    monkeypatch.setenv("COLUMNS", "120")

    def test_int_msg(subtests: pytest.Subtests) -> None:
        with subtests.test(42):  # type: ignore[arg-type]
            assert False, "subtest failure"

    def test_no_msg(subtests: pytest.Subtests) -> None:
        with subtests.test():
            assert False, "subtest failure"

    module = build_module("test_msg_not_a_string", test_int_msg, test_no_msg)
    record = run_tests(module, spec=_rendering_spec(tmp_path), capture_output=True)
    record.stdout.fnmatch_lines(
        [
            "SUBFAILED[[]42[]] test_msg_not_a_string.py::test_int_msg - AssertionError: subtest failure",
            "SUBFAILED(<subtest>) test_msg_not_a_string.py::test_no_msg - AssertionError: subtest failure",
        ]
    )


@pytest.mark.parametrize("flag", ["--last-failed", "--stepwise"])
def test_subtests_last_failed_step_wise(tmp_path: Path, flag: str) -> None:
    """Check that --last-failed and --step-wise correctly rerun tests with failed subtests."""

    def test_foo(subtests: pytest.Subtests) -> None:
        for i in range(3):
            with subtests.test("custom", i=i):
                assert i % 2 == 0

    # Both flags read the cache the first run leaves behind in the shared
    # rootpath, so the two runs have to agree on it. The terminal plugin is
    # not optional here: what marks the top level test as failed is a *side
    # effect* of ``pytest_report_teststatus``, and without a terminal nothing
    # calls that hook while the run is in progress - so the last-failed cache
    # would never learn about the failure and the flag below would be a no-op.
    spec = _rendering_spec(tmp_path, "-v").with_plugins("cacheprovider", "stepwise")
    name = "test_subtests_last_failed_step_wise"
    record = run_tests(test_foo, spec=spec, name=name, capture_output=True)
    record.stdout.fnmatch_lines(["* 2 failed, 2 subtests passed in *"])
    record.assert_outcomes(failed=2)
    assert record.outcomes()["subtests passed"] == 2

    record = run_tests(
        test_foo, spec=spec.replace(args=("-v", flag)), name=name, capture_output=True
    )
    # proof the flag had the previous run's state to act on at all
    record.stdout.fnmatch_lines(
        [
            {
                "--last-failed": "run-last-failure: rerun previous 1 failure",
                # stepwise only records state when it was itself active, and
                # the first run above was a plain one - same as in the original
                "--stepwise": "stepwise: no previously failed tests, not skipping.",
            }[flag],
            "* 2 failed, 2 subtests passed in *",
        ]
    )
    record.assert_outcomes(failed=2)
    assert record.outcomes()["subtests passed"] == 2


class TestUnittestSubTest:
    """Test unittest.TestCase.subTest functionality."""

    def test_failures(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLUMNS", "120")

        class T(unittest.TestCase):
            def test_foo(self) -> None:
                with self.subTest("foo subtest"):
                    assert False, "foo subtest failure"

            def test_bar(self) -> None:
                with self.subTest("bar subtest"):
                    assert False, "bar subtest failure"
                assert False, "test_bar also failed"

            def test_zaz(self) -> None:
                with self.subTest("zaz subtest"):
                    pass

        record = run_tests(T, spec=_subtests_spec(tmp_path))
        record.assert_outcomes(failed=3, passed=2)

    def test_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLUMNS", "120")

        class T(unittest.TestCase):
            def test_foo(self) -> None:
                with self.subTest("foo subtest"):
                    pass

            def test_bar(self) -> None:
                with self.subTest("bar subtest"):
                    pass

            def test_zaz(self) -> None:
                with self.subTest("zaz subtest"):
                    pass

        record = run_tests(T, spec=_subtests_spec(tmp_path))
        record.assert_outcomes(passed=3)

    def test_skip(self, tmp_path: Path) -> None:
        class T(unittest.TestCase):
            def test_foo(self) -> None:
                for i in range(5):
                    with self.subTest(msg="custom", i=i):
                        if i % 2 == 0:
                            self.skipTest("even number")

        # This output might change #13756.
        record = run_tests(T, spec=_subtests_spec(tmp_path))
        record.assert_outcomes(passed=1)

    def test_non_subtest_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COLUMNS", "120")

        class T(unittest.TestCase):
            def test_foo(self) -> None:
                with self.subTest(msg="subtest"):
                    assert False, "failed subtest"
                self.skipTest("non-subtest skip")

        # This output might change #13756.
        record = run_tests(
            T,
            spec=_rendering_spec(tmp_path),
            name="test_non_subtest_skip",
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "SUBFAILED[[]subtest[]] test_non_subtest_skip.py::T::test_foo*",
                "* 1 failed, 1 skipped in *",
            ]
        )
        record.assert_outcomes(failed=1, skipped=1)

    def test_xfail(self, tmp_path: Path) -> None:
        class T(unittest.TestCase):
            @unittest.expectedFailure
            def test_foo(self) -> None:
                for i in range(5):
                    with self.subTest(msg="custom", i=i):
                        if i % 2 == 0:
                            raise pytest.xfail("even number")

        # This output might change #13756.
        record = run_tests(T, spec=_subtests_spec(tmp_path))
        record.assert_outcomes(xfailed=1)

    def test_only_original_skip_is_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test for pytest-dev/pytest-subtests#173."""
        monkeypatch.setenv("COLUMNS", "120")

        @unittest.skip("skip this test")
        class T(unittest.TestCase):
            def test_foo(self) -> None:
                # deliberately false; the class level skip must keep it from running
                assert 1 == 2  # type: ignore[comparison-overlap]

        record = run_tests(
            T,
            spec=_rendering_spec(tmp_path, "-v", "-rsf"),
            name="test_only_original_skip_is_called",
            capture_output=True,
        )
        # the original spelled out "test_only_original_skip_is_called.py:6"
        # here; the location of an in-memory source is anchored in *this* file.
        record.stdout.fnmatch_lines(["SKIPPED [[]1[]] *: skip this test"])
        record.assert_outcomes(skipped=1)

    def test_skip_with_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COLUMNS", "120")

        class T(unittest.TestCase):
            def test_foo(self) -> None:
                with self.subTest("subtest 1"):
                    self.skipTest("skip subtest 1")
                # skipTest() is typed NoReturn, but subTest() swallows the skip
                with self.subTest("subtest 2"):  # type: ignore[unreachable]
                    assert False, "fail subtest 2"

        name = "test_skip_with_failure"
        record = run_tests(
            T, spec=_rendering_spec(tmp_path, "-ra"), name=name, capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                # see test_failures on the dropped "[100%]"
                "*.py u.",
                "*=== short test summary info ===*",
                "SUBFAILED[[]subtest 2[]] *.py::T::test_foo - AssertionError: fail subtest 2",
                "* 1 failed, 1 passed in *",
            ]
        )
        record.assert_outcomes(failed=1, passed=1)

        record = run_tests(
            T,
            spec=_rendering_spec(tmp_path, "-v", "-ra"),
            name=name,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "*.py::T::test_foo SUBSKIPPED[[]subtest 1[]] (skip subtest 1)      *            [[]100%[]]",
                "*.py::T::test_foo SUBFAILED[[]subtest 2[]]                        *            [[]100%[]]",
                "*.py::T::test_foo PASSED                                          *            [[]100%[]]",
                "SUBSKIPPED[[]subtest 1[]] [[]1[]] *.py:*: skip subtest 1",
                "SUBFAILED[[]subtest 2[]] *.py::T::test_foo - AssertionError: fail subtest 2",
                "* 1 failed, 1 passed, 1 skipped in *",
            ]
        )
        record.assert_outcomes(failed=1, passed=1, skipped=1)

        record = run_tests(
            T,
            spec=_rendering_spec(tmp_path, "-v", "-ra", verbosity_subtests="0"),
            name=name,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "*.py::T::test_foo SUBFAILED[[]subtest 2[]]                        *            [[]100%[]]",
                "*.py::T::test_foo PASSED                                          *            [[]100%[]]",
                "*=== short test summary info ===*",
                r"SUBFAILED[[]subtest 2[]] *.py::T::test_foo - AssertionError: fail subtest 2",
                r"* 1 failed, 1 passed in *",
            ]
        )
        record.stdout.no_fnmatch_line(
            "*.py::T::test_foo SUBSKIPPED[[]subtest 1[]] (skip subtest 1) * [[]100%[]]"
        )
        record.stdout.no_fnmatch_line(
            "SUBSKIPPED[[]subtest 1[]] [[]1[]] *.py:*: skip subtest 1"
        )

    def test_msg_not_a_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Using a non-string in TestCase.subTest should still show it in the terminal (#14195)."""
        monkeypatch.setenv("COLUMNS", "120")

        class T(unittest.TestCase):
            def test_int_msg(self) -> None:
                with self.subTest(42):
                    assert False, "subtest failure"

            def test_no_msg(self) -> None:
                with self.subTest():
                    assert False, "subtest failure"

        record = run_tests(
            T,
            spec=_rendering_spec(tmp_path),
            name="test_msg_not_a_string",
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "SUBFAILED[[]42[]] test_msg_not_a_string.py::T::test_int_msg - AssertionError: subtest failure",
                "SUBFAILED(<subtest>) test_msg_not_a_string.py::T::test_no_msg - AssertionError: subtest failure",
            ]
        )


# ensemble: every test in this class needs capture *around the whole item* -
# the "__ test __" section holding the top level "start test"/"end test"
# output, ``-s``, and ``capsys``/``capfd``. An ensemble config never runs
# ``pytest_load_initial_conftests``, so its capture manager never starts
# global capturing: the subtests' own CaptureFixture still works, but the
# enclosing test's output is neither captured nor reported - it escapes to
# the *host's* stdout instead.
class TestCapture:
    def create_file(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            """
            import sys
            def test(subtests):
                print()
                print('start test')

                with subtests.test(i='A'):
                    print("hello stdout A")
                    print("hello stderr A", file=sys.stderr)
                    assert 0

                with subtests.test(i='B'):
                    print("hello stdout B")
                    print("hello stderr B", file=sys.stderr)
                    assert 0

                print('end test')
                assert 0
        """
        )

    @pytest.mark.parametrize("mode", ["fd", "sys"])
    def test_capturing(self, pytester: pytest.Pytester, mode: str) -> None:
        self.create_file(pytester)
        result = pytester.runpytest(f"--capture={mode}")
        result.stdout.fnmatch_lines(
            [
                "*__ test (i='A') __*",
                "*Captured stdout call*",
                "hello stdout A",
                "*Captured stderr call*",
                "hello stderr A",
                "*__ test (i='B') __*",
                "*Captured stdout call*",
                "hello stdout B",
                "*Captured stderr call*",
                "hello stderr B",
                "*__ test __*",
                "*Captured stdout call*",
                "start test",
                "end test",
            ]
        )

    def test_no_capture(self, pytester: pytest.Pytester) -> None:
        self.create_file(pytester)
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            [
                "start test",
                "hello stdout A",
                "uhello stdout B",
                "uend test",
                "*__ test (i='A') __*",
                "*__ test (i='B') __*",
                "*__ test __*",
            ]
        )
        result.stderr.fnmatch_lines(["hello stderr A", "hello stderr B"])

    @pytest.mark.parametrize("fixture", ["capsys", "capfd"])
    def test_capture_with_fixture(
        self, pytester: pytest.Pytester, fixture: Literal["capsys", "capfd"]
    ) -> None:
        pytester.makepyfile(
            rf"""
            import sys

            def test(subtests, {fixture}):
                print('start test')

                with subtests.test(i='A'):
                    print("hello stdout A")
                    print("hello stderr A", file=sys.stderr)

                out, err = {fixture}.readouterr()
                assert out == 'start test\nhello stdout A\n'
                assert err == 'hello stderr A\n'
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*1 passed*",
            ]
        )


class TestLogging:
    def create_module(self) -> types.ModuleType:
        def test_foo(subtests: pytest.Subtests) -> None:
            logging.info("before")

            with subtests.test("sub1"):
                print("sub1 stdout")
                logging.info("sub1 logging")
                logging.debug("sub1 logging debug")

            with subtests.test("sub2"):
                print("sub2 stdout")
                logging.info("sub2 logging")
                logging.debug("sub2 logging debug")
                assert False

        return build_module("test_logging", test_foo)

    def test_capturing_info(self, tmp_path: Path) -> None:
        # the subtest sections need the capture plugin for stdout and the
        # logging plugin for the log records; neither is an ensemble default.
        spec = _rendering_spec(tmp_path, "--log-level=INFO").with_plugins(
            "logging", "capture"
        )
        record = run_tests(self.create_module(), spec=spec, capture_output=True)
        record.stdout.fnmatch_lines(
            [
                "*___ test_foo [[]sub2[]] __*",
                "*-- Captured stdout call --*",
                "sub2 stdout",
                "*-- Captured log call ---*",
                "INFO     * before",
                "INFO     * sub1 logging",
                "INFO     * sub2 logging",
                "*== short test summary info ==*",
            ]
        )
        record.stdout.no_fnmatch_line("sub1 logging debug")
        record.stdout.no_fnmatch_line("sub2 logging debug")

    def test_capturing_debug(self, tmp_path: Path) -> None:
        spec = _rendering_spec(tmp_path, "--log-level=DEBUG").with_plugins(
            "logging", "capture"
        )
        record = run_tests(self.create_module(), spec=spec, capture_output=True)
        record.stdout.fnmatch_lines(
            [
                "*___ test_foo [[]sub2[]] __*",
                "*-- Captured stdout call --*",
                "sub2 stdout",
                "*-- Captured log call ---*",
                "INFO     * before",
                "INFO     * sub1 logging",
                "DEBUG    * sub1 logging debug",
                "INFO     * sub2 logging",
                "DEBUG    * sub2 logging debug",
                "*== short test summary info ==*",
            ]
        )

    def test_caplog(self, tmp_path: Path) -> None:
        def test(subtests: pytest.Subtests, caplog: pytest.LogCaptureFixture) -> None:
            caplog.set_level(logging.INFO)
            logging.info("start test")

            with subtests.test("sub1"):
                logging.info("inside %s", "subtest1")

            assert len(caplog.records) == 2
            assert caplog.records[0].getMessage() == "start test"
            assert caplog.records[1].getMessage() == "inside subtest1"

        spec = _subtests_spec(tmp_path).with_plugins("logging")
        record = run_tests(test, spec=spec, name="test_caplog")
        record.assert_outcomes(passed=1)

    def test_no_logging(self, tmp_path: Path) -> None:
        def test(subtests: pytest.Subtests) -> None:
            logging.info("start log line")

            with subtests.test("sub passing"):
                logging.info("inside %s", "passing log line")

            with subtests.test("sub failing"):
                logging.info("inside %s", "failing log line")
                assert False

            logging.info("end log line")

        # the original passed "-p no:logging"; an ensemble simply never loads
        # the logging plugin in the first place.
        record = run_tests(
            test,
            spec=_rendering_spec(tmp_path),
            name="test_no_logging",
            capture_output=True,
        )
        record.assert_outcomes(failed=2)
        record.stdout.no_fnmatch_line("*root:*log line*")


class TestDebugging:
    """Check --pdb support for subtests fixture and TestCase.subTest."""

    class _FakePdb:
        """Fake debugger class implementation that tracks which methods were called on it."""

        quitting: bool = False
        calls: list[str] = []

        def __init__(self, *_: object, **__: object) -> None:
            self.calls.append("init")

        def reset(self) -> None:
            self.calls.append("reset")

        def interaction(self, *_: object) -> None:
            self.calls.append("interaction")

    @pytest.fixture(autouse=True)
    def cleanup_calls(self) -> None:
        self._FakePdb.calls.clear()

    def test_pdb_fixture(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def test(subtests: pytest.Subtests) -> None:
            with subtests.test():
                assert 0

        self.run_and_check_pdb(test, tmp_path, monkeypatch)

    def test_pdb_unittest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Test(unittest.TestCase):
            def test(self) -> None:
                with self.subTest():
                    assert 0

        self.run_and_check_pdb(Test, tmp_path, monkeypatch)

    def run_and_check_pdb(
        self,
        source: Source,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Install the fake pdb implementation in _pytest.subtests so we can reference
        # it in the command line (any module would do).
        import _pytest.subtests

        monkeypatch.setattr(
            _pytest.subtests, "_CustomPdb", self._FakePdb, raising=False
        )
        spec = _rendering_spec(
            tmp_path, "--pdb", "--pdbcls=_pytest.subtests:_CustomPdb"
        ).with_plugins("debugging")
        record = run_tests(source, spec=spec, name="test_pdb", capture_output=True)

        # Ensure pytest entered in debugging mode when encountering the failing
        # assert.
        record.stdout.fnmatch_lines("*entering PDB*")
        assert self._FakePdb.calls == ["init", "reset", "interaction"]


def test_exitfirst(tmp_path: Path) -> None:
    """Validate that when passing --exitfirst the test exits after the first failed subtest."""

    def test_foo(subtests: pytest.Subtests) -> None:
        with subtests.test("sub1"):
            assert False

        with subtests.test("sub2"):
            assert False

    record = run_tests(
        test_foo,
        spec=_rendering_spec(tmp_path, "--exitfirst"),
        name="test_exitfirst",
        capture_output=True,
    )
    assert record.outcomes()["failed"] == 2
    record.stdout.fnmatch_lines(
        [
            "SUBFAILED*[[]sub1[]] *.py::test_foo - assert False*",
            "FAILED *.py::test_foo - assert False",
            "* stopping after 2 failures*",
        ],
        consecutive=True,
    )
    record.stdout.no_fnmatch_line("*sub2*")  # sub2 not executed.


def test_do_not_swallow_pytest_exit(tmp_path: Path) -> None:
    reports: list[TestReport] = []

    class ReportRecorder:
        def pytest_runtest_logreport(self, report: TestReport) -> None:
            reports.append(report)

    def test(subtests: pytest.Subtests) -> None:
        with subtests.test():
            pytest.exit()

    def test2() -> None:
        pass

    module = build_module("test_do_not_swallow_pytest_exit", test, test2)
    spec = _subtests_spec(tmp_path).with_plugins(ReportRecorder())
    # the original observed the Exit escaping as a subprocess traceback; here
    # it simply has to come back out of the run.
    with pytest.raises(Exit):
        run_tests(module, spec=spec)
    # the subtest was reported as failed, and ``test2`` never ran
    assert [(report.when, report.outcome) for report in reports] == [
        ("setup", "passed"),
        ("call", "failed"),
    ]
    assert isinstance(reports[-1], SubtestReport)
    assert reports[-1].nodeid == "test_do_not_swallow_pytest_exit.py::test"


def test_nested(tmp_path: Path) -> None:
    """
    Currently we do nothing special with nested subtests.

    This test only sediments how they work now, we might reconsider adding some kind of nesting support in the future.
    """

    def test(subtests: pytest.Subtests) -> None:
        with subtests.test("a"):
            with subtests.test("b"):
                assert False, "b failed"
            assert False, "a failed"

    record = run_tests(
        test, spec=_rendering_spec(tmp_path), name="test_nested", capture_output=True
    )
    record.stdout.fnmatch_lines(
        [
            "SUBFAILED[b] test_nested.py::test - AssertionError: b failed",
            "SUBFAILED[a] test_nested.py::test - AssertionError: a failed",
            "* 3 failed in *",
        ]
    )
    record.assert_outcomes(failed=3)


class MyEnum(Enum):
    """Used in test_serialization, needs to be declared at the module level to be pickled."""

    A = "A"


def test_serialization() -> None:
    """Ensure subtest's kwargs are serialized using `saferepr` (pytest-dev/pytest-xdist#1273)."""
    from _pytest.subtests import pytest_report_from_serializable
    from _pytest.subtests import pytest_report_to_serializable

    report = SubtestReport(
        "test_foo::test_foo",
        ("test_foo.py", 12, ""),
        keywords={},
        outcome="passed",
        when="call",
        longrepr=None,
        context=SubtestContext(msg="custom message", kwargs=dict(i=10, a=MyEnum.A)),
    )
    data = pytest_report_to_serializable(report)
    assert data is not None
    # Ensure the report is actually serializable to JSON.
    _ = json.dumps(data)
    new_report = pytest_report_from_serializable(data)
    assert new_report is not None
    assert new_report.context == SubtestContext(
        msg="custom message", kwargs=dict(i=saferepr(10), a=saferepr(MyEnum.A))
    )


# ensemble: needs xdist, which reruns the tests in worker *subprocesses* over
# real files on disk (hence the syspathinsert).
def test_serialization_xdist(pytester: pytest.Pytester) -> None:  # pragma: no cover
    """Regression test for pytest-dev/pytest-xdist#1273."""
    pytest.importorskip("xdist")
    pytester.makepyfile(
        """
        from enum import Enum
        import unittest

        class MyEnum(Enum):
            A = "A"

        def test(subtests):
            with subtests.test(a=MyEnum.A):
                pass

        class T(unittest.TestCase):

            def test(self):
                with self.subTest(a=MyEnum.A):
                    pass
        """
    )
    pytester.syspathinsert()
    result = pytester.runpytest("-n1", "-pxdist.plugin")
    result.assert_outcomes(passed=2)

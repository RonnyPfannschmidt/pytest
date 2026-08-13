# mypy: allow-untyped-defs
from __future__ import annotations

from pathlib import Path
import textwrap

from _pytest._code import ExceptionInfo
from _pytest.ensemble import build_module
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import Ensemble
from _pytest.ensemble import run_tests
from _pytest.ensemble import RunRecord
from _pytest.outcomes import Exit
from _pytest.pytester import Pytester
from _pytest.runner import runtestprotocol
from _pytest.skipping import evaluate_skip_marks
from _pytest.skipping import evaluate_xfail_marks
from _pytest.skipping import pytest_runtest_setup
import pytest


def setup_longrepr(record: RunRecord, name: str) -> str:
    """The rendered longrepr of a test's setup report.

    Skip reasons and setup errors live here; the ensemble equivalent of
    matching them in the terminal's short summary.
    """
    setup = record[name].setup
    assert setup is not None
    return setup.longreprtext


class TestEvaluation:
    def test_no_marker(self, tmp_path: Path) -> None:
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            skipped = evaluate_skip_marks(item)
            assert not skipped

    def test_marked_xfail_no_args(self, tmp_path: Path) -> None:
        @pytest.mark.xfail
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            xfailed = evaluate_xfail_marks(item)
            assert xfailed
            assert xfailed.reason == ""
            assert xfailed.run

    def test_marked_skipif_no_args(self, tmp_path: Path) -> None:
        # A bare `skipif` (no condition) is deliberately not a valid call.
        @pytest.mark.skipif  # type: ignore[arg-type]
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            skipped = evaluate_skip_marks(item)
            assert skipped
            assert skipped.reason == ""

    def test_marked_one_arg(self, tmp_path: Path) -> None:
        @pytest.mark.skipif("hasattr(os, 'sep')")
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            skipped = evaluate_skip_marks(item)
            assert skipped
            assert skipped.reason == "condition: hasattr(os, 'sep')"

    def test_marked_one_arg_with_reason(self, tmp_path: Path) -> None:
        @pytest.mark.skipif(  # type: ignore[call-arg]
            "hasattr(os, 'sep')", attr=2, reason="hello world"
        )
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            skipped = evaluate_skip_marks(item)
            assert skipped
            assert skipped.reason == "hello world"

    def test_marked_one_arg_twice(self, tmp_path: Path) -> None:
        # The original generated both stacking orders from the same two
        # source lines; here they are spelled out, one function each.
        @pytest.mark.skipif("not hasattr(os, 'murks')")
        @pytest.mark.skipif(condition="hasattr(os, 'murks')")
        def test_func_string_first():
            pass

        @pytest.mark.skipif(condition="hasattr(os, 'murks')")
        @pytest.mark.skipif("not hasattr(os, 'murks')")
        def test_func_keyword_first():
            pass

        for func in (test_func_string_first, test_func_keyword_first):
            with Ensemble(func, rootpath=tmp_path) as ensemble:
                (item,) = ensemble.collect()
                skipped = evaluate_skip_marks(item)
                assert skipped
                assert skipped.reason == "condition: not hasattr(os, 'murks')"

    def test_marked_one_arg_twice2(self, tmp_path: Path) -> None:
        @pytest.mark.skipif("hasattr(os, 'murks')")
        @pytest.mark.skipif("not hasattr(os, 'murks')")
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            skipped = evaluate_skip_marks(item)
            assert skipped
            assert skipped.reason == "condition: not hasattr(os, 'murks')"

    def test_marked_skipif_with_boolean_without_reason(self, tmp_path: Path) -> None:
        @pytest.mark.skipif(False)
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            with pytest.raises(pytest.fail.Exception) as excinfo:
                evaluate_skip_marks(item)
        assert excinfo.value.msg is not None
        assert (
            """Error evaluating 'skipif': you need to specify reason=STRING when using booleans as conditions."""
            in excinfo.value.msg
        )

    def test_marked_skipif_with_invalid_boolean(self, tmp_path: Path) -> None:
        class InvalidBool:
            def __bool__(self):
                raise TypeError("INVALID")

        @pytest.mark.skipif(InvalidBool(), reason="xxx")  # type: ignore[arg-type]
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            with pytest.raises(pytest.fail.Exception) as excinfo:
                evaluate_skip_marks(item)
        assert excinfo.value.msg is not None
        assert "Error evaluating 'skipif' condition as a boolean" in excinfo.value.msg
        assert "INVALID" in excinfo.value.msg

    def test_skipif_class(self, tmp_path: Path) -> None:
        class TestClass:
            pytestmark = pytest.mark.skipif("config._hackxyz")

            def test_func(self):
                pass

        with Ensemble(TestClass, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            item.config._hackxyz = 3  # type: ignore[attr-defined]
            skipped = evaluate_skip_marks(item)
            assert skipped
            assert skipped.reason == "condition: config._hackxyz"

    def test_skipif_markeval_namespace(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            def pytest_markeval_namespace(self):
                return {"color": "green"}

        @pytest.mark.skipif("color == 'green'")
        def test_1():
            assert True

        @pytest.mark.skipif("color == 'red'")
        def test_2():
            assert True

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(test_1, test_2, spec=spec)
        # Stronger than the two separate "*1 skipped*"/"*1 passed*" matches:
        # this also pins that nothing else happened.
        record.assert_outcomes(passed=1, skipped=1)

    # ensemble: the point is that a conftest deeper in the tree overrides the
    # namespace of one above it, and ensembles have no directory scoping.
    def test_skipif_markeval_namespace_multiple(self, pytester: Pytester) -> None:
        """Keys defined by ``pytest_markeval_namespace()`` in nested plugins override top-level ones."""
        root = pytester.mkdir("root")
        root.joinpath("__init__.py").touch()
        root.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
            import pytest

            def pytest_markeval_namespace():
                return {"arg": "root"}
            """
            ),
            encoding="utf-8",
        )
        root.joinpath("test_root.py").write_text(
            textwrap.dedent(
                """\
            import pytest

            @pytest.mark.skipif("arg == 'root'")
            def test_root():
                assert False
            """
            ),
            encoding="utf-8",
        )
        foo = root.joinpath("foo")
        foo.mkdir()
        foo.joinpath("__init__.py").touch()
        foo.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
            import pytest

            def pytest_markeval_namespace():
                return {"arg": "foo"}
            """
            ),
            encoding="utf-8",
        )
        foo.joinpath("test_foo.py").write_text(
            textwrap.dedent(
                """\
            import pytest

            @pytest.mark.skipif("arg == 'foo'")
            def test_foo():
                assert False
            """
            ),
            encoding="utf-8",
        )
        bar = root.joinpath("bar")
        bar.mkdir()
        bar.joinpath("__init__.py").touch()
        bar.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
            import pytest

            def pytest_markeval_namespace():
                return {"arg": "bar"}
            """
            ),
            encoding="utf-8",
        )
        bar.joinpath("test_bar.py").write_text(
            textwrap.dedent(
                """\
            import pytest

            @pytest.mark.skipif("arg == 'bar'")
            def test_bar():
                assert False
            """
            ),
            encoding="utf-8",
        )

        reprec = pytester.inline_run("-vs", "--capture=no")
        reprec.assertoutcome(skipped=3)

    def test_skipif_markeval_namespace_ValueError(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            def pytest_markeval_namespace(self):
                return True

        @pytest.mark.skipif("color == 'green'")
        def test_1():
            assert True

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(test_1, spec=spec)
        # The ValueError escapes pytest_runtest_setup, so this is an error,
        # not a failure - the nonzero exit code of the original did not say
        # which.
        record.assert_outcomes(errors=1)
        assert (
            "ValueError: pytest_markeval_namespace() needs to return a dict, got True"
            in setup_longrepr(record, "test_1")
        )


class TestXFail:
    @pytest.mark.parametrize("strict", [True, False])
    def test_xfail_simple(self, tmp_path: Path, strict: bool) -> None:
        @pytest.mark.xfail(strict=strict)
        def test_func():
            assert 0

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            reports = runtestprotocol(item, log=False)
        assert len(reports) == 3
        callreport = reports[1]
        assert callreport.skipped
        assert callreport.wasxfail == ""

    def test_xfail_xpassed(self, tmp_path: Path) -> None:
        @pytest.mark.xfail(reason="this is an xfail")
        def test_func():
            assert 1

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            reports = runtestprotocol(item, log=False)
        assert len(reports) == 3
        callreport = reports[1]
        assert callreport.passed
        assert callreport.wasxfail == "this is an xfail"

    def test_xfail_using_platform(self, tmp_path: Path) -> None:
        """Verify that platform can be used with xfail statements."""

        @pytest.mark.xfail("platform.platform() == platform.platform()")
        def test_func():
            assert 0

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            reports = runtestprotocol(item, log=False)
        assert len(reports) == 3
        callreport = reports[1]
        assert callreport.wasxfail

    def test_xfail_xpassed_strict(self, tmp_path: Path) -> None:
        @pytest.mark.xfail(strict=True, reason="nope")
        def test_func():
            assert 1

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            reports = runtestprotocol(item, log=False)
        assert len(reports) == 3
        callreport = reports[1]
        assert callreport.failed
        assert str(callreport.longrepr) == "[XPASS(strict)] nope"
        assert not hasattr(callreport, "wasxfail")

    def test_xfail_run_anyway(self, tmp_path: Path) -> None:
        @pytest.mark.xfail
        def test_func():
            assert 0

        def test_func2():
            pytest.xfail("hello")

        spec = ConfigSpec(rootpath=tmp_path, args=("--runxfail",))
        record = run_tests(test_func, test_func2, spec=spec, capture_output=True)
        # --runxfail replaces pytest.xfail with a no-op, so test_func2 passes.
        record.assert_outcomes(failed=1, passed=1)
        record.stdout.fnmatch_lines(
            ["*def test_func():*", "*assert 0*", "*1 failed*1 pass*"]
        )

    # ensemble: the expected `-rs` line names the source file and the line the
    # skip mark sits on, which for an ensemble source is this very file.
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                ["-rs"],
                ["SKIPPED [1] test_sample.py:2: unconditional skip", "*1 skipped*"],
            ),
            (
                ["-rs", "--runxfail"],
                ["SKIPPED [1] test_sample.py:2: unconditional skip", "*1 skipped*"],
            ),
        ],
    )
    def test_xfail_run_with_skip_mark(
        self, pytester: Pytester, test_input, expected
    ) -> None:
        pytester.makepyfile(
            test_sample="""
            import pytest
            @pytest.mark.skip
            def test_skip_location() -> None:
                assert 0
        """
        )
        result = pytester.runpytest(*test_input)
        result.stdout.fnmatch_lines(expected)

    def test_xfail_evalfalse_but_fails(self, tmp_path: Path) -> None:
        @pytest.mark.xfail("False")
        def test_func():
            assert 0

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            reports = runtestprotocol(item, log=False)
        callreport = reports[1]
        assert callreport.failed
        assert not hasattr(callreport, "wasxfail")
        assert "xfail" in callreport.keywords

    def test_xfail_not_report_default(self, tmp_path: Path) -> None:
        @pytest.mark.xfail
        def test_this():
            assert 0

        spec = ConfigSpec(rootpath=tmp_path, args=("-v",))
        record = run_tests(
            build_module("test_one", test_this=test_this),
            spec=spec,
            capture_output=True,
        )
        # result.stdout.fnmatch_lines([
        #    "*HINT*use*-r*"
        # ])
        record.assert_outcomes(xfailed=1)
        # Without a report char there is no short summary section at all.
        record.stdout.no_fnmatch_line("*short test summary info*")

    def test_xfail_not_run_xfail_reporting(self, tmp_path: Path) -> None:
        @pytest.mark.xfail(run=False, reason="noway")
        def test_this():
            assert 0

        @pytest.mark.xfail("True", run=False)
        def test_this_true():
            assert 0

        @pytest.mark.xfail("False", run=False, reason="huh")
        def test_this_false():
            assert 1

        spec = ConfigSpec(rootpath=tmp_path, args=("-rx",))
        record = run_tests(
            build_module(
                "test_one",
                test_this=test_this,
                test_this_true=test_this_true,
                test_this_false=test_this_false,
            ),
            spec=spec,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "*test_one*test_this - *NOTRUN* noway",
                "*test_one*test_this_true - *NOTRUN* condition: True",
                "*1 passed*",
            ]
        )
        record.assert_outcomes(passed=1, xfailed=2)

    def test_xfail_not_run_does_not_format_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @pytest.mark.xfail(run=False, reason="noway")
        def test_func():
            assert 0

        getrepr = ExceptionInfo.getrepr
        styles = []

        def spy_getrepr(self, *args, **kwargs):
            styles.append(kwargs["style"])
            return getrepr(self, *args, **kwargs)

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            monkeypatch.setattr(ExceptionInfo, "getrepr", spy_getrepr)
            reports = runtestprotocol(item, log=False)

        assert reports[0].skipped
        assert styles == ["value"]

    def test_xfail_not_run_no_setup_run(self, tmp_path: Path) -> None:
        @pytest.mark.xfail(run=False, reason="hello")
        def test_this():
            assert 0

        def setup_module(mod):
            raise ValueError(42)

        spec = ConfigSpec(rootpath=tmp_path, args=("-rx",))
        record = run_tests(
            build_module("test_one", test_this=test_this, setup_module=setup_module),
            spec=spec,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*test_one*test_this*NOTRUN*hello", "*1 xfailed*"])
        record.assert_outcomes(xfailed=1)

    def test_xfail_xpass(self, tmp_path: Path) -> None:
        @pytest.mark.xfail
        def test_that():
            assert 1

        spec = ConfigSpec(rootpath=tmp_path, args=("-rX",))
        record = run_tests(
            build_module("test_one", test_that=test_that),
            spec=spec,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*XPASS*test_that*", "*1 xpassed*"])
        # An xpass is not a failure, which is what `ret == 0` stood for.
        record.assert_outcomes(xpassed=1)

    def test_xfail_imperative(self, tmp_path: Path) -> None:
        def test_this():
            pytest.xfail("hello")

        record = run_tests(test_this, rootpath=tmp_path)
        record.assert_outcomes(xfailed=1)
        record = run_tests(
            test_this,
            spec=ConfigSpec(rootpath=tmp_path, args=("-rx",)),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*XFAIL*test_this*hello*"])
        record = run_tests(
            test_this, spec=ConfigSpec(rootpath=tmp_path, args=("--runxfail",))
        )
        record.assert_outcomes(passed=1)

    def test_xfail_imperative_in_setup_function(self, tmp_path: Path) -> None:
        def setup_function(function):
            pytest.xfail("hello")

        def test_this():
            assert 0

        module = build_module(
            "test_one", setup_function=setup_function, test_this=test_this
        )
        record = run_tests(module, rootpath=tmp_path)
        record.assert_outcomes(xfailed=1)
        record = run_tests(
            module,
            spec=ConfigSpec(rootpath=tmp_path, args=("-rx",)),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*XFAIL*test_this*hello*"])
        record = run_tests(
            module,
            spec=ConfigSpec(rootpath=tmp_path, args=("--runxfail",)),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            """
            *def test_this*
            *1 fail*
        """
        )
        record.assert_outcomes(failed=1)

    # ensemble: not a test - the name does not start with `test`, so it has
    # never been collected; left untouched rather than resurrected here.
    def xtest_dynamic_xfail_set_during_setup(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            """
            import pytest
            def setup_function(function):
                pytest.mark.xfail(function)
            def test_this():
                assert 0
            def test_that():
                assert 1
        """
        )
        result = pytester.runpytest(p, "-rxX")
        result.stdout.fnmatch_lines(["*XFAIL*test_this*", "*XPASS*test_that*"])

    def test_dynamic_xfail_no_run(self, tmp_path: Path) -> None:
        @pytest.fixture
        def arg(request):
            request.applymarker(pytest.mark.xfail(run=False))

        def test_this(arg):
            assert 0

        spec = ConfigSpec(rootpath=tmp_path, args=("-rxX",))
        record = run_tests(arg, test_this, spec=spec, capture_output=True)
        record.stdout.fnmatch_lines(["*XFAIL*test_this*NOTRUN*"])
        record.assert_outcomes(xfailed=1)

    def test_dynamic_xfail_set_during_funcarg_setup(self, tmp_path: Path) -> None:
        @pytest.fixture
        def arg(request):
            request.applymarker(pytest.mark.xfail)

        def test_this2(arg):
            assert 0

        record = run_tests(arg, test_this2, rootpath=tmp_path)
        record.assert_outcomes(xfailed=1)

    def test_dynamic_xfail_set_during_runtest_failed(self, tmp_path: Path) -> None:
        # Issue #7486.
        def test_this(request):
            request.node.add_marker(pytest.mark.xfail(reason="xfail"))
            assert 0

        record = run_tests(test_this, rootpath=tmp_path)
        record.assert_outcomes(xfailed=1)

    def test_dynamic_xfail_set_during_runtest_passed_strict(
        self, tmp_path: Path
    ) -> None:
        # Issue #7486.
        def test_this(request):
            request.node.add_marker(pytest.mark.xfail(reason="xfail", strict=True))

        record = run_tests(test_this, rootpath=tmp_path)
        record.assert_outcomes(failed=1)

    @pytest.mark.parametrize(
        "expected, actual, outcome",
        [
            (TypeError, TypeError, "xfailed"),
            ((AttributeError, TypeError), TypeError, "xfailed"),
            (TypeError, IndexError, "failed"),
            ((AttributeError, TypeError), IndexError, "failed"),
        ],
    )
    def test_xfail_raises(self, expected, actual, outcome, tmp_path: Path) -> None:
        @pytest.mark.xfail(raises=expected)
        def test_raises():
            raise actual()

        record = run_tests(test_raises, rootpath=tmp_path)
        # Stronger than the single summary-line match of the original.
        record.assert_outcomes(**{outcome: 1})

    def test_strict_sanity(self, tmp_path: Path) -> None:
        """Sanity check for xfail(strict=True): a failing test should behave
        exactly like a normal xfail."""

        @pytest.mark.xfail(reason="unsupported feature", strict=True)
        def test_foo():
            assert 0

        spec = ConfigSpec(rootpath=tmp_path, args=("-rxX",))
        record = run_tests(test_foo, spec=spec, capture_output=True)
        record.stdout.fnmatch_lines(["*XFAIL*unsupported feature*"])
        # `ret == 0` stood for "nothing failed".
        record.assert_outcomes(xfailed=1)

    @pytest.mark.parametrize("strict", [True, False])
    def test_strict_xfail(self, tmp_path: Path, strict: bool) -> None:
        executed = []

        @pytest.mark.xfail(reason="unsupported feature", strict=strict)
        def test_foo():
            executed.append(True)  # make sure test executes

        spec = ConfigSpec(rootpath=tmp_path, args=("-rxX",))
        record = run_tests(
            build_module("test_strict_xfail", test_foo=test_foo),
            spec=spec,
            capture_output=True,
        )
        if strict:
            record.stdout.fnmatch_lines(
                ["*test_foo*", "*XPASS(strict)*unsupported feature*"]
            )
            record.assert_outcomes(failed=1)
        else:
            record.stdout.fnmatch_lines(
                [
                    "*test_strict_xfail*",
                    "XPASS test_strict_xfail.py::test_foo - unsupported feature",
                ]
            )
            record.assert_outcomes(xpassed=1)
        assert executed == [True]

    @pytest.mark.parametrize("strict", [True, False])
    def test_strict_xfail_condition(self, tmp_path: Path, strict: bool) -> None:
        @pytest.mark.xfail(False, reason="unsupported feature", strict=strict)
        def test_foo():
            pass

        # `-rxX` is dropped: it only selects short summary lines, which this
        # no longer matches on, and it is a terminal plugin option.
        record = run_tests(test_foo, rootpath=tmp_path)
        record.assert_outcomes(passed=1)

    @pytest.mark.parametrize("strict", [True, False])
    def test_xfail_condition_keyword(self, tmp_path: Path, strict: bool) -> None:
        @pytest.mark.xfail(condition=False, reason="unsupported feature", strict=strict)
        def test_foo():
            pass

        record = run_tests(test_foo, rootpath=tmp_path)
        record.assert_outcomes(passed=1)

    @pytest.mark.parametrize("strict_val", ["true", "false"])
    @pytest.mark.parametrize("option_name", ["strict_xfail", "strict"])
    def test_strict_xfail_default_from_file(
        self, tmp_path: Path, strict_val: str, option_name: str
    ) -> None:
        @pytest.mark.xfail(reason="unsupported feature")
        def test_foo():
            pass

        spec = ConfigSpec(rootpath=tmp_path, inicfg={option_name: strict_val})
        record = run_tests(test_foo, spec=spec)
        strict = strict_val == "true"
        if strict:
            record.assert_outcomes(failed=1)
        else:
            record.assert_outcomes(xpassed=1)

    def test_xfail_markeval_namespace(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            def pytest_markeval_namespace(self):
                return {"color": "green"}

        @pytest.mark.xfail("color == 'green'")
        def test_1():
            assert False

        @pytest.mark.xfail("color == 'red'")
        def test_2():
            assert False

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(test_1, test_2, spec=spec)
        record.assert_outcomes(failed=1, xfailed=1)


class TestXFailwithSetupTeardown:
    def test_failing_setup_issue9(self, tmp_path: Path) -> None:
        def setup_function(func):
            assert 0

        @pytest.mark.xfail
        def test_func():
            pass

        record = run_tests(
            build_module(
                "test_one", setup_function=setup_function, test_func=test_func
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(xfailed=1)

    def test_failing_teardown_issue9(self, tmp_path: Path) -> None:
        def teardown_function(func):
            assert 0

        @pytest.mark.xfail
        def test_func():
            pass

        record = run_tests(
            build_module(
                "test_one", teardown_function=teardown_function, test_func=test_func
            ),
            rootpath=tmp_path,
        )
        # The call phase passes (so: xpassed) and only the teardown turns into
        # an xfail; "*1 xfail*" matched the "1 xpassed, 1 xfailed" summary and
        # hid the xpass.
        record.assert_outcomes(xpassed=1, xfailed=1)


class TestSkip:
    def test_skip_class(self, tmp_path: Path) -> None:
        @pytest.mark.skip
        class TestSomething:
            def test_foo(self):
                pass

            def test_bar(self):
                pass

        def test_baz():
            pass

        record = run_tests(TestSomething, test_baz, rootpath=tmp_path)
        record.assert_outcomes(skipped=2, passed=1)

    def test_skips_on_false_string(self, tmp_path: Path) -> None:
        @pytest.mark.skip("False")
        def test_foo():
            pass

        record = run_tests(test_foo, rootpath=tmp_path)
        record.assert_outcomes(skipped=1)

    def test_arg_as_reason(self, tmp_path: Path) -> None:
        @pytest.mark.skip("testing stuff")
        def test_bar():
            pass

        record = run_tests(test_bar, rootpath=tmp_path)
        record.assert_outcomes(skipped=1)
        assert "testing stuff" in setup_longrepr(record, "test_bar")

    def test_skip_no_reason(self, tmp_path: Path) -> None:
        @pytest.mark.skip
        def test_foo():
            pass

        record = run_tests(test_foo, rootpath=tmp_path)
        record.assert_outcomes(skipped=1)
        assert "unconditional skip" in setup_longrepr(record, "test_foo")

    def test_skip_with_reason(self, tmp_path: Path) -> None:
        @pytest.mark.skip(reason="for lolz")
        def test_bar():
            pass

        record = run_tests(test_bar, rootpath=tmp_path)
        record.assert_outcomes(skipped=1)
        assert "for lolz" in setup_longrepr(record, "test_bar")

    def test_only_skips_marked_test(self, tmp_path: Path) -> None:
        @pytest.mark.skip
        def test_foo():
            pass

        @pytest.mark.skip(reason="nothing in particular")
        def test_bar():
            pass

        def test_baz():
            assert True

        record = run_tests(test_foo, test_bar, test_baz, rootpath=tmp_path)
        record.assert_outcomes(passed=1, skipped=2)
        assert "nothing in particular" in setup_longrepr(record, "test_bar")

    def test_strict_and_skip(self, tmp_path: Path) -> None:
        @pytest.mark.skip
        def test_hello():
            pass

        spec = ConfigSpec(rootpath=tmp_path, args=("--strict-markers",))
        record = run_tests(test_hello, spec=spec)
        record.assert_outcomes(skipped=1)
        assert "unconditional skip" in setup_longrepr(record, "test_hello")

    def test_wrong_skip_usage(self, tmp_path: Path) -> None:
        # Deliberately wrong: `skip` takes no condition.
        @pytest.mark.skip(False, reason="I thought this was skipif")  # type: ignore[call-overload]
        def test_hello():
            pass

        record = run_tests(test_hello, rootpath=tmp_path)
        # The TypeError escapes pytest_runtest_setup: an error, not a failure.
        record.assert_outcomes(errors=1)
        longrepr = setup_longrepr(record, "test_hello")
        assert "TypeError: " in longrepr
        assert (
            "got multiple values for argument 'reason'"
            " - maybe you meant pytest.mark.skipif?" in longrepr
        )


class TestSkipif:
    def test_skipif_conditional(self, tmp_path: Path) -> None:
        @pytest.mark.skipif("hasattr(os, 'sep')")
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            x = pytest.raises(pytest.skip.Exception, lambda: pytest_runtest_setup(item))
        assert x.value.msg == "condition: hasattr(os, 'sep')"

    # ensemble: the expected `-rs` line names the file the skipped function
    # lives in, which for an ensemble source is this very file.
    @pytest.mark.parametrize(
        "params", ["\"hasattr(sys, 'platform')\"", 'True, reason="invalid platform"']
    )
    def test_skipif_reporting(self, pytester: Pytester, params) -> None:
        p = pytester.makepyfile(
            test_foo=f"""
            import pytest
            @pytest.mark.skipif({params})
            def test_that():
                assert 0
        """
        )
        result = pytester.runpytest(p, "-s", "-rs")
        result.stdout.fnmatch_lines(["*SKIP*1*test_foo.py*platform*", "*1 skipped*"])
        assert result.ret == 0

    def test_skipif_using_platform(self, tmp_path: Path) -> None:
        @pytest.mark.skipif("platform.platform() == platform.platform()")
        def test_func():
            pass

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            with pytest.raises(pytest.skip.Exception):
                pytest_runtest_setup(item)

    # ensemble: the `SKIP` half of this expects a `-rs` line naming the file the
    # skipped function lives in, which for an ensemble source is this very file.
    @pytest.mark.parametrize(
        "marker, msg1, msg2",
        [("skipif", "SKIP", "skipped"), ("xfail", "XPASS", "xpassed")],
    )
    def test_skipif_reporting_multiple(
        self, pytester: Pytester, marker, msg1, msg2
    ) -> None:
        pytester.makepyfile(
            test_foo=f"""
            import pytest
            @pytest.mark.{marker}(False, reason='first_condition')
            @pytest.mark.{marker}(True, reason='second_condition')
            def test_foobar():
                assert 1
        """
        )
        result = pytester.runpytest("-s", "-rsxX")
        result.stdout.fnmatch_lines(
            [f"*{msg1}*test_foo.py*second_condition*", f"*1 {msg2}*"]
        )
        assert result.ret == 0


def test_skip_not_report_default(tmp_path: Path) -> None:
    def test_this():
        pytest.skip("hello")

    spec = ConfigSpec(rootpath=tmp_path, args=("-v",))
    record = run_tests(
        build_module("test_one", test_this=test_this), spec=spec, capture_output=True
    )
    # "*HINT*use*-r*",
    record.assert_outcomes(skipped=1)
    # Without a report char there is no short summary section at all.
    record.stdout.no_fnmatch_line("*short test summary info*")


def test_skipif_class(tmp_path: Path) -> None:
    class TestClass:
        pytestmark = pytest.mark.skipif("True")

        def test_that(self):
            assert 0

        def test_though(self):
            assert 0

    record = run_tests(TestClass, rootpath=tmp_path)
    record.assert_outcomes(skipped=2)


# ensemble: asserts the exact `file:line` of every skip, and both the skipping
# helper module and the reported locations are host-anchored here.
def test_skipped_reasons_functional(pytester: Pytester) -> None:
    pytester.makepyfile(
        test_one="""
            import pytest
            from helpers import doskip

            def setup_function(func):  # LINE 4
                doskip("setup function")

            def test_func():
                pass

            class TestClass:
                def test_method(self):
                    doskip("test method")

                @pytest.mark.skip("via_decorator")  # LINE 14
                def test_deco(self):
                    assert 0
        """,
        helpers="""
            import pytest, sys
            def doskip(reason):
                assert sys._getframe().f_lineno == 3
                pytest.skip(reason)  # LINE 4
        """,
    )
    result = pytester.runpytest("-rs")
    result.stdout.fnmatch_lines_random(
        [
            "SKIPPED [[]1[]] test_one.py:7: setup function",
            "SKIPPED [[]1[]] helpers.py:4: test method",
            "SKIPPED [[]1[]] test_one.py:14: via_decorator",
        ]
    )
    assert result.ret == 0


# ensemble: the folded `-rs` line names the file the skipped tests live in,
# which for an ensemble source is this very file.
def test_skipped_folding(pytester: Pytester) -> None:
    pytester.makepyfile(
        test_one="""
            import pytest
            pytestmark = pytest.mark.skip("Folding")
            def setup_function(func):
                pass
            def test_func():
                pass
            class TestClass(object):
                def test_method(self):
                    pass
       """
    )
    result = pytester.runpytest("-rs")
    result.stdout.fnmatch_lines(["*SKIP*2*test_one.py: Folding"])
    assert result.ret == 0


def test_reportchars(tmp_path: Path) -> None:
    def test_1():
        assert 0

    @pytest.mark.xfail
    def test_2():
        assert 0

    @pytest.mark.xfail
    def test_3():
        pass

    def test_4():
        pytest.skip("four")

    spec = ConfigSpec(rootpath=tmp_path, args=("-rfxXs",))
    record = run_tests(test_1, test_2, test_3, test_4, spec=spec, capture_output=True)
    record.stdout.fnmatch_lines(
        ["FAIL*test_1*", "XFAIL*test_2*", "XPASS*test_3*", "SKIP*four*"]
    )


def test_reportchars_error(tmp_path: Path) -> None:
    class ConftestPlugin:
        def pytest_runtest_teardown(self):
            assert 0

    def test_foo():
        pass

    spec = ConfigSpec(
        rootpath=tmp_path, args=("-rE",), extra_plugins=(ConftestPlugin(),)
    )
    record = run_tests(
        build_module("test_simple", test_foo=test_foo), spec=spec, capture_output=True
    )
    record.stdout.fnmatch_lines(["ERROR*test_foo*"])


def test_reportchars_all(tmp_path: Path) -> None:
    def test_1():
        assert 0

    @pytest.mark.xfail
    def test_2():
        assert 0

    @pytest.mark.xfail
    def test_3():
        pass

    def test_4():
        pytest.skip("four")

    @pytest.fixture
    def fail():
        assert 0

    def test_5(fail):
        pass

    spec = ConfigSpec(rootpath=tmp_path, args=("-ra",))
    record = run_tests(
        test_1, test_2, test_3, test_4, fail, test_5, spec=spec, capture_output=True
    )
    record.stdout.fnmatch_lines(
        [
            "SKIP*four*",
            "XFAIL*test_2*",
            "XPASS*test_3*",
            "ERROR*test_5*",
            "FAIL*test_1*",
        ]
    )


def test_reportchars_all_error(tmp_path: Path) -> None:
    class ConftestPlugin:
        def pytest_runtest_teardown(self):
            assert 0

    def test_foo():
        pass

    spec = ConfigSpec(
        rootpath=tmp_path, args=("-ra",), extra_plugins=(ConftestPlugin(),)
    )
    record = run_tests(
        build_module("test_simple", test_foo=test_foo), spec=spec, capture_output=True
    )
    record.stdout.fnmatch_lines(["ERROR*test_foo*"])


def test_errors_in_xfail_skip_expressions(tmp_path: Path) -> None:
    @pytest.mark.skipif("asd")
    def test_nameerror():
        pass

    @pytest.mark.xfail("syntax error")
    def test_syntax():
        pass

    def test_func():
        pass

    record = run_tests(
        test_nameerror, test_syntax, test_func, rootpath=tmp_path, capture_output=True
    )

    expected = [
        "*ERROR*test_nameerror*",
        "*asd*",
        "",
        "During handling of the above exception, another exception occurred:",
    ]

    expected += [
        "*evaluating*skipif*condition*",
        "*asd*",
        "*ERROR*test_syntax*",
        "*evaluating*xfail*condition*",
        "    syntax error",
        "            ^",
        "SyntaxError: invalid syntax",
        "*1 pass*2 errors*",
    ]
    record.stdout.fnmatch_lines(expected)
    record.assert_outcomes(passed=1, errors=2)


# ensemble: string conditions are eval'd in the *source function's* __globals__,
# which for an ensemble source is this module, not the synthesized one - so the
# module-global `x` this test is about cannot be set up.
def test_xfail_skipif_with_globals(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest
        x = 3
        @pytest.mark.skipif("x == 3")
        def test_skip1():
            pass
        @pytest.mark.xfail("x == 3")
        def test_boolean():
            assert 0
    """
    )
    result = pytester.runpytest("-rsx")
    result.stdout.fnmatch_lines(["*SKIP*x == 3*", "*XFAIL*test_boolean*x == 3*"])


# ensemble: `--markers` is served from pytest_cmdline_main, which an ensemble
# never runs.
def test_default_markers(pytester: Pytester) -> None:
    result = pytester.runpytest("--markers")
    result.stdout.fnmatch_lines(
        [
            "*skipif(condition, ..., [*], reason=...)*skip*",
            "*xfail(condition, ..., [*], reason=..., run=True, raises=None, strict=strict_xfail)*expected failure*",
        ]
    )


def test_xfail_test_setup_exception(tmp_path: Path) -> None:
    class ConftestPlugin:
        def pytest_runtest_setup(self):
            0 / 0  # noqa: B018

    @pytest.mark.xfail
    def test_func():
        assert 0

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
    record = run_tests(test_func, spec=spec)
    # Stronger than "xfailed in stdout and xpassed not in stdout", and covers
    # `ret == 0` too: an xfail is the only thing that happened.
    record.assert_outcomes(xfailed=1)


def test_imperativeskip_on_xfail_test(tmp_path: Path) -> None:
    class ConftestPlugin:
        def pytest_runtest_setup(self, item):
            pytest.skip("abc")

    @pytest.mark.xfail
    def test_that_fails():
        assert 0

    @pytest.mark.skipif("True")
    def test_hello():
        pass

    spec = ConfigSpec(
        rootpath=tmp_path, args=("-rsxX",), extra_plugins=(ConftestPlugin(),)
    )
    record = run_tests(test_that_fails, test_hello, spec=spec, capture_output=True)
    record.stdout.fnmatch_lines_random(
        """
        *SKIP*abc*
        *SKIP*condition: True*
        *2 skipped*
    """
    )
    record.assert_outcomes(skipped=2)


class TestBooleanCondition:
    def test_skipif(self, tmp_path: Path) -> None:
        @pytest.mark.skipif(True, reason="True123")
        def test_func1():
            pass

        @pytest.mark.skipif(False, reason="True123")
        def test_func2():
            pass

        record = run_tests(test_func1, test_func2, rootpath=tmp_path)
        record.assert_outcomes(passed=1, skipped=1)

    def test_skipif_noreason(self, tmp_path: Path) -> None:
        @pytest.mark.skipif(True)
        def test_func():
            pass

        record = run_tests(test_func, rootpath=tmp_path)
        # The missing-reason failure happens in setup, so it is an error.
        record.assert_outcomes(errors=1)

    def test_xfail(self, tmp_path: Path) -> None:
        @pytest.mark.xfail(True, reason="True123")
        def test_func():
            assert 0

        spec = ConfigSpec(rootpath=tmp_path, args=("-rxs",))
        record = run_tests(test_func, spec=spec, capture_output=True)
        record.stdout.fnmatch_lines(
            """
            *XFAIL*True123*
            *1 xfail*
        """
        )
        record.assert_outcomes(xfailed=1)


# ensemble: the item is produced by a `pytest_collect_file` hook, and an
# ensemble serves a preset collection tree instead of walking files.
def test_xfail_item(pytester: Pytester) -> None:
    # Ensure pytest.xfail works with non-Python Item
    pytester.makeconftest(
        """
        import pytest

        class MyItem(pytest.Item):
            nodeid = 'foo'
            def runtest(self):
                pytest.xfail("Expected Failure")

        def pytest_collect_file(file_path, parent):
            return MyItem.from_parent(name="foo", parent=parent)
    """
    )
    result = pytester.inline_run()
    _passed, skipped, failed = result.listoutcomes()
    assert not failed
    xfailed = [r for r in skipped if hasattr(r, "wasxfail")]
    assert xfailed


# ensemble: the skip has to happen while the module is being imported, and an
# ensemble module is handed over as an object rather than imported.
def test_module_level_skip_error(pytester: Pytester) -> None:
    """Verify that using pytest.skip at module level causes a collection error."""
    pytester.makepyfile(
        """
        import pytest
        pytest.skip("skip_module_level")

        def test_func():
            assert True
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        ["*Using pytest.skip outside of a test will skip the entire module*"]
    )


# ensemble: same - the skip is raised at module import time.
def test_module_level_skip_with_allow_module_level(pytester: Pytester) -> None:
    """Verify that using pytest.skip(allow_module_level=True) is allowed."""
    pytester.makepyfile(
        """
        import pytest
        pytest.skip("skip_module_level", allow_module_level=True)

        def test_func():
            assert 0
    """
    )
    result = pytester.runpytest("-rxs")
    result.stdout.fnmatch_lines(["*SKIP*skip_module_level"])


# ensemble: same - the TypeError is raised at module import time.
def test_invalid_skip_keyword_parameter(pytester: Pytester) -> None:
    """Verify that using pytest.skip() with unknown parameter raises an error."""
    pytester.makepyfile(
        """
        import pytest
        pytest.skip("skip_module_level", unknown=1)

        def test_func():
            assert 0
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*TypeError:*['unknown']*"])


# ensemble: the item is produced by a `pytest_collect_file` hook, and an
# ensemble serves a preset collection tree instead of walking files.
def test_mark_xfail_item(pytester: Pytester) -> None:
    # Ensure pytest.mark.xfail works with non-Python Item
    pytester.makeconftest(
        """
        import pytest

        class MyItem(pytest.Item):
            nodeid = 'foo'
            def setup(self):
                marker = pytest.mark.xfail("1 == 2", reason="Expected failure - false")
                self.add_marker(marker)
                marker = pytest.mark.xfail(True, reason="Expected failure - true")
                self.add_marker(marker)
            def runtest(self):
                assert False

        def pytest_collect_file(file_path, parent):
            return MyItem.from_parent(name="foo", parent=parent)
    """
    )
    result = pytester.inline_run()
    _passed, skipped, failed = result.listoutcomes()
    assert not failed
    xfailed = [r for r in skipped if hasattr(r, "wasxfail")]
    assert xfailed


def test_summary_list_after_errors(tmp_path: Path) -> None:
    """Ensure the list of errors/fails/xfails/skips appears after tracebacks in terminal reporting."""

    def test_fail():
        assert 0

    spec = ConfigSpec(rootpath=tmp_path, args=("-ra",))
    record = run_tests(
        build_module("test_summary_list_after_errors", test_fail=test_fail),
        spec=spec,
        capture_output=True,
    )
    record.stdout.fnmatch_lines(
        [
            "=* FAILURES *=",
            "*= short test summary info =*",
            "FAILED test_summary_list_after_errors.py::test_fail - assert 0",
        ]
    )


def test_importorskip() -> None:
    with pytest.raises(
        pytest.skip.Exception,
        match=r"^could not import 'doesnotexist': No module named .*",
    ):
        pytest.importorskip("doesnotexist")


# ensemble: asserts the skip's `tests/test_1.py:2` location relative to a
# `--rootdir` below it; both the real path layout and the location are
# host-anchored.
def test_relpath_rootdir(pytester: Pytester) -> None:
    pytester.makepyfile(
        **{
            "tests/test_1.py": """
        import pytest
        @pytest.mark.skip()
        def test_pass():
            pass
            """,
        }
    )
    result = pytester.runpytest("-rs", "tests/test_1.py", "--rootdir=tests")
    result.stdout.fnmatch_lines(
        ["SKIPPED [[]1[]] tests/test_1.py:2: unconditional skip"]
    )


# ensemble: same - the expected line pins `tests/test_1.py:2` as the reported
# skip location.
def test_skip_from_fixture(pytester: Pytester) -> None:
    pytester.makepyfile(
        **{
            "tests/test_1.py": """
        import pytest
        def test_pass(arg):
            pass
        @pytest.fixture
        def arg():
            condition = True
            if condition:
                pytest.skip("Fixture conditional skip")
            """,
        }
    )
    result = pytester.runpytest("-rs", "tests/test_1.py", "--rootdir=tests")
    result.stdout.fnmatch_lines(
        ["SKIPPED [[]1[]] tests/test_1.py:2: Fixture conditional skip"]
    )


def test_skip_using_reason_works_ok(tmp_path: Path) -> None:
    def test_skipping_reason():
        pytest.skip(reason="skippedreason")

    record = run_tests(test_skipping_reason, rootpath=tmp_path)
    # `warnings=0` is what the no_fnmatch_line on PytestDeprecationWarning was
    # after, checked against the recorded warnings rather than the rendering.
    record.assert_outcomes(skipped=1, warnings=0)


def test_fail_using_reason_works_ok(tmp_path: Path) -> None:
    def test_failing_reason():
        pytest.fail(reason="failedreason")

    record = run_tests(test_failing_reason, rootpath=tmp_path)
    record.assert_outcomes(failed=1, warnings=0)


def test_exit_with_reason_works_ok(tmp_path: Path) -> None:
    def test_exit_reason_only():
        pytest.exit(reason="foo")

    # An ensemble has no `wrap_session` catching Exit and rendering it, so the
    # exception itself is what the rendered `Exit: foo` line stood for.
    with pytest.raises(Exit, match=r"^foo$"):
        run_tests(test_exit_reason_only, rootpath=tmp_path)

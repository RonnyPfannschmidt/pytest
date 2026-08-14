# mypy: allow-untyped-defs
from __future__ import annotations

import io
from pathlib import Path

from _pytest.config import ExitCode
from _pytest.ensemble import build_module
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import configured
from _pytest.ensemble import Ensemble
from _pytest.ensemble import run_tests
from _pytest.ensemble import RunRecord
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester
import pytest


def run_testloop(ensemble: Ensemble, expect: type[BaseException]) -> RunRecord:
    """Drive the real ``pytest_runtestloop`` over an ensemble's items.

    ``run_items`` (and with it ``Ensemble.run``) runs every item it is given
    unconditionally; the early exit on ``session.shouldfail``/``shouldstop``
    lives in ``pytest_runtestloop``, so tests about ``--exitfirst``,
    ``--maxfail`` and friends have to go through that hook instead. The
    reports are recorded on the config either way, so an empty final ``run``
    hands back everything the loop produced before it bailed out.
    """
    ensemble.collect()
    session = ensemble.session
    with pytest.raises(expect):
        session.config.hook.pytest_runtestloop(session=session)
    return ensemble.run([])


class SessionTests:
    def test_basic_testitem_events(self, tmp_path: Path) -> None:
        def test_one():
            pass

        def test_one_one():
            assert 0

        def test_other():
            raise ValueError(23)

        class TestClass:
            def test_two(self, someargs):
                pass

        record = run_tests(
            test_one,
            test_one_one,
            test_other,
            TestClass,
            rootpath=tmp_path,
            name="test_basic_testitem_events",
        )
        # HookRecorder.listoutcomes counted all three of these as failed; the
        # missing ``someargs`` fixture fails at setup, which is a category of
        # its own here.
        record.assert_outcomes(passed=1, failed=2, errors=1)
        # ... and one item collected per test, in definition order.
        assert list(record.by_test) == [
            "test_basic_testitem_events.py::test_one",
            "test_basic_testitem_events.py::test_one_one",
            "test_basic_testitem_events.py::test_other",
            "test_basic_testitem_events.py::TestClass::test_two",
        ]

    # ensemble: the whole subject is a real module import failing on an import
    # of another real module - the chokepoint EnsembleModule bypasses.
    def test_nested_import_error(self, pytester: Pytester) -> None:
        tfile = pytester.makepyfile(
            """
            import import_fails
            def test_this():
                assert import_fails.a == 1
        """,
            import_fails="""
            import does_not_work
            a = 1
        """,
        )
        reprec = pytester.inline_run(tfile)
        values = reprec.getfailedcollections()
        assert len(values) == 1
        out = str(values[0].longrepr)
        assert out.find("does_not_work") != -1

    def test_raises_output(self, tmp_path: Path) -> None:
        def test_raises_doesnt():
            with pytest.raises(ValueError):
                int("3")

        record = run_tests(test_raises_doesnt, rootpath=tmp_path)
        record.assert_outcomes(failed=1)
        call = record["test_raises_doesnt"].call
        assert call is not None
        out = call.longrepr.reprcrash.message  # type: ignore[union-attr]
        assert "DID NOT RAISE" in out

    # ensemble: a module that is not python at all only exists as a file that
    # fails to compile on import.
    def test_syntax_error_module(self, pytester: Pytester) -> None:
        reprec = pytester.inline_runsource("this is really not python")
        values = reprec.getfailedcollections()
        assert len(values) == 1
        out = str(values[0].longrepr)
        assert out.find("not python") != -1

    def test_exit_first_problem(self, tmp_path: Path) -> None:
        def test_one():
            assert 0

        def test_two():
            assert 0

        spec = ConfigSpec(rootpath=tmp_path, args=("--exitfirst",))
        with Ensemble(test_one, test_two, spec=spec, name="test_exitfirst") as ensemble:
            record = run_testloop(ensemble, Session.Failed)
            assert ensemble.session.shouldfail == "stopping after 1 failures"
        record.assert_outcomes(failed=1)
        # the second test never ran
        assert list(record.by_test) == ["test_exitfirst.py::test_one"]

    def test_maxfail(self, tmp_path: Path) -> None:
        def test_one():
            assert 0

        def test_two():
            assert 0

        def test_three():
            assert 0

        spec = ConfigSpec(rootpath=tmp_path, args=("--maxfail=2",))
        with Ensemble(
            test_one, test_two, test_three, spec=spec, name="test_maxfail"
        ) as ensemble:
            record = run_testloop(ensemble, Session.Failed)
            assert ensemble.session.shouldfail == "stopping after 2 failures"
        record.assert_outcomes(failed=2)
        # the third test never ran
        assert list(record.by_test) == [
            "test_maxfail.py::test_one",
            "test_maxfail.py::test_two",
        ]

    def test_broken_repr(self, tmp_path: Path) -> None:
        class reprexc(BaseException):
            def __str__(self):
                return "Ha Ha fooled you, I'm a broken repr()."

        class BrokenRepr1:
            foo = 0

            def __repr__(self):
                raise reprexc

        class TestBrokenClass:
            def test_explicit_bad_repr(self):
                t = BrokenRepr1()
                with pytest.raises(BaseException, match="broken repr"):
                    repr(t)

            def test_implicit_bad_repr1(self):
                t = BrokenRepr1()
                assert t.foo == 1

        record = run_tests(TestBrokenClass, rootpath=tmp_path, name="test_broken_repr")
        record.assert_outcomes(passed=1, failed=1)
        call = record["test_implicit_bad_repr1"].call
        assert call is not None
        out = call.longrepr.reprcrash.message  # type: ignore[union-attr]
        assert out.find("<[reprexc() raised in repr()] BrokenRepr1") != -1

    def test_broken_repr_with_showlocals_verbose(self, tmp_path: Path) -> None:
        class ObjWithErrorInRepr:
            def __repr__(self):
                raise NotImplementedError

        def test_repr_error():
            x = ObjWithErrorInRepr()
            assert x == "value"  # type: ignore[comparison-overlap]

        # --showlocals and -vv are terminal options, so the terminal plugin
        # has to be loaded; capture_output binds its stream to a private
        # buffer instead of the outer stdout.
        spec = ConfigSpec(rootpath=tmp_path, args=("--showlocals", "-vv"))
        record = run_tests(test_repr_error, spec=spec, capture_output=True)
        record.assert_outcomes(failed=1)
        call = record["test_repr_error"].call
        assert call is not None
        entries = call.longrepr.reprtraceback.reprentries  # type: ignore[union-attr]
        assert len(entries) == 1
        repr_locals = entries[0].reprlocals
        assert repr_locals.lines
        # ObjWithErrorInRepr is a closure cell here - a module global, and so
        # not a local at all, in the original - hence the extra line.
        assert len(repr_locals.lines) == 2
        assert repr_locals.lines[-1].startswith(
            "x          = <[NotImplementedError() raised in repr()] ObjWithErrorInRepr"
        )

    # ensemble: pytest_collect_file never fires for an ensemble - its
    # collection tree is preset, no path is ever walked.
    def test_skip_file_by_conftest(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            conftest="""
            import pytest
            def pytest_collect_file():
                pytest.skip("intentional")
        """,
            test_file="""
            def test_one(): pass
        """,
        )
        try:
            reprec = pytester.inline_run(pytester.path)
        except pytest.skip.Exception:  # pragma: no cover
            pytest.fail("wrong skipped caught")
        reports = reprec.getreports("pytest_collectreport")
        # Session, Dir
        assert len(reports) == 2
        assert reports[1].skipped


class TestNewSession(SessionTests):
    def test_order_of_execution(self, tmp_path: Path) -> None:
        # a closure rather than a module global: ensemble sources keep this
        # file's globals, which are emphatically not the ensemble's.
        values: list[int] = []

        def test_1():
            values.append(1)

        def test_2():
            values.append(2)

        def test_3():
            assert values == [1, 2]

        class Testmygroup:
            reslist = values

            def test_1(self):
                self.reslist.append(1)

            def test_2(self):
                self.reslist.append(2)

            def test_3(self):
                self.reslist.append(3)

            def test_4(self):
                assert self.reslist == [1, 2, 1, 2, 3]

        record = run_tests(
            test_1, test_2, test_3, Testmygroup, rootpath=tmp_path, name="test_order"
        )
        record.assert_outcomes(passed=7)

    # ensemble: a directory argument, an __init__.py and a module that is not
    # python - all filesystem, plus the collect event counting is about the
    # tree an ensemble presets rather than walks.
    def test_collect_only_with_various_situations(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            test_one="""
                def test_one():
                    raise ValueError()

                class TestX(object):
                    def test_method_one(self):
                        pass

                class TestY(TestX):
                    pass
            """,
            test_three="xxxdsadsadsadsa",
            __init__="",
        )
        reprec = pytester.inline_run("--collect-only", p.parent)

        itemstarted = reprec.getcalls("pytest_itemcollected")
        assert len(itemstarted) == 3
        assert not reprec.getreports("pytest_runtest_logreport")
        started = reprec.getcalls("pytest_collectstart")
        finished = reprec.getreports("pytest_collectreport")
        assert len(started) == len(finished)
        assert len(started) == 6
        colfail = [x for x in finished if x.failed]
        assert len(colfail) == 1

    # ensemble: import errors of real files, collected via a directory
    # argument.
    def test_minus_x_import_error(self, pytester: Pytester) -> None:
        pytester.makepyfile(__init__="")
        pytester.makepyfile(test_one="xxxx", test_two="yyyy")
        reprec = pytester.inline_run("-x", pytester.path)
        finished = reprec.getreports("pytest_collectreport")
        colfail = [x for x in finished if x.failed]
        assert len(colfail) == 1

    # ensemble: as above - import errors of real files behind a directory
    # argument.
    def test_minus_x_overridden_by_maxfail(self, pytester: Pytester) -> None:
        pytester.makepyfile(__init__="")
        pytester.makepyfile(test_one="xxxx", test_two="yyyy", test_third="zzz")
        reprec = pytester.inline_run("-x", "--maxfail=2", pytester.path)
        finished = reprec.getreports("pytest_collectreport")
        colfail = [x for x in finished if x.failed]
        assert len(colfail) == 2


# ensemble: ``-p`` is handled by PytestPluginManager.consider_preparse, which
# is part of Config._preparse; an ensemble config builds its plugin set from
# the spec instead and never runs it, so ``args=("-p", ...)`` lands in
# config.option.plugins and is then ignored - the test would pass without
# testing anything.
def test_plugin_specify(pytester: Pytester) -> None:
    with pytest.raises(ImportError):
        pytester.parseconfig("-p", "nqweotexistent")
    # pytest.raises(ImportError,
    #    "config.do_configure(config)"
    # )


# ensemble: same as above - ``-p`` is never consumed by an ensemble config.
def test_plugin_already_exists(tmp_path: Path) -> None:
    # ``-p terminal`` names a plugin that is loaded already; configure and
    # unconfigure must both survive it. The stream is a private buffer, since
    # a loaded terminal plugin would otherwise bind the outer test's stdout.
    spec = ConfigSpec(
        rootpath=tmp_path, args=("-p", "terminal"), output=io.StringIO()
    ).with_plugins("terminal")
    with configured(spec) as config:
        assert config.option.plugins == ["terminal"]


# ensemble: --ignore excludes filesystem paths from a directory walk.
def test_exclude(pytester: Pytester) -> None:
    hellodir = pytester.mkdir("hello")
    hellodir.joinpath("test_hello.py").write_text("x y syntaxerror", encoding="utf-8")
    hello2dir = pytester.mkdir("hello2")
    hello2dir.joinpath("test_hello2.py").write_text("x y syntaxerror", encoding="utf-8")
    pytester.makepyfile(test_ok="def test_pass(): pass")
    result = pytester.runpytest("--ignore=hello", "--ignore=hello2")
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*1 passed*"])


# ensemble: as above, --ignore-glob is about the directory walk.
def test_exclude_glob(pytester: Pytester) -> None:
    hellodir = pytester.mkdir("hello")
    hellodir.joinpath("test_hello.py").write_text("x y syntaxerror", encoding="utf-8")
    hello2dir = pytester.mkdir("hello2")
    hello2dir.joinpath("test_hello2.py").write_text("x y syntaxerror", encoding="utf-8")
    hello3dir = pytester.mkdir("hallo3")
    hello3dir.joinpath("test_hello3.py").write_text("x y syntaxerror", encoding="utf-8")
    subdir = pytester.mkdir("sub")
    subdir.joinpath("test_hello4.py").write_text("x y syntaxerror", encoding="utf-8")
    pytester.makepyfile(test_ok="def test_pass(): pass")
    result = pytester.runpytest("--ignore-glob=*h[ea]llo*")
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*1 passed*"])


def test_deselect(tmp_path: Path) -> None:
    def test_a1():
        pass

    @pytest.mark.parametrize("b", range(3))
    def test_a2(b):
        pass

    class TestClass:
        def test_c1(self):
            pass

        def test_c2(self):
            pass

    module = build_module("test_a", test_a1, test_a2, TestClass)
    spec = ConfigSpec(
        rootpath=tmp_path,
        args=(
            "-v",
            "--deselect=test_a.py::test_a2[1]",
            "--deselect=test_a.py::test_a2[2]",
            "--deselect=test_a.py::TestClass::test_c1",
        ),
    )
    record = run_tests(module, spec=spec, capture_output=True)
    record.assert_outcomes(passed=3, deselected=3)
    # the verbose listing really is there, so the loop below has something to
    # not find
    record.stdout.fnmatch_lines(["test_a.py::test_a2[[]0[]] PASSED*"])
    for line in record.stdout.lines:
        assert not line.startswith(("test_a.py::test_a2[1]", "test_a.py::test_a2[2]"))


# ensemble: the cwd is restored by wrap_session, which an ensemble does not go
# through (and the subject of the assertion is the exit status).
def test_sessionfinish_with_start(pytester: Pytester) -> None:
    pytester.makeconftest(
        """
        import os
        values = []
        def pytest_sessionstart():
            values.append(os.getcwd())
            os.chdir("..")

        def pytest_sessionfinish():
            assert values[0] == os.getcwd()

    """
    )
    res = pytester.runpytest("--collect-only")
    assert res.ret == ExitCode.NO_TESTS_COLLECTED


# ensemble: collection driven by path arguments (and --keep-duplicates over a
# directory given twice); an ensemble's collection tree is preset, so there
# are no initial arguments to deduplicate.
def test_collection_args_do_not_duplicate_modules(pytester: Pytester) -> None:
    """Test that when multiple collection args are specified on the command line
    for the same module, only a single Module collector is created.

    Regression test for #723, #3358.
    """
    pytester.makepyfile(
        **{
            "d/test_it": """
                def test_1(): pass
                def test_2(): pass
                """
        }
    )

    result = pytester.runpytest(
        "--collect-only",
        "d/test_it.py::test_1",
        "d/test_it.py::test_2",
    )
    result.stdout.fnmatch_lines(
        [
            "  <Dir d>",
            "    <Module test_it.py>",
            "      <Function test_1>",
            "      <Function test_2>",
        ],
        consecutive=True,
    )

    # Different, but related case.
    result = pytester.runpytest(
        "--collect-only",
        "--keep-duplicates",
        "d",
        "d",
    )
    result.stdout.fnmatch_lines(
        [
            "  <Dir d>",
            "    <Module test_it.py>",
            "      <Function test_1>",
            "      <Function test_2>",
            "      <Function test_1>",
            "      <Function test_2>",
        ],
        consecutive=True,
    )


# ensemble: --rootdir feeds rootdir *discovery*, which an ensemble config
# skips entirely - its rootpath is handed to it.
@pytest.mark.parametrize("path", ["root", "{relative}/root", "{environment}/root"])
def test_rootdir_option_arg(
    pytester: Pytester, monkeypatch: MonkeyPatch, path: str
) -> None:
    monkeypatch.setenv("PY_ROOTDIR_PATH", str(pytester.path))
    path = path.format(relative=str(pytester.path), environment="$PY_ROOTDIR_PATH")

    rootdir = pytester.path / "root" / "tests"
    rootdir.mkdir(parents=True)
    pytester.makepyfile(
        """
        import os
        def test_one():
            assert 1
    """
    )

    result = pytester.runpytest(f"--rootdir={path}")
    result.stdout.fnmatch_lines(
        [
            f"*rootdir: {pytester.path}/root",
            "root/test_rootdir_option_arg.py *",
            "*1 passed*",
        ]
    )


# ensemble: as above, plus the assertion is on stderr of a real run.
def test_rootdir_wrong_option_arg(pytester: Pytester) -> None:
    result = pytester.runpytest("--rootdir=wrong_dir")
    result.stderr.fnmatch_lines(
        ["*Directory *wrong_dir* not found. Check your '--rootdir' option.*"]
    )


def test_shouldfail_is_sticky(tmp_path: Path) -> None:
    """Test that session.shouldfail cannot be reset to False after being set.

    Issue #11706.
    """
    recorded_warnings: list[str] = []

    class ConftestPlugin:
        def pytest_sessionfinish(self, session):
            assert session.shouldfail
            session.shouldfail = False
            assert session.shouldfail

        # the RunRecord is built before pytest_sessionfinish runs, so the
        # warning raised in there is recorded here instead.
        def pytest_warning_recorded(self, warning_message):
            recorded_warnings.append(str(warning_message.message))

    def test_foo():
        pytest.fail("This is a failing test")

    def test_bar():
        pass

    spec = ConfigSpec(
        rootpath=tmp_path,
        args=("--maxfail=1",),
        # -Wall in the original; the host's ``filterwarnings = error`` would
        # otherwise turn the warning into an exception out of sessionfinish.
        inicfg={"filterwarnings": ["always"]},
        extra_plugins=(ConftestPlugin(),),
    )
    with Ensemble(test_foo, test_bar, spec=spec, name="test_shouldfail") as ensemble:
        record = run_testloop(ensemble, Session.Failed)

    record.assert_outcomes(failed=1)
    assert recorded_warnings == [
        "session.shouldfail cannot be unset after it has been set; ignoring."
    ]


def test_shouldstop_is_sticky(tmp_path: Path) -> None:
    """Test that session.shouldstop cannot be reset to False after being set.

    Issue #11706.
    """
    recorded_warnings: list[str] = []

    class ConftestPlugin:
        session: Session

        def pytest_sessionstart(self, session):
            self.session = session

        # --stepwise sets shouldstop in the original; the stepwise plugin
        # needs the cache and terminal plugins, and what is under test is the
        # setter, so the flag is set the same way stepwise sets it.
        def pytest_runtest_logreport(self, report):
            if report.failed:
                self.session.shouldstop = (
                    "Test failed, continuing from this test next run."
                )

        def pytest_sessionfinish(self, session):
            assert session.shouldstop
            session.shouldstop = False
            assert session.shouldstop

        def pytest_warning_recorded(self, warning_message):
            recorded_warnings.append(str(warning_message.message))

    def test_foo():
        pytest.fail("This is a failing test")

    def test_bar():
        pass

    spec = ConfigSpec(
        rootpath=tmp_path,
        inicfg={"filterwarnings": ["always"]},
        extra_plugins=(ConftestPlugin(),),
    )
    with Ensemble(test_foo, test_bar, spec=spec, name="test_shouldstop") as ensemble:
        record = run_testloop(ensemble, Session.Interrupted)

    record.assert_outcomes(failed=1)
    assert recorded_warnings == [
        "session.shouldstop cannot be unset after it has been set; ignoring."
    ]

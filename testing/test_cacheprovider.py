# mypy: allow-untyped-defs
from __future__ import annotations

from collections.abc import Generator
from collections.abc import Sequence
from enum import auto
from enum import Enum
import os
from pathlib import Path
import shutil
import types
from typing import Any

from _pytest.compat import assert_never
from _pytest.ensemble import build_module
from _pytest.ensemble import collect_tests
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import configured
from _pytest.ensemble import Ensemble
from _pytest.ensemble import module_from_path
from _pytest.ensemble import run_tests
from _pytest.ensemble.collection import ensemble_collection
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester
from _pytest.tmpdir import TempPathFactory
import pytest


pytest_plugins = ("pytester",)


def cache_spec(rootpath: Path, *args: str, **inicfg: object) -> ConfigSpec:
    """A nested config with the cacheprovider plugin loaded.

    ``cacheprovider`` is not in the ensemble default plugin set, but loading
    it needs nothing else: the cache lives under the config's rootdir, and an
    ensemble's rootdir is a real directory. Two ensembles built from the same
    *rootpath* therefore share one cache, which is what makes ``--lf``/``--ff``
    expressible as "run the same spec twice".
    """
    return ConfigSpec(rootpath=rootpath, args=args, inicfg=inicfg).with_plugins(
        "cacheprovider"
    )


def write_source(rootpath: Path, relpath: str, source: str) -> Path:
    """Write a test module to disk under *rootpath* and return its path.

    Sources that must exist as files are the ones whose *path* is part of what
    is being tested: ``--lf`` skips collecting whole files by path, and ``--nf``
    orders them by mtime. Both need a path that exists, which a synthesized
    in-memory module deliberately does not have (see :func:`in_memory`).
    """
    path = rootpath.joinpath(relpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def in_memory(name: str, *members: object) -> types.ModuleType:
    """A synthesized module whose path under the rootdir does not exist.

    The last-failed collection wrapper only filters/skips *files that exist*,
    so an in-memory module always reaches ``pytest_collection_modifyitems``
    with all of its items - the same way a file named on the command line does
    in a real run, because it is then an initial path. This is what keeps the
    "N deselected" assertions of the original tests meaningful.
    """
    return build_module(name, *members)


class TestNewAPI:
    def test_config_cache_mkdir(self, tmp_path: Path) -> None:
        with configured(cache_spec(tmp_path)) as config:
            assert config.cache is not None
            with pytest.raises(ValueError):
                config.cache.mkdir("key/name")

            p = config.cache.mkdir("name")
            assert p.is_dir()

    def test_cache_dir_permissions(self, tmp_path: Path) -> None:
        """The .pytest_cache directory should have world-readable permissions
        (depending on umask).

        Regression test for #12308.
        """
        with configured(cache_spec(tmp_path)) as config:
            assert config.cache is not None
            p = config.cache.mkdir("name")
            assert p.is_dir()
            # Instead of messing with umask, make sure .pytest_cache has the same
            # permissions as the default that `mkdir` gives `p`.
            assert (p.parent.stat().st_mode & 0o777) == (p.stat().st_mode & 0o777)

    def test_config_cache_dataerror(self, tmp_path: Path) -> None:
        with configured(cache_spec(tmp_path)) as config:
            assert config.cache is not None
            cache = config.cache
            with pytest.raises(TypeError):
                cache.set("key/name", cache)
            config.cache.set("key/name", 0)
            config.cache._getvaluepath("key/name").write_bytes(b"123invalid")
            val = config.cache.get("key/name", -2)
            assert val == -2

    @pytest.mark.filterwarnings("ignore:could not create cache path")
    def test_cache_writefail_cachefile_silent(self, tmp_path: Path) -> None:
        tmp_path.joinpath(".pytest_cache").write_text("gone wrong", encoding="utf-8")
        with configured(cache_spec(tmp_path)) as config:
            cache = config.cache
            assert cache is not None
            cache.set("test/broken", [])

    @pytest.fixture
    def unwritable_cache_dir(self, tmp_path: Path) -> Generator[Path]:
        cache_dir = tmp_path.joinpath(".pytest_cache")
        cache_dir.mkdir()
        mode = cache_dir.stat().st_mode
        cache_dir.chmod(0)
        if os.access(cache_dir, os.W_OK):
            pytest.skip("Failed to make cache dir unwritable")

        yield cache_dir
        cache_dir.chmod(mode)

    @pytest.mark.filterwarnings(
        "ignore:could not create cache path:pytest.PytestWarning"
    )
    def test_cache_writefail_permissions(
        self, unwritable_cache_dir: Path, tmp_path: Path
    ) -> None:
        with configured(cache_spec(tmp_path)) as config:
            cache = config.cache
            assert cache is not None
            cache.set("test/broken", [])

    def test_cache_failure_warns(
        self,
        tmp_path: Path,
        unwritable_cache_dir: Path,
    ) -> None:
        # The original disabled plugin autoloading so that no other plugin
        # could add warnings; an ensemble never autoloads anything. The host's
        # ``filterwarnings = error`` is what the ensemble's own ini overrides.
        def test_error():
            raise Exception

        record = run_tests(
            test_error,
            spec=cache_spec(tmp_path, filterwarnings=["always"]),
            capture_output=True,
        )
        # warnings from nodeids and lastfailed
        record.assert_outcomes(failed=1, warnings=2)
        record.stdout.fnmatch_lines(
            [
                # Validate location/stacklevel of warning from cacheprovider.
                "*= warnings summary =*",
                "*/cacheprovider.py:*",
                "  */cacheprovider.py:*: PytestCacheWarning: could not create cache path "
                f"{unwritable_cache_dir}/v/cache/nodeids: *",
                '    config.cache.set("cache/nodeids", sorted(self.cached_nodeids))',
                "*1 failed, 2 warnings in*",
            ]
        )

    def test_config_cache(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            def pytest_configure(self, config):
                # see that we get cache information early on
                assert hasattr(config, "cache")

        def test_session(pytestconfig):
            assert hasattr(pytestconfig, "cache")

        record = run_tests(
            test_session,
            spec=cache_spec(tmp_path).replace(extra_plugins=(ConftestPlugin(),)),
            capture_output=True,
        )
        record.assert_outcomes(passed=1)
        record.stdout.fnmatch_lines(["*1 passed*"])

    def test_cachefuncarg(self, tmp_path: Path) -> None:
        def test_cachefuncarg(cache):
            val = cache.get("some/thing", None)
            assert val is None
            cache.set("some/thing", [1])
            with pytest.raises(TypeError):
                cache.get("some/thing")
            val = cache.get("some/thing", [])
            assert val == [1]

        record = run_tests(
            test_cachefuncarg, spec=cache_spec(tmp_path), capture_output=True
        )
        record.assert_outcomes(passed=1)
        record.stdout.fnmatch_lines(["*1 passed*"])

    def test_custom_rel_cache_dir(self, tmp_path: Path) -> None:
        rel_cache_dir = os.path.join("custom_cache_dir", "subdir")

        def test_error():
            assert False

        run_tests(
            test_error,
            spec=cache_spec(tmp_path, cache_dir=rel_cache_dir),
            name="test_errored",
        )
        assert tmp_path.joinpath(rel_cache_dir).is_dir()

    def test_custom_abs_cache_dir(
        self, tmp_path: Path, tmp_path_factory: TempPathFactory
    ) -> None:
        tmp = tmp_path_factory.mktemp("tmp")
        abs_cache_dir = tmp / "custom_cache_dir"

        def test_error():
            assert False

        run_tests(
            test_error,
            spec=cache_spec(tmp_path, cache_dir=str(abs_cache_dir)),
            name="test_errored",
        )
        assert abs_cache_dir.is_dir()

    def test_custom_cache_dir_with_env_var(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("env_var", "custom_cache_dir")

        def test_error():
            assert False

        run_tests(
            test_error,
            spec=cache_spec(tmp_path, cache_dir="$env_var"),
            name="test_errored",
        )
        assert tmp_path.joinpath("custom_cache_dir").is_dir()


@pytest.mark.parametrize("env", ((), ("TOX_ENV_DIR", "mydir/tox-env")))
def test_cache_reportheader(
    env: Sequence[str], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    def test_foo():
        pass

    if env:
        monkeypatch.setenv(*env)
        expected = os.path.join(env[1], ".pytest_cache")
    else:
        monkeypatch.delenv("TOX_ENV_DIR", raising=False)
        expected = ".pytest_cache"
    record = run_tests(test_foo, spec=cache_spec(tmp_path, "-v"), capture_output=True)
    record.stdout.fnmatch_lines([f"cachedir: {expected}"])


def test_cache_reportheader_external_abspath(
    tmp_path: Path, tmp_path_factory: TempPathFactory
) -> None:
    external_cache = tmp_path_factory.mktemp(
        "test_cache_reportheader_external_abspath_abs"
    )

    def test_hello():
        pass

    record = run_tests(
        test_hello,
        spec=cache_spec(tmp_path, "-v", cache_dir=str(external_cache)),
        capture_output=True,
    )
    record.stdout.fnmatch_lines([f"cachedir: {external_cache}"])


def test_cache_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # ensemble: ``--cache-show`` is served from ``pytest_cmdline_main``, which
    # an ensemble never runs - so the reporting function is driven directly,
    # against a config that has the option set. It writes to a TerminalWriter
    # of its own, i.e. to the host's stdout, hence capsys rather than
    # capture_output.
    from _pytest.cacheprovider import cacheshow

    with Ensemble(spec=cache_spec(tmp_path, "--cache-show")) as ensemble:
        assert cacheshow(ensemble.config, ensemble.session) == 0
    result = capsys.readouterr().out
    assert "cache is empty" in result

    class ConftestPlugin:
        def pytest_configure(self, config):
            config.cache.set("my/name", [1, 2, 3])
            config.cache.set("my/hello", "world")
            config.cache.set("other/some", {1: 2})
            dp = config.cache.mkdir("mydb")
            dp.joinpath("hello").touch()
            dp.joinpath("world").touch()

    record = run_tests(
        spec=cache_spec(tmp_path).replace(extra_plugins=(ConftestPlugin(),))
    )
    assert record.outcomes() == {}  # no tests executed

    with Ensemble(spec=cache_spec(tmp_path, "--cache-show")) as ensemble:
        assert cacheshow(ensemble.config, ensemble.session) == 0
    matcher = pytest.LineMatcher(capsys.readouterr().out.splitlines())
    matcher.fnmatch_lines(
        [
            "*cachedir:*",
            "*- cache values for '[*]' -*",
            "cache/nodeids contains:",
            "my/name contains:",
            "  [1, 2, 3]",
            "other/some contains:",
            "  {*'1': 2}",
            "*- cache directories for '[*]' -*",
            "*mydb/hello*length 0*",
            "*mydb/world*length 0*",
        ]
    )

    with Ensemble(spec=cache_spec(tmp_path, "--cache-show", "*/hello")) as ensemble:
        assert cacheshow(ensemble.config, ensemble.session) == 0
    stdout = capsys.readouterr().out
    matcher = pytest.LineMatcher(stdout.splitlines())
    matcher.fnmatch_lines(
        [
            "*cachedir:*",
            "*- cache values for '[*]/hello' -*",
            "my/hello contains:",
            "  *'world'",
            "*- cache directories for '[*]/hello' -*",
            "d/mydb/hello*length 0*",
        ]
    )
    assert "other/some" not in stdout
    assert "d/mydb/world" not in stdout


class TestLastFailed:
    def test_lastfailed_usecase(self, tmp_path: Path) -> None:
        def failing() -> types.ModuleType:
            def test_1():
                assert 0

            def test_2():
                assert 0

            def test_3():
                assert 1

            return in_memory("test_lastfailed_usecase", test_1, test_2, test_3)

        def fixed() -> types.ModuleType:
            def test_1():
                assert 1

            def test_2():
                assert 1

            def test_3():
                assert 0

            return in_memory("test_lastfailed_usecase", test_1, test_2, test_3)

        record = run_tests(failing(), spec=cache_spec(tmp_path), capture_output=True)
        record.stdout.fnmatch_lines(["*2 failed*"])
        record = run_tests(
            fixed(), spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "collected 3 items / 1 deselected / 2 selected",
                "run-last-failure: rerun previous 2 failures",
                "*= 2 passed, 1 deselected in *",
            ]
        )
        record = run_tests(
            fixed(), spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "collected 3 items",
                "run-last-failure: no previously failed tests, not deselecting items.",
                "*1 failed*2 passed*",
            ]
        )
        tmp_path.joinpath(".pytest_cache", ".git").mkdir(parents=True)
        record = run_tests(
            fixed(),
            spec=cache_spec(tmp_path, "--lf", "--cache-clear"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*1 failed*2 passed*"])
        assert tmp_path.joinpath(".pytest_cache", "README.md").is_file()
        assert tmp_path.joinpath(".pytest_cache", ".git").is_dir()

        # Run this again to make sure clear-cache is robust
        shutil.rmtree(tmp_path / ".pytest_cache")
        record = run_tests(
            fixed(),
            spec=cache_spec(tmp_path, "--lf", "--cache-clear"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*1 failed*2 passed*"])

    def test_failedfirst_order(self, tmp_path: Path) -> None:
        def test_always_passes():
            pass

        def test_always_fails():
            assert 0

        test_a = in_memory("test_a", test_always_passes)
        test_b = in_memory("test_b", test_always_fails)

        record = run_tests(
            test_a, test_b, spec=cache_spec(tmp_path), capture_output=True
        )
        # Test order will be collection order; alphabetical
        record.stdout.fnmatch_lines(["test_a.py*", "test_b.py*"])
        record = run_tests(
            test_a, test_b, spec=cache_spec(tmp_path, "--ff"), capture_output=True
        )
        # Test order will be failing tests first
        record.stdout.fnmatch_lines(
            [
                "collected 2 items",
                "run-last-failure: rerun previous 1 failure first",
                "test_b.py*",
                "test_a.py*",
            ]
        )
        assert [
            report.nodeid for report in record.reports if report.when == "call"
        ] == [
            "test_b.py::test_always_fails",
            "test_a.py::test_always_passes",
        ]

    def test_lastfailed_failedfirst_order(self, tmp_path: Path) -> None:
        def test_always_passes():
            assert 1

        def test_always_fails():
            assert 0

        test_a = in_memory("test_a", test_always_passes)
        test_b = in_memory("test_b", test_always_fails)

        record = run_tests(
            test_a, test_b, spec=cache_spec(tmp_path), capture_output=True
        )
        # Test order will be collection order; alphabetical
        record.stdout.fnmatch_lines(["test_a.py*", "test_b.py*"])
        record = run_tests(
            test_a,
            test_b,
            spec=cache_spec(tmp_path, "--lf", "--ff"),
            capture_output=True,
        )
        # Test order will be failing tests first
        record.stdout.fnmatch_lines(["test_b.py*"])
        record.stdout.no_fnmatch_line("*test_a.py*")

    def test_lastfailed_difference_invocations(self, tmp_path: Path) -> None:
        def test_a1():
            assert 0

        def test_a2():
            assert 1

        test_a = in_memory("test_a", test_a1, test_a2)

        def failing_b() -> types.ModuleType:
            def test_b1():
                assert 0

            return in_memory("test_b", test_b1)

        def fixed_b() -> types.ModuleType:
            def test_b1():
                assert 1

            return in_memory("test_b", test_b1)

        record = run_tests(
            test_a, failing_b(), spec=cache_spec(tmp_path), capture_output=True
        )
        record.stdout.fnmatch_lines(["*2 failed*"])
        # Selecting a subset is expressed by handing the ensemble fewer sources.
        record = run_tests(
            failing_b(), spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(["*1 failed*"])

        record = run_tests(
            fixed_b(), spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(["*1 passed*"])
        record = run_tests(
            test_a, spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "collected 2 items / 1 deselected / 1 selected",
                "run-last-failure: rerun previous 1 failure",
                "*= 1 failed, 1 deselected in *",
            ]
        )

    def test_lastfailed_usecase_splice(self, tmp_path: Path) -> None:
        def test_1():
            assert 0

        def test_2():
            assert 0

        main = in_memory("test_lastfailed_usecase_splice", test_1)
        other = in_memory("test_something", test_2)

        record = run_tests(main, other, spec=cache_spec(tmp_path), capture_output=True)
        record.stdout.fnmatch_lines(["*2 failed*"])
        record = run_tests(
            other, spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(["*1 failed*"])
        record = run_tests(
            main, other, spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(["*2 failed*"])

    def test_lastfailed_xpass(self, tmp_path: Path) -> None:
        @pytest.mark.xfail
        def test_hello():
            assert 1

        run_tests(test_hello, spec=cache_spec(tmp_path)).assert_outcomes(xpassed=1)
        with configured(cache_spec(tmp_path)) as config:
            assert config.cache is not None
            lastfailed = config.cache.get("cache/lastfailed", -1)
            assert lastfailed == -1

    def test_non_serializable_parametrize(self, tmp_path: Path) -> None:
        """Test that failed parametrized tests with unmarshable parameters
        don't break pytest-cache.
        """

        @pytest.mark.parametrize(
            "val",
            [
                b"\xac\x10\x02G",
            ],
        )
        def test_fail(val):
            assert False

        record = run_tests(test_fail, spec=cache_spec(tmp_path), capture_output=True)
        record.stdout.fnmatch_lines(["*1 failed in*"])
        # The unmarshable parameter must not have kept the cache from being
        # written at all.
        assert (tmp_path / ".pytest_cache/v/cache/lastfailed").is_file()

    # ensemble: the "package" parametrization needs a real ``__init__.py``
    # package, and an ensemble's collection tree has no Package (or Dir) nodes
    # at all - every module is a direct child of the session.
    @pytest.mark.parametrize("parent", ("directory", "package"))
    def test_terminal_report_lastfailed(self, pytester: Pytester, parent: str) -> None:
        if parent == "package":
            pytester.makepyfile(
                __init__="",
            )

        test_a = pytester.makepyfile(
            test_a="""
            def test_a1(): pass
            def test_a2(): pass
        """
        )
        test_b = pytester.makepyfile(
            test_b="""
            def test_b1(): assert 0
            def test_b2(): assert 0
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["collected 4 items", "*2 failed, 2 passed in*"])

        result = pytester.runpytest("--lf")
        result.stdout.fnmatch_lines(
            [
                "collected 2 items",
                "run-last-failure: rerun previous 2 failures (skipped 1 file)",
                "*2 failed in*",
            ]
        )

        result = pytester.runpytest(test_a, "--lf")
        result.stdout.fnmatch_lines(
            [
                "collected 2 items",
                "run-last-failure: 2 known failures not in selected tests",
                "*2 passed in*",
            ]
        )

        result = pytester.runpytest(test_b, "--lf")
        result.stdout.fnmatch_lines(
            [
                "collected 2 items",
                "run-last-failure: rerun previous 2 failures",
                "*2 failed in*",
            ]
        )

        result = pytester.runpytest("test_b.py::test_b1", "--lf")
        result.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "run-last-failure: rerun previous 1 failure",
                "*1 failed in*",
            ]
        )

    def test_terminal_report_failedfirst(self, tmp_path: Path) -> None:
        def test_a1():
            assert 0

        def test_a2():
            pass

        test_a = in_memory("test_a", test_a1, test_a2)

        record = run_tests(test_a, spec=cache_spec(tmp_path), capture_output=True)
        record.stdout.fnmatch_lines(["collected 2 items", "*1 failed, 1 passed in*"])

        record = run_tests(
            test_a, spec=cache_spec(tmp_path, "--ff"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "collected 2 items",
                "run-last-failure: rerun previous 1 failure first",
                "*1 failed, 1 passed in*",
            ]
        )

    # ensemble: the subject is a module that raises at *import* time; ensemble
    # sources are real objects that were imported by the host, so there is no
    # import of them left to fail.
    def test_lastfailed_collectfailure(
        self, pytester: Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        pytester.makepyfile(
            test_maybe="""
            import os
            env = os.environ
            if '1' == env['FAILIMPORT']:
                raise ImportError('fail')
            def test_hello():
                assert '0' == env['FAILTEST']
        """
        )

        def rlf(fail_import: int, fail_run: int) -> Any:
            monkeypatch.setenv("FAILIMPORT", str(fail_import))
            monkeypatch.setenv("FAILTEST", str(fail_run))

            pytester.runpytest("-q")
            config = pytester.parseconfigure()
            assert config.cache is not None
            lastfailed = config.cache.get("cache/lastfailed", -1)
            return lastfailed

        lastfailed = rlf(fail_import=0, fail_run=0)
        assert lastfailed == -1

        lastfailed = rlf(fail_import=1, fail_run=0)
        assert list(lastfailed) == ["test_maybe.py"]

        lastfailed = rlf(fail_import=0, fail_run=1)
        assert list(lastfailed) == ["test_maybe.py::test_hello"]

    # ensemble: same as above - the failure being recorded is an import error.
    def test_lastfailed_failure_subset(
        self, pytester: Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        pytester.makepyfile(
            test_maybe="""
            import os
            env = os.environ
            if '1' == env['FAILIMPORT']:
                raise ImportError('fail')
            def test_hello():
                assert '0' == env['FAILTEST']
        """
        )

        pytester.makepyfile(
            test_maybe2="""
            import os
            env = os.environ
            if '1' == env['FAILIMPORT']:
                raise ImportError('fail')

            def test_hello():
                assert '0' == env['FAILTEST']

            def test_pass():
                pass
        """
        )

        def rlf(
            fail_import: int, fail_run: int, args: Sequence[str] = ()
        ) -> tuple[Any, Any]:
            monkeypatch.setenv("FAILIMPORT", str(fail_import))
            monkeypatch.setenv("FAILTEST", str(fail_run))

            result = pytester.runpytest("-q", "--lf", *args)
            config = pytester.parseconfigure()
            assert config.cache is not None
            lastfailed = config.cache.get("cache/lastfailed", -1)
            return result, lastfailed

        result, lastfailed = rlf(fail_import=0, fail_run=0)
        assert lastfailed == -1
        result.stdout.fnmatch_lines(["*3 passed*"])

        result, lastfailed = rlf(fail_import=1, fail_run=0)
        assert sorted(list(lastfailed)) == ["test_maybe.py", "test_maybe2.py"]

        result, lastfailed = rlf(fail_import=0, fail_run=0, args=("test_maybe2.py",))
        assert list(lastfailed) == ["test_maybe.py"]

        # edge case of test selection - even if we remember failures
        # from other tests we still need to run all tests if no test
        # matches the failures
        result, lastfailed = rlf(fail_import=0, fail_run=0, args=("test_maybe2.py",))
        assert list(lastfailed) == ["test_maybe.py"]
        result.stdout.fnmatch_lines(["*2 passed*"])

    def test_lastfailed_creates_cache_when_needed(self, tmp_path: Path) -> None:
        # Issue #1342
        # The original's -q only affected rendering, and nothing here reads
        # the output; an ensemble without the terminal plugin has no -q.
        lastfailed = tmp_path / ".pytest_cache/v/cache/lastfailed"

        test_empty = in_memory("test_empty")
        run_tests(test_empty, spec=cache_spec(tmp_path, "--lf"))
        assert not lastfailed.exists()

        def test_success():
            assert True

        test_successful = in_memory("test_successful", test_success)
        run_tests(test_empty, test_successful, spec=cache_spec(tmp_path, "--lf"))
        assert not lastfailed.exists()

        def test_error():
            assert False

        test_errored = in_memory("test_errored", test_error)
        run_tests(
            test_empty,
            test_successful,
            test_errored,
            spec=cache_spec(tmp_path, "--lf"),
        )
        assert lastfailed.exists()

    def test_xfail_not_considered_failure(self, tmp_path: Path) -> None:
        @pytest.mark.xfail
        def test():
            assert 0

        record = run_tests(
            in_memory("test_xfail_not_considered_failure", test),
            spec=cache_spec(tmp_path),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*1 xfailed*"])
        assert self.get_cached_last_failed(tmp_path) == []

    def test_xfail_strict_considered_failure(self, tmp_path: Path) -> None:
        @pytest.mark.xfail(strict=True)
        def test():
            pass

        record = run_tests(
            in_memory("test_xfail_strict_considered_failure", test),
            spec=cache_spec(tmp_path),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*1 failed*"])
        assert self.get_cached_last_failed(tmp_path) == [
            "test_xfail_strict_considered_failure.py::test"
        ]

    @pytest.mark.parametrize("mark", ["mark.xfail", "mark.skip"])
    def test_failed_changed_to_xfail_or_skip(self, tmp_path: Path, mark: str) -> None:
        decorator, outcomes = {
            "mark.xfail": (pytest.mark.xfail, {"xfailed": 1}),
            "mark.skip": (pytest.mark.skip, {"skipped": 1}),
        }[mark]

        def test():
            assert 0

        record = run_tests(
            in_memory("test_failed_changed_to_xfail_or_skip", test),
            spec=cache_spec(tmp_path),
        )
        assert self.get_cached_last_failed(tmp_path) == [
            "test_failed_changed_to_xfail_or_skip.py::test"
        ]
        # ``result.ret == 1``: the run failed.
        record.assert_outcomes(failed=1)

        record = run_tests(
            in_memory("test_failed_changed_to_xfail_or_skip", decorator(test)),
            spec=cache_spec(tmp_path),
        )
        # ``result.ret == 0``: nothing failed any more.
        record.assert_outcomes(**outcomes)
        assert self.get_cached_last_failed(tmp_path) == []

    @pytest.mark.parametrize("quiet", [True, False])
    @pytest.mark.parametrize("opt", ["--ff", "--lf"])
    def test_lf_and_ff_prints_no_needless_message(
        self, quiet: bool, opt: str, tmp_path: Path
    ) -> None:
        # Issue 3853
        def test():
            assert 0

        module = in_memory("test_lf_and_ff", test)
        args = [opt]
        if quiet:
            args.append("-q")
        record = run_tests(
            module, spec=cache_spec(tmp_path, *args), capture_output=True
        )
        record.stdout.no_fnmatch_line("*run all*")

        record = run_tests(
            module, spec=cache_spec(tmp_path, *args), capture_output=True
        )
        if quiet:
            record.stdout.no_fnmatch_line("*run all*")
        else:
            assert "rerun previous" in record.output

    def get_cached_last_failed(self, rootpath: Path) -> list[str]:
        with configured(cache_spec(rootpath)) as config:
            assert config.cache is not None
            return sorted(config.cache.get("cache/lastfailed", {}))

    def test_cache_cumulative(self, tmp_path: Path) -> None:
        """Test workflow where user fixes errors gradually file by file using --lf."""
        # The sources are real files here: the workflow is "file by file", and
        # the file-level collection skipping that produces the "(skipped N
        # files)" messages only applies to paths that exist.
        # 1. initial run
        write_source(
            tmp_path,
            "test_bar.py",
            "def test_bar_1(): pass\ndef test_bar_2(): assert 0\n",
        )
        write_source(
            tmp_path,
            "test_foo.py",
            "def test_foo_3(): pass\ndef test_foo_4(): assert 0\n",
        )
        test_bar = tmp_path / "test_bar.py"
        test_foo = tmp_path / "test_foo.py"

        run_tests(
            module_from_path(test_bar),
            module_from_path(test_foo),
            spec=cache_spec(tmp_path),
        )
        assert self.get_cached_last_failed(tmp_path) == [
            "test_bar.py::test_bar_2",
            "test_foo.py::test_foo_4",
        ]

        # 2. fix test_bar_2, run only test_bar.py
        write_source(
            tmp_path, "test_bar.py", "def test_bar_1(): pass\ndef test_bar_2(): pass\n"
        )
        record = run_tests(
            module_from_path(test_bar), spec=cache_spec(tmp_path), capture_output=True
        )
        record.stdout.fnmatch_lines(["*2 passed*"])
        # ensure cache does not forget that test_foo_4 failed once before
        assert self.get_cached_last_failed(tmp_path) == ["test_foo.py::test_foo_4"]

        record = run_tests(
            module_from_path(test_bar),
            module_from_path(test_foo),
            spec=cache_spec(tmp_path, "--last-failed"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "run-last-failure: rerun previous 1 failure (skipped 1 file)",
                "*= 1 failed in *",
            ]
        )
        assert self.get_cached_last_failed(tmp_path) == ["test_foo.py::test_foo_4"]

        # 3. fix test_foo_4, run only test_foo.py
        write_source(
            tmp_path, "test_foo.py", "def test_foo_3(): pass\ndef test_foo_4(): pass\n"
        )
        record = run_tests(
            module_from_path(test_foo),
            spec=cache_spec(tmp_path, "--last-failed"),
            capture_output=True,
        )
        # The original passed test_foo.py as an argument, which made it an
        # initial path and so exempt from the file-level filtering; here the
        # known failure is filtered out during collection instead of being
        # deselected afterwards, so this collects 1 item rather than
        # collecting 2 and deselecting 1.
        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "run-last-failure: rerun previous 1 failure",
                "*= 1 passed in *",
            ]
        )
        assert self.get_cached_last_failed(tmp_path) == []

        record = run_tests(
            module_from_path(test_bar),
            module_from_path(test_foo),
            spec=cache_spec(tmp_path, "--last-failed"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*4 passed*"])
        assert self.get_cached_last_failed(tmp_path) == []

    def test_lastfailed_no_failures_behavior_all_passed(self, tmp_path: Path) -> None:
        def test_1():
            pass

        def test_2():
            pass

        module = in_memory("test_lastfailed_no_failures", test_1, test_2)

        record = run_tests(module, spec=cache_spec(tmp_path), capture_output=True)
        record.stdout.fnmatch_lines(["*2 passed*"])
        record = run_tests(
            module, spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(["*2 passed*"])
        record = run_tests(
            module,
            spec=cache_spec(tmp_path, "--lf", "--lfnf", "all"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*2 passed*"])

        # Ensure the list passed to pytest_deselected is a copy,
        # and not a reference which is cleared right after.
        class DeselectedPlugin:
            def __init__(self) -> None:
                self.deselected: list[object] = []

            def pytest_deselected(self, items):
                self.deselected = items

        plugin = DeselectedPlugin()
        record = run_tests(
            module,
            spec=cache_spec(tmp_path, "--lf", "--lfnf", "none").replace(
                extra_plugins=(plugin,)
            ),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "collected 2 items / 2 deselected / 0 selected",
                "run-last-failure: no previously failed tests, deselecting all items.",
                "* 2 deselected in *",
            ]
        )
        # The original printed this from a sessionfinish hook; asserting the
        # retained list directly is what that print was a proxy for.
        assert len(plugin.deselected) == 2
        # ``result.ret == ExitCode.NO_TESTS_COLLECTED``
        assert record.outcomes() == {}
        assert record.deselected == 2

    def test_lastfailed_no_failures_behavior_empty_cache(self, tmp_path: Path) -> None:
        def test_1():
            pass

        def test_2():
            assert 0

        module = in_memory("test_lastfailed_empty_cache", test_1, test_2)

        record = run_tests(
            module,
            spec=cache_spec(tmp_path, "--lf", "--cache-clear"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*1 failed*1 passed*"])
        record = run_tests(
            module,
            spec=cache_spec(tmp_path, "--lf", "--cache-clear", "--lfnf", "all"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*1 failed*1 passed*"])
        record = run_tests(
            module,
            spec=cache_spec(tmp_path, "--lf", "--cache-clear", "--lfnf", "none"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["*2 desel*"])

    def test_lastfailed_skip_collection(self, tmp_path: Path) -> None:
        """
        Test --lf behavior regarding skipping collection of files that are not marked as
        failed in the cache (#5172).
        """
        write_source(
            tmp_path,
            "pkg1/test_1.py",
            "import pytest\n\n"
            "@pytest.mark.parametrize('i', range(3))\n"
            "def test_1(i): pass\n",
        )
        write_source(
            tmp_path,
            "pkg2/test_2.py",
            "import pytest\n\n"
            "@pytest.mark.parametrize('i', range(5))\n"
            "def test_1(i):\n"
            "    assert i not in (1, 3)\n",
        )

        def sources() -> tuple[types.ModuleType, ...]:
            return tuple(
                module_from_path(path)
                for path in sorted(tmp_path.rglob("pkg*/test_*.py"))
            )

        # first run: collects 8 items (test_1: 3, test_2: 5)
        record = run_tests(*sources(), spec=cache_spec(tmp_path), capture_output=True)
        record.stdout.fnmatch_lines(["collected 8 items", "*2 failed*6 passed*"])
        # second run: collects only 5 items from test_2, because all tests from test_1 have passed
        record = run_tests(
            *sources(), spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "collected 2 items",
                "run-last-failure: rerun previous 2 failures (skipped 1 file)",
                "*= 2 failed in *",
            ]
        )

        # add another file and check if message is correct when skipping more than 1 file
        write_source(tmp_path, "pkg1/test_3.py", "def test_3(): pass\n")
        record = run_tests(
            *sources(), spec=cache_spec(tmp_path, "--lf"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "collected 2 items",
                "run-last-failure: rerun previous 2 failures (skipped 2 files)",
                "*= 2 failed in *",
            ]
        )

    # ensemble: the point is a file nested at a different *level* of the
    # collection tree; an ensemble's tree is flat - every module is a direct
    # child of the session - and packages have no representation in it.
    def test_lastfailed_skip_collection_with_nesting(self, pytester: Pytester) -> None:
        """Check that file skipping works even when the file with failures is
        nested at a different level of the collection tree."""
        pytester.makepyfile(
            **{
                "test_1.py": """
                    def test_1(): pass
                """,
                "pkg/__init__.py": "",
                "pkg/test_2.py": """
                    def test_2(): assert False
                """,
            }
        )
        # first run
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["collected 2 items", "*1 failed*1 passed*"])
        # second run - test_1.py is skipped.
        result = pytester.runpytest("--lf")
        result.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "run-last-failure: rerun previous 1 failure (skipped 1 file)",
                "*= 1 failed in *",
            ]
        )

    def test_lastfailed_with_known_failures_not_being_selected(
        self, tmp_path: Path
    ) -> None:
        write_source(tmp_path, "pkg1/test_1.py", """def test_1(): assert 0""")
        write_source(tmp_path, "pkg1/test_2.py", """def test_2(): pass""")
        test_1 = tmp_path / "pkg1/test_1.py"
        test_2 = tmp_path / "pkg1/test_2.py"

        record = run_tests(
            module_from_path(test_1),
            module_from_path(test_2),
            spec=cache_spec(tmp_path),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["collected 2 items", "* 1 failed, 1 passed in *"])

        test_1.unlink()
        record = run_tests(
            module_from_path(test_2),
            spec=cache_spec(tmp_path, "--lf"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "run-last-failure: 1 known failures not in selected tests",
                "* 1 passed in *",
            ]
        )

        # Recreate file with known failure.
        write_source(tmp_path, "pkg1/test_1.py", """def test_1(): assert 0""")
        record = run_tests(
            module_from_path(test_1),
            module_from_path(test_2),
            spec=cache_spec(tmp_path, "--lf"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "run-last-failure: rerun previous 1 failure (skipped 1 file)",
                "* 1 failed in *",
            ]
        )

        # Remove/rename test: collects the file again.
        write_source(tmp_path, "pkg1/test_1.py", """def test_renamed(): assert 0""")
        record = run_tests(
            module_from_path(test_1),
            module_from_path(test_2),
            spec=cache_spec(tmp_path, "--lf", "-rf"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "collected 2 items",
                "run-last-failure: 1 known failures not in selected tests",
                "pkg1/test_1.py F *",
                "pkg1/test_2.py . *",
                # Assertion rewriting is not applied to ensemble sources, so
                # the one-line reason is the bare exception.
                "FAILED pkg1/test_1.py::test_renamed - AssertionError",
                "* 1 failed, 1 passed in *",
            ]
        )

        record = run_tests(
            module_from_path(test_1),
            module_from_path(test_2),
            spec=cache_spec(tmp_path, "--lf", "--co"),
            capture_output=True,
        )
        # The tree has no <Dir> nodes: an ensemble collects modules directly
        # under the session.
        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "run-last-failure: rerun previous 1 failure (skipped 1 file)",
                "",
                "<EnsembleModule pkg1/test_1.py>",
                "  <Function test_renamed>",
            ]
        )

    def test_lastfailed_args_with_deselected(self, tmp_path: Path) -> None:
        """Test regression with --lf running into NoMatch error.

        This was caused by it not collecting (non-failed) nodes given as
        arguments.
        """

        def test_pass():
            pass

        def test_fail():
            assert 0

        module = in_memory("pkg1/test_1", test_pass, test_fail)

        record = run_tests(module, spec=cache_spec(tmp_path), capture_output=True)
        record.stdout.fnmatch_lines(["collected 2 items", "* 1 failed, 1 passed in *"])
        record.assert_outcomes(passed=1, failed=1)

        # Selecting single node ids on the command line has no ensemble
        # equivalent; -k selects the same items.
        record = run_tests(
            module,
            spec=cache_spec(tmp_path, "-k", "test_pass", "--lf", "--co"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "*collected 2 items / 1 deselected / 1 selected",
                "run-last-failure: 1 known failures not in selected tests",
                "",
                "<EnsembleModule pkg1/test_1.py>",
                "  <Function test_pass>",
            ],
            consecutive=True,
        )

        record = run_tests(
            module, spec=cache_spec(tmp_path, "--lf", "--co"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "collected 2 items / 1 deselected / 1 selected",
                "run-last-failure: rerun previous 1 failure",
                "",
                "<EnsembleModule pkg1/test_1.py>",
                "  <Function test_fail>",
                "",
                "*= 1/2 tests collected (1 deselected) in *",
            ],
        )

    def test_lastfailed_with_class_items(self, tmp_path: Path) -> None:
        """Test regression with --lf deselecting whole classes."""

        class TestFoo:
            def test_pass(self):
                pass

            def test_fail(self):
                assert 0

        def test_other():
            assert 0

        module = in_memory("pkg1/test_1", TestFoo, test_other)

        record = run_tests(module, spec=cache_spec(tmp_path), capture_output=True)
        record.stdout.fnmatch_lines(["collected 3 items", "* 2 failed, 1 passed in *"])
        record.assert_outcomes(passed=1, failed=2)

        record = run_tests(
            module, spec=cache_spec(tmp_path, "--lf", "--co"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "collected 3 items / 1 deselected / 2 selected",
                "run-last-failure: rerun previous 2 failures",
                "",
                "<EnsembleModule pkg1/test_1.py>",
                "  <Class TestFoo>",
                "    <Function test_fail>",
                "  <Function test_other>",
                "",
                "*= 2/3 tests collected (1 deselected) in *",
            ],
            consecutive=True,
        )

    def test_lastfailed_with_all_filtered(self, tmp_path: Path) -> None:
        def test_fail():
            assert 0

        def test_pass():
            pass

        record = run_tests(
            in_memory("pkg1/test_1", test_fail, test_pass),
            spec=cache_spec(tmp_path),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(["collected 2 items", "* 1 failed, 1 passed in *"])
        record.assert_outcomes(passed=1, failed=1)

        # Remove known failure.
        record = run_tests(
            in_memory("pkg1/test_1", test_pass),
            spec=cache_spec(tmp_path, "--lf", "--co"),
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "run-last-failure: 1 known failures not in selected tests",
                "",
                "<EnsembleModule pkg1/test_1.py>",
                "  <Function test_pass>",
                "",
                "*= 1 test collected in*",
            ],
            consecutive=True,
        )

    # ensemble: Package nodes are the whole subject, and an ensemble has none.
    def test_packages(self, pytester: Pytester) -> None:
        """Regression test for #7758.

        The particular issue here was that Package nodes were included in the
        filtering, being themselves Modules for the __init__.py, even if they
        had failed Modules in them.

        The tests includes a test in an __init__.py file just to make sure the
        fix doesn't somehow regress that, it is not critical for the issue.
        """
        pytester.makepyfile(
            **{
                "__init__.py": "",
                "a/__init__.py": "def test_a_init(): assert False",
                "a/test_one.py": "def test_1(): assert False",
                "b/__init__.py": "",
                "b/test_two.py": "def test_2(): assert False",
            },
        )
        pytester.makeini(
            """
            [pytest]
            python_files = *.py
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=3)
        result = pytester.runpytest("--lf")
        result.assert_outcomes(failed=3)

    def test_non_python_file_skipped(self, tmp_path: Path) -> None:
        # The yaml collector of the ``dummy_yaml_custom_test`` fixture, as a
        # collector handed to the ensemble directly: ``pytest_collect_file`` is
        # never called, because an ensemble is given its collectors rather than
        # walking the filesystem for them.
        class YamlItem(pytest.Item):
            def runtest(self) -> None:
                pass

        class YamlFile(pytest.File):
            def collect(self):
                yield YamlItem.from_parent(name=self.path.name, parent=self)

        tmp_path.joinpath("test1.yaml").write_text("", encoding="utf-8")
        write_source(tmp_path, "test_bad.py", """def test_bad(): assert False""")
        test_bad = tmp_path / "test_bad.py"

        def run(*args: str):
            with Ensemble(
                module_from_path(test_bad),
                spec=cache_spec(tmp_path, *args),
                capture_output=True,
            ) as ensemble:
                ensemble_collection(ensemble.session).collectors.append(
                    YamlFile.from_parent(
                        parent=ensemble.session, path=tmp_path / "test1.yaml"
                    )
                )
                record = ensemble.run()
            return ensemble.final_record(record)

        record = run()
        record.stdout.fnmatch_lines(["collected 2 items", "* 1 failed, 1 passed in *"])

        record = run("--lf")
        record.stdout.fnmatch_lines(
            [
                "collected 1 item",
                "run-last-failure: rerun previous 1 failure (skipped 1 file)",
                "* 1 failed in *",
            ]
        )


class TestNewFirst:
    def test_newfirst_usecase(self, tmp_path: Path) -> None:
        write_source(tmp_path, "test_1/test_1.py", "def test_1(): assert 1\n")
        write_source(tmp_path, "test_2/test_2.py", "def test_1(): assert 1\n")

        p1 = tmp_path.joinpath("test_1/test_1.py")
        p2 = tmp_path.joinpath("test_2/test_2.py")
        os.utime(p1, ns=(p1.stat().st_atime_ns, int(1e9)))

        def sources() -> tuple[types.ModuleType, ...]:
            # Distinct module names, so that both are importable side by side.
            return (
                module_from_path(p1, "test_1_test_1"),
                module_from_path(p2, "test_2_test_2"),
            )

        def ran(record) -> list[str]:
            return [r.nodeid for r in record.reports if r.when == "call"]

        record = run_tests(
            *sources(), spec=cache_spec(tmp_path, "-v"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            ["*test_1/test_1.py::test_1 PASSED*", "*test_2/test_2.py::test_1 PASSED*"]
        )
        assert ran(record) == ["test_1/test_1.py::test_1", "test_2/test_2.py::test_1"]

        record = run_tests(
            *sources(), spec=cache_spec(tmp_path, "-v", "--nf"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            ["*test_2/test_2.py::test_1 PASSED*", "*test_1/test_1.py::test_1 PASSED*"]
        )
        assert ran(record) == ["test_2/test_2.py::test_1", "test_1/test_1.py::test_1"]

        p1.write_text(
            "def test_1(): assert 1\ndef test_2(): assert 1\n", encoding="utf-8"
        )
        os.utime(p1, ns=(p1.stat().st_atime_ns, int(1e9)))

        items = collect_tests(*sources(), spec=cache_spec(tmp_path, "--nf", "--co"))
        assert [item.nodeid for item in items] == [
            "test_1/test_1.py::test_2",
            "test_2/test_2.py::test_1",
            "test_1/test_1.py::test_1",
        ]

        # Newest first with (plugin) pytest_collection_modifyitems hook.
        class MyPlugin:
            def __init__(self) -> None:
                self.new_items: list[str] = []

            def pytest_collection_modifyitems(self, items):
                items[:] = sorted(items, key=lambda item: item.nodeid)
                self.new_items = [x.nodeid for x in items]

        plugin = MyPlugin()
        items = collect_tests(
            *sources(),
            spec=cache_spec(tmp_path, "--nf", "--co").replace(extra_plugins=(plugin,)),
        )
        assert plugin.new_items == [
            "test_1/test_1.py::test_1",
            "test_1/test_1.py::test_2",
            "test_2/test_2.py::test_1",
        ]
        assert [item.nodeid for item in items] == [
            "test_1/test_1.py::test_2",
            "test_2/test_2.py::test_1",
            "test_1/test_1.py::test_1",
        ]

    def test_newfirst_parametrize(self, tmp_path: Path) -> None:
        write_source(
            tmp_path,
            "test_1/test_1.py",
            "import pytest\n"
            "@pytest.mark.parametrize('num', [1, 2])\n"
            "def test_1(num): assert num\n",
        )
        write_source(
            tmp_path,
            "test_2/test_2.py",
            "import pytest\n"
            "@pytest.mark.parametrize('num', [1, 2])\n"
            "def test_1(num): assert num\n",
        )

        p1 = tmp_path.joinpath("test_1/test_1.py")
        p2 = tmp_path.joinpath("test_2/test_2.py")
        os.utime(p1, ns=(p1.stat().st_atime_ns, int(1e9)))

        def sources(*paths: Path) -> tuple[types.ModuleType, ...]:
            return tuple(
                module_from_path(path, f"{path.parent.name}_{path.stem}")
                for path in (paths or (p1, p2))
            )

        record = run_tests(
            *sources(), spec=cache_spec(tmp_path, "-v"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "*test_1/test_1.py::test_1[1*",
                "*test_1/test_1.py::test_1[2*",
                "*test_2/test_2.py::test_1[1*",
                "*test_2/test_2.py::test_1[2*",
            ]
        )

        record = run_tests(
            *sources(), spec=cache_spec(tmp_path, "-v", "--nf"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "*test_2/test_2.py::test_1[1*",
                "*test_2/test_2.py::test_1[2*",
                "*test_1/test_1.py::test_1[1*",
                "*test_1/test_1.py::test_1[2*",
            ]
        )

        p1.write_text(
            "import pytest\n"
            "@pytest.mark.parametrize('num', [1, 2, 3])\n"
            "def test_1(num): assert num\n",
            encoding="utf-8",
        )
        os.utime(p1, ns=(p1.stat().st_atime_ns, int(1e9)))

        # Running only a subset does not forget about existing ones.
        record = run_tests(
            *sources(p2), spec=cache_spec(tmp_path, "-v", "--nf"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            ["*test_2/test_2.py::test_1[1*", "*test_2/test_2.py::test_1[2*"]
        )

        record = run_tests(
            *sources(), spec=cache_spec(tmp_path, "-v", "--nf"), capture_output=True
        )
        record.stdout.fnmatch_lines(
            [
                "*test_1/test_1.py::test_1[3*",
                "*test_2/test_2.py::test_1[1*",
                "*test_2/test_2.py::test_1[2*",
                "*test_1/test_1.py::test_1[1*",
                "*test_1/test_1.py::test_1[2*",
            ]
        )


class TestReadme:
    def check_readme(self, rootpath: Path) -> bool:
        with configured(cache_spec(rootpath)) as config:
            assert config.cache is not None
            readme = config.cache._cachedir.joinpath("README.md")
            return readme.is_file()

    def test_readme_passed(self, tmp_path: Path) -> None:
        def test_always_passes():
            pass

        run_tests(test_always_passes, spec=cache_spec(tmp_path))
        assert self.check_readme(tmp_path) is True

    def test_readme_failed(self, tmp_path: Path) -> None:
        def test_always_fails():
            assert 0

        run_tests(test_always_fails, spec=cache_spec(tmp_path))
        assert self.check_readme(tmp_path) is True


class Action(Enum):
    """Action to perform on the cache directory."""

    MKDIR = auto()
    SET = auto()


@pytest.mark.parametrize("action", list(Action))
def test_gitignore(
    tmp_path: Path,
    action: Action,
) -> None:
    """Ensure we automatically create .gitignore file in the pytest_cache directory (#3286)."""
    from _pytest.cacheprovider import Cache

    with configured(cache_spec(tmp_path)) as config:
        cache = Cache.for_config(config, _ispytest=True)
        if action == Action.MKDIR:
            cache.mkdir("foo")
        elif action == Action.SET:
            cache.set("foo", "bar")
        else:
            assert_never(action)
        msg = "# Created by pytest automatically.\n*\n"
        gitignore_path = cache._cachedir.joinpath(".gitignore")
        assert gitignore_path.read_text(encoding="UTF-8") == msg

        # Does not overwrite existing/custom one.
        gitignore_path.write_text("custom", encoding="utf-8")
        if action == Action.MKDIR:
            cache.mkdir("something")
        elif action == Action.SET:
            cache.set("something", "else")
        else:
            assert_never(action)
        assert gitignore_path.read_text(encoding="UTF-8") == "custom"


def test_preserve_keys_order(tmp_path: Path) -> None:
    """Ensure keys order is preserved when saving dicts (#9205)."""
    from _pytest.cacheprovider import Cache

    with configured(cache_spec(tmp_path)) as config:
        cache = Cache.for_config(config, _ispytest=True)
        cache.set("foo", {"z": 1, "b": 2, "a": 3, "d": 10})
        read_back = cache.get("foo", None)
        assert list(read_back.items()) == [("z", 1), ("b", 2), ("a", 3), ("d", 10)]


def test_does_not_create_boilerplate_in_existing_dirs(tmp_path: Path) -> None:
    from _pytest.cacheprovider import Cache

    with configured(cache_spec(tmp_path, cache_dir=".")) as config:
        cache = Cache.for_config(config, _ispytest=True)
        cache.set("foo", "bar")

    assert tmp_path.joinpath("v").is_dir()  # cache contents
    assert not tmp_path.joinpath(".gitignore").exists()
    assert not tmp_path.joinpath("README.md").exists()


def test_cachedir_tag(tmp_path: Path) -> None:
    """Ensure we automatically create CACHEDIR.TAG file in the pytest_cache directory (#4278)."""
    from _pytest.cacheprovider import Cache
    from _pytest.cacheprovider import CACHEDIR_FILES

    with configured(cache_spec(tmp_path)) as config:
        cache = Cache.for_config(config, _ispytest=True)
        cache.set("foo", "bar")
        cachedir_tag_path = cache._cachedir.joinpath("CACHEDIR.TAG")
        assert cachedir_tag_path.read_bytes() == CACHEDIR_FILES["CACHEDIR.TAG"]


# ensemble: --help is served from pytest_cmdline_main, which an ensemble
# never runs.
def test_clioption_with_cacheshow_and_help(pytester: Pytester) -> None:
    result = pytester.runpytest("--cache-show", "--help")
    assert result.ret == 0


def test_make_cachedir_cleans_up_on_base_exception(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Ensure _make_cachedir cleans up the temp directory on BaseException.

    When a BaseException (like KeyboardInterrupt) is raised during cache
    directory creation, the temporary directory should be cleaned up before
    re-raising the exception.
    """
    from _pytest.cacheprovider import _make_cachedir

    target = tmp_path / ".pytest_cache"

    def raise_keyboard_interrupt(self: Path, target: Path) -> None:
        raise KeyboardInterrupt("simulated interrupt")

    # Patch Path.rename only for the duration of the _make_cachedir call
    with monkeypatch.context() as m:
        m.setattr(Path, "rename", raise_keyboard_interrupt)

        # Verify the exception is re-raised
        with pytest.raises(KeyboardInterrupt, match="simulated interrupt"):
            _make_cachedir(target)

    # Verify no temp directories were left behind
    temp_dirs = list(tmp_path.glob("pytest-cache-files-*"))
    assert temp_dirs == [], f"Temp directories not cleaned up: {temp_dirs}"

    # Verify the target directory was not created
    assert not target.exists()

# mypy: allow-untyped-defs
from __future__ import annotations

from itertools import zip_longest
import os
from pathlib import Path
import sys
import textwrap

from _pytest.compat import getfuncargnames
from _pytest.config import ExitCode
from _pytest.ensemble import build_module
from _pytest.ensemble import collect_tests
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import Ensemble
from _pytest.ensemble import module_from_path
from _pytest.ensemble import run_tests
from _pytest.fixtures import deduplicate_names
from _pytest.fixtures import ParamValueKey
from _pytest.fixtures import TopRequest
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import get_public_names
from _pytest.pytester import LineMatcher
from _pytest.pytester import Pytester
from _pytest.python import Function
import pytest


def unregistered_mark(name: str, *args: object, **kwargs: object) -> MarkDecorator:
    """Build a mark decorator without consulting the host configuration.

    ``pytest.mark.<name>`` resolves against the *host* config at decoration
    time, and this suite runs with strict markers, so a deliberately
    unregistered mark applied to an ensemble source has to be built directly.
    """
    return MarkDecorator(Mark(name, args, kwargs, _ispytest=True), _ispytest=True)


#: The example scripts, run as themselves rather than copied somewhere first.
EXAMPLES = Path(__file__).parent.parent / "example_scripts"
FILL_FIXTURES = EXAMPLES / "fixtures/fill_fixtures"


def test_getfuncargnames_functions():
    """Test getfuncargnames for normal functions"""

    def f():
        raise NotImplementedError()

    assert not getfuncargnames(f)

    def g(arg):
        raise NotImplementedError()

    assert getfuncargnames(g) == ("arg",)

    def h(arg1, arg2="hello"):
        raise NotImplementedError()

    assert getfuncargnames(h) == ("arg1",)

    def j(arg1, arg2, arg3="hello"):
        raise NotImplementedError()

    assert getfuncargnames(j) == ("arg1", "arg2")


def test_getfuncargnames_methods():
    """Test getfuncargnames for normal methods"""

    class A:
        def f(self, arg1, arg2="hello"):
            raise NotImplementedError()

        def g(self, /, arg1, arg2="hello"):
            raise NotImplementedError()

        def h(self, *, arg1, arg2="hello"):
            raise NotImplementedError()

        def j(self, arg1, *, arg2, arg3="hello"):
            raise NotImplementedError()

        def k(self, /, arg1, *, arg2, arg3="hello"):
            raise NotImplementedError()

    assert getfuncargnames(A().f) == ("arg1",)
    assert getfuncargnames(A().g) == ("arg1",)
    assert getfuncargnames(A().h) == ("arg1",)
    assert getfuncargnames(A().j) == ("arg1", "arg2")
    assert getfuncargnames(A().k) == ("arg1", "arg2")


def test_getfuncargnames_staticmethod():
    """Test getfuncargnames for staticmethods"""

    class A:
        @staticmethod
        def static(arg1, arg2, x=1):
            raise NotImplementedError()

    assert getfuncargnames(A.static, cls=A) == ("arg1", "arg2")


def test_getfuncargnames_staticmethod_inherited() -> None:
    """Test getfuncargnames for inherited staticmethods (#8061)"""

    class A:
        @staticmethod
        def static(arg1, arg2, x=1):
            raise NotImplementedError()

    class B(A):
        pass

    assert getfuncargnames(B.static, cls=B) == ("arg1", "arg2")


@pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="""\
In python 3.13, this will raise FutureWarning:
functools.partial will be a method descriptor in future Python versions;
wrap it in staticmethod() if you want to preserve the old behavior

But the wrapped 'functools.partial' is tested by 'test_getfuncargnames_staticmethod_partial' below.
""",
)
def test_getfuncargnames_partial():
    """Check getfuncargnames for methods defined with functools.partial (#5701)"""
    import functools

    def check(arg1, arg2, i):
        raise NotImplementedError()

    class T:
        test_ok = functools.partial(check, i=2)

    values = getfuncargnames(T().test_ok, name="test_ok")
    assert values == ("arg1", "arg2")


def test_getfuncargnames_staticmethod_partial():
    """Check getfuncargnames for staticmethods defined with functools.partial (#5701)"""
    import functools

    def check(arg1, arg2, i):
        raise NotImplementedError()

    class T:
        test_ok = staticmethod(functools.partial(check, i=2))

    values = getfuncargnames(T().test_ok, name="test_ok")
    assert values == ("arg1", "arg2")


@pytest.mark.pytester_example_path("fixtures/fill_fixtures")
class TestFillFixtures:
    def test_funcarg_lookupfails(self) -> None:
        example = FILL_FIXTURES / "test_funcarg_lookupfails.py"
        record = run_tests(
            module_from_path(example), rootpath=example.parent, capture_output=True
        )
        # A fixture missing at setup is an error, not a failure.
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            [
                # The example is reported at its own path and line, not at
                # this file's - that is what running it as itself buys.
                "file *test_funcarg_lookupfails.py, line 12",
                "*def test_func(some)*",
                "*fixture*some*not found*",
                "*xyzsomething*",
            ]
        )

    def test_detect_recursive_dependency_error(self) -> None:
        example = FILL_FIXTURES / "test_detect_recursive_dependency_error.py"
        record = run_tests(
            module_from_path(example), rootpath=example.parent, capture_output=True
        )
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            ["*recursive dependency involving fixture 'fix1' detected*"]
        )

    def test_funcarg_basic(self) -> None:
        example = FILL_FIXTURES / "test_funcarg_basic.py"
        with Ensemble(module_from_path(example), rootpath=example.parent) as ensemble:
            (item,) = ensemble.collect()
            assert isinstance(item, Function)
            # The example is collected where it lives, so it keeps its identity.
            assert item.nodeid == "test_funcarg_basic.py::test_func"
            assert item.path == example
            # Execute's item's setup, which fills fixtures.
            item.session._setupstate.setup(item)
            del item.funcargs["request"]
            assert len(get_public_names(item.funcargs)) == 2
            assert item.funcargs["some"] == "test_func"
            assert item.funcargs["other"] == 42

    def test_funcarg_lookup_modulelevel(self) -> None:
        example = FILL_FIXTURES / "test_funcarg_lookup_modulelevel.py"
        record = run_tests(module_from_path(example), rootpath=example.parent)
        record.assert_outcomes(passed=2)
        assert sorted(record.by_test) == [
            "test_funcarg_lookup_modulelevel.py::TestClass::test_method",
            "test_funcarg_lookup_modulelevel.py::test_func",
        ]

    def test_funcarg_lookup_classlevel(self) -> None:
        example = FILL_FIXTURES / "test_funcarg_lookup_classlevel.py"
        record = run_tests(module_from_path(example), rootpath=example.parent)
        record.assert_outcomes(passed=1)

    # ensemble: conftest visibility is per-directory, and ensembles have no
    # directory tree below the rootdir to scope conftests to. Running the
    # example modules in place does not help: the conftests beside them are
    # exactly what is under test, and ensembles never load conftest files.
    def test_conftest_funcargs_only_available_in_subdir(
        self, pytester: Pytester
    ) -> None:
        pytester.copy_example()
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)

    def test_extend_fixture_module_class(self) -> None:
        example = FILL_FIXTURES / "test_extend_fixture_module_class.py"
        record = run_tests(module_from_path(example), rootpath=example.parent)
        record.assert_outcomes(passed=1)
        assert record["test_extend_fixture_module_class.py::TestSpam::test_spam"].passed

    def test_extend_fixture_conftest_module(self) -> None:
        # The example's conftest sits at its root, which is exactly what an
        # ensemble plugin object stands for - so the conftest and the test
        # module can both be run as themselves. The second run of the
        # original (passing the test file directly) only covered conftest
        # collection for an explicit file argument, which an ensemble has no
        # equivalent of.
        example_dir = FILL_FIXTURES / "test_extend_fixture_conftest_module"
        conftest = module_from_path(example_dir / "conftest.py")
        example = example_dir / "test_extend_fixture_conftest_module.py"
        spec = ConfigSpec(rootpath=example_dir, extra_plugins=(conftest,))
        record = run_tests(module_from_path(example), spec=spec)
        record.assert_outcomes(passed=1)
        assert record["test_extend_fixture_conftest_module.py::test_spam"].passed

    # ensemble: two conftests at different directory levels; running the
    # example module in place would not load either of them.
    def test_extend_fixture_conftest_conftest(self, pytester: Pytester) -> None:
        p = pytester.copy_example()
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = pytester.runpytest(str(next(Path(str(p)).rglob("test_*.py"))))
        result.stdout.fnmatch_lines(["*1 passed*"])

    # ensemble: needs a real importable plugin module named in `pytest_plugins`.
    def test_extend_fixture_conftest_plugin(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            testplugin="""
            import pytest

            @pytest.fixture
            def foo():
                return 7
        """
        )
        pytester.syspathinsert()
        pytester.makeconftest(
            """
            import pytest

            pytest_plugins = 'testplugin'

            @pytest.fixture
            def foo(foo):
                return foo + 7
        """
        )
        pytester.makepyfile(
            """
            def test_foo(foo):
                assert foo == 14
        """
        )
        result = pytester.runpytest("-s")
        assert result.ret == 0

    def test_extend_fixture_plugin_plugin(self, tmp_path: Path) -> None:
        # Two plugins should extend each order in loading order
        class TestPlugin0:
            @pytest.fixture
            def foo(self):
                return 7

        class TestPlugin1:
            @pytest.fixture
            def foo(self, foo):
                return foo + 7

        def test_foo(foo):
            assert foo == 14

        spec = ConfigSpec(
            rootpath=tmp_path, extra_plugins=(TestPlugin0(), TestPlugin1())
        )
        run_tests(test_foo, spec=spec).assert_outcomes(passed=1)

    def test_override_parametrized_fixture_conftest_module(
        self, tmp_path: Path
    ) -> None:
        """Test override of the parametrized fixture with non-parametrized one on the test module level."""

        class ConftestPlugin:
            @pytest.fixture(params=[1, 2, 3])
            def spam(self, request):
                return request.param

        @pytest.fixture
        def spam():
            return "spam"

        def test_spam(spam):
            assert spam == "spam"

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_override", spam=spam, test_spam=test_spam), spec=spec
        )
        record.assert_outcomes(passed=1)

    # ensemble: the override lives in a subdirectory conftest.
    def test_override_parametrized_fixture_conftest_conftest(
        self, pytester: Pytester
    ) -> None:
        """Test override of the parametrized fixture with non-parametrized one on the conftest level."""
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[1, 2, 3])
            def spam(request):
                return request.param
        """
        )
        subdir = pytester.mkpydir("subdir")
        subdir.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture
                def spam():
                    return 'spam'
                """
            ),
            encoding="utf-8",
        )
        testfile = subdir.joinpath("test_spam.py")
        testfile.write_text(
            textwrap.dedent(
                """\
                def test_spam(spam):
                    assert spam == "spam"
                """
            ),
            encoding="utf-8",
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*1 passed*"])
        result = pytester.runpytest(testfile)
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_override_non_parametrized_fixture_conftest_module(
        self, tmp_path: Path
    ) -> None:
        """Test override of the non-parametrized fixture with parametrized one on the test module level."""

        class ConftestPlugin:
            @pytest.fixture
            def spam(self):
                return "spam"

        @pytest.fixture(params=[1, 2, 3])
        def spam(request):
            return request.param

        # The module-level ``params`` dict of the original becomes a closure:
        # a source function keeps the *host* module's globals.
        params = {"spam": 1}

        def test_spam(spam):
            assert spam == params["spam"]
            params["spam"] += 1

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_override", spam=spam, test_spam=test_spam), spec=spec
        )
        record.assert_outcomes(passed=3)

    # ensemble: the override lives in a subdirectory conftest.
    def test_override_non_parametrized_fixture_conftest_conftest(
        self, pytester: Pytester
    ) -> None:
        """Test override of the non-parametrized fixture with parametrized one on the conftest level."""
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture
            def spam():
                return 'spam'
        """
        )
        subdir = pytester.mkpydir("subdir")
        subdir.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[1, 2, 3])
                def spam(request):
                    return request.param
                """
            ),
            encoding="utf-8",
        )
        testfile = subdir.joinpath("test_spam.py")
        testfile.write_text(
            textwrap.dedent(
                """\
                params = {'spam': 1}

                def test_spam(spam):
                    assert spam == params['spam']
                    params['spam'] += 1
                """
            ),
            encoding="utf-8",
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*3 passed*"])
        result = pytester.runpytest(testfile)
        result.stdout.fnmatch_lines(["*3 passed*"])

    # ensemble: the override lives in a subdirectory conftest.
    def test_override_autouse_fixture_with_parametrized_fixture_conftest_conftest(
        self, pytester: Pytester
    ) -> None:
        """Test override of the autouse fixture with parametrized one on the conftest level.
        This test covers the issue explained in issue 1601
        """
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(autouse=True)
            def spam():
                return 'spam'
        """
        )
        subdir = pytester.mkpydir("subdir")
        subdir.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[1, 2, 3])
                def spam(request):
                    return request.param
                """
            ),
            encoding="utf-8",
        )
        testfile = subdir.joinpath("test_spam.py")
        testfile.write_text(
            textwrap.dedent(
                """\
                params = {'spam': 1}

                def test_spam(spam):
                    assert spam == params['spam']
                    params['spam'] += 1
                """
            ),
            encoding="utf-8",
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*3 passed*"])
        result = pytester.runpytest(testfile)
        result.stdout.fnmatch_lines(["*3 passed*"])

    def test_override_fixture_reusing_super_fixture_parametrization(
        self, tmp_path: Path
    ) -> None:
        """Override a fixture at a lower level, reusing the higher-level fixture that
        is parametrized (#1953).
        """

        class ConftestPlugin:
            @pytest.fixture(params=[1, 2])
            def foo(self, request):
                return request.param

        @pytest.fixture
        def foo(foo):
            return foo * 2

        def test_spam(foo):
            assert foo in (2, 4)

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_override", foo=foo, test_spam=test_spam), spec=spec
        )
        record.assert_outcomes(passed=2)

    def test_override_parametrize_fixture_and_indirect(self, tmp_path: Path) -> None:
        """Override a fixture at a lower level, reusing the higher-level fixture that
        is parametrized, while also using indirect parametrization.
        """

        class ConftestPlugin:
            @pytest.fixture(params=[1, 2])
            def foo(self, request):
                return request.param

        @pytest.fixture
        def foo(foo):
            return foo * 2

        @pytest.fixture
        def bar(request):
            return request.param * 100

        @pytest.mark.parametrize("bar", [42], indirect=True)
        def test_spam(bar, foo):
            assert bar == 4200
            assert foo in (2, 4)

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_override", foo=foo, bar=bar, test_spam=test_spam),
            spec=spec,
        )
        record.assert_outcomes(passed=2)

    def test_override_top_level_fixture_reusing_super_fixture_parametrization(
        self, tmp_path: Path
    ) -> None:
        """Same as the above test, but with another level of overwriting."""

        class ConftestPlugin:
            @pytest.fixture(params=["unused", "unused"])
            def foo(self, request):
                return request.param

        @pytest.fixture(params=[1, 2])
        def foo(request):
            return request.param

        class Test:
            @pytest.fixture
            def foo(self, foo):
                return foo * 2

            def test_spam(self, foo):
                assert foo in (2, 4)

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(build_module("test_override", foo=foo, Test=Test), spec=spec)
        record.assert_outcomes(passed=2)

    def test_override_parametrized_fixture_with_new_parametrized_fixture(
        self, tmp_path: Path
    ) -> None:
        """Overriding a parametrized fixture, while also parametrizing the new fixture and
        simultaneously requesting the overwritten fixture as parameter, yields the same value
        as ``request.param``.
        """

        class ConftestPlugin:
            @pytest.fixture(params=["ignored", "ignored"])
            def foo(self, request):
                return request.param

        @pytest.fixture(params=[10, 20])
        def foo(foo, request):
            assert request.param == foo
            return foo * 2

        def test_spam(foo):
            assert foo in (20, 40)

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_override", foo=foo, test_spam=test_spam), spec=spec
        )
        record.assert_outcomes(passed=2)

    @pytest.mark.xfail(reason="not handled currently")
    def test_override_parametrized_fixture_via_transitive_fixture(
        self, tmp_path: Path
    ) -> None:
        """Test that overriding a parametrized fixture works even the super
        fixture is requested only transitively.

        Regression test for #7737.
        """

        @pytest.fixture(params=[1, 2])
        def foo(request):
            return request.param

        @pytest.fixture
        def bar(foo):
            return foo

        class TestIt:
            @pytest.fixture
            def foo(self, bar):
                return bar * 2

            def test_it(self, foo):
                pass

        record = run_tests(foo, bar, TestIt, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_autouse_fixture_plugin(self, tmp_path: Path) -> None:
        # A fixture from a plugin has no baseid set, which screwed up
        # the autouse fixture handling.
        class TestPlugin:
            @pytest.fixture(autouse=True)
            def foo(self, request):
                request.function.foo = 7

        def test_foo(request):
            assert request.function.foo == 7

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(TestPlugin(),))
        run_tests(test_foo, spec=spec).assert_outcomes(passed=1)

    def test_funcarg_lookup_error(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            @pytest.fixture
            def a_fixture(self): ...

            @pytest.fixture
            def b_fixture(self): ...

            @pytest.fixture
            def c_fixture(self): ...

            @pytest.fixture
            def d_fixture(self): ...

        def test_lookup_error(unknown):
            pass

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(test_lookup_error, spec=spec, capture_output=True)
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            [
                "*ERROR at setup of test_lookup_error*",
                # indentation is host-anchored: the source lives in this file
                "*def test_lookup_error(unknown):*",
                "E       fixture 'unknown' not found",
                ">       available fixtures:*a_fixture,*b_fixture,*c_fixture,*d_fixture*monkeypatch,*",
                # sorted
                ">       use 'py*test --fixtures *' for help on them.",
                "*1 error*",
            ]
        )
        record.stdout.no_fnmatch_line("*INTERNAL*")

    def test_fixture_excinfo_leak(self, tmp_path: Path) -> None:
        # on python2 sys.excinfo would leak into fixture executions
        import traceback

        @pytest.fixture
        def leak():
            if sys.exc_info()[0]:  # python3 bug :)
                traceback.print_exc()
            # fails
            assert sys.exc_info() == (None, None, None)

        def test_leak(leak):
            if sys.exc_info()[0]:  # python3 bug :)
                traceback.print_exc()
            assert sys.exc_info() == (None, None, None)

        run_tests(leak, test_leak, rootpath=tmp_path).assert_outcomes(passed=1)


class TestRequestBasic:
    def test_request_attributes(self, tmp_path: Path) -> None:
        @pytest.fixture
        def something(request): ...

        def test_func(something): ...

        with Ensemble(something, test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            assert isinstance(item, Function)
            req = TopRequest(item, _ispytest=True)
            assert req.function == item.obj
            assert req.keywords == item.keywords
            assert hasattr(req.module, "test_func")
            assert req.cls is None
            assert req.function.__name__ == "test_func"
            assert req.config == item.config
            assert repr(req).find(req.function.__name__) != -1

    def test_request_attributes_method(self, tmp_path: Path) -> None:
        class TestB:
            @pytest.fixture
            def something(self, request):
                return 1

            def test_func(self, something):
                pass

        with Ensemble(TestB, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            assert isinstance(item, Function)
            req = item._request
            assert req.cls.__name__ == "TestB"
            assert req.instance.__class__ == req.cls

    def test_request_contains_funcarg_arg2fixturedefs(self, tmp_path: Path) -> None:
        @pytest.fixture
        def something(request):
            pass

        class TestClass:
            def test_method(self, something):
                pass

        with Ensemble(something, TestClass, rootpath=tmp_path) as ensemble:
            (item1,) = ensemble.collect()
            assert isinstance(item1, Function)
            assert item1.name == "test_method"
            arg2fixturedefs = TopRequest(item1, _ispytest=True)._arg2fixturedefs
            assert len(arg2fixturedefs) == 1
            assert arg2fixturedefs["something"][0].argname == "something"

    # ensemble: needs a subprocess run (gc debug state is process-global).
    @pytest.mark.skipif(
        hasattr(sys, "pypy_version_info"),
        reason="this method of test doesn't work on pypy",
    )
    def test_request_garbage(self, pytester: Pytester) -> None:
        try:
            import xdist  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.xfail("this test is flaky when executed with xdist")
        pytester.makepyfile(
            """
            import sys
            import pytest
            from _pytest.fixtures import RequestFixtureDef
            import gc

            @pytest.fixture(autouse=True)
            def something(request):
                original = gc.get_debug()
                gc.set_debug(gc.DEBUG_SAVEALL)
                gc.collect()

                yield

                try:
                    gc.collect()
                    leaked = [x for _ in gc.garbage if isinstance(_, RequestFixtureDef)]
                    assert leaked == []
                finally:
                    gc.set_debug(original)

            def test_func():
                pass
        """
        )
        result = pytester.runpytest_subprocess()
        result.stdout.fnmatch_lines(["* 1 passed in *"])

    def test_getfixturevalue_recursive(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            @pytest.fixture
            def something(self, request):
                return 1

        @pytest.fixture
        def something(request):
            return request.getfixturevalue("something") + 1

        def test_func(something):
            assert something == 2

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_recursive", something=something, test_func=test_func),
            spec=spec,
        )
        record.assert_outcomes(passed=1)

    def test_getfixturevalue_teardown(self, tmp_path: Path) -> None:
        """
        Issue #1895

        `test_inner` requests `inner` fixture, which in turn requests `resource`
        using `getfixturevalue`. `test_func` then requests `resource`.

        `resource` is teardown before `inner` because the fixture mechanism won't consider
        `inner` dependent on `resource` when it is used via `getfixturevalue`: `test_func`
        will then cause the `resource`'s finalizer to be called first because of this.
        """

        @pytest.fixture(scope="session")
        def resource():
            r = ["value"]
            yield r
            r.pop()

        @pytest.fixture(scope="session")
        def inner(request):
            resource = request.getfixturevalue("resource")
            assert resource == ["value"]
            yield
            assert resource == ["value"]

        def test_inner(inner):
            pass

        def test_func(resource):
            pass

        record = run_tests(resource, inner, test_inner, test_func, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_getfixturevalue_teardown_previously_requested_does_not_warn(
        self, tmp_path: Path
    ) -> None:
        """Test that requesting a fixture during teardown that was previously
        requested is OK (#12882).

        Note: this is still kinda dubious so don't let this test lock you in to
        allowing this behavior forever...
        """

        @pytest.fixture
        def fix(request, tmp_path):
            yield
            assert request.getfixturevalue("tmp_path") == tmp_path

        def test_it(fix):
            pass

        # -Werror of the original: any warning would fail the run.
        spec = ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["error"]})
        record = run_tests(fix, test_it, spec=spec)
        record.assert_outcomes(passed=1, warnings=0)

    def test_getfixturevalue_teardown_new_fixture_deprecated(
        self, tmp_path: Path
    ) -> None:
        """Test that requesting a fixture during teardown that was not
        previously requested raises a deprecation warning (#12882).

        Note: this is a case that previously worked but will become a hard
        error after the deprecation is completed.
        """

        @pytest.fixture(scope="session")
        def resource():
            return "value"

        @pytest.fixture
        def fix(request):
            yield
            with pytest.warns(
                pytest.PytestRemovedIn10Warning,
                match=r'Calling request\.getfixturevalue\("resource"\) during teardown is deprecated',
            ):
                assert request.getfixturevalue("resource") == "value"

        def test_it(fix):
            pass

        record = run_tests(resource, fix, test_it, rootpath=tmp_path)
        record.assert_outcomes(passed=1)

    def test_getfixturevalue_teardown_new_inactive_fixture_errors(
        self, tmp_path: Path
    ) -> None:
        """Test that requesting a fixture during teardown that was not
        previously requested raises an error (#12882)."""

        @pytest.fixture
        def fix(request):
            yield
            request.getfixturevalue("tmp_path")

        def test_it(fix):
            pass

        record = run_tests(fix, test_it, rootpath=tmp_path)
        # The call phase passes; the teardown error is a separate report.
        record.assert_outcomes(passed=1, errors=1)
        teardown = record["test_it"].teardown
        assert teardown is not None
        assert (
            'The fixture value for "tmp_path" is not available during '
            "teardown because it was not previously requested." in teardown.longreprtext
        )

    def test_getfixturevalue_teardown_new_inactive_fixture_errors_top_request(
        self, tmp_path: Path
    ) -> None:
        """Test that requesting a fixture during teardown that was not
        previously requested raises an error (tricky case) (#12882)."""

        def test_it(request):
            request.addfinalizer(lambda: request.getfixturevalue("tmp_path"))

        record = run_tests(test_it, rootpath=tmp_path)
        record.assert_outcomes(passed=1, errors=1)
        teardown = record["test_it"].teardown
        assert teardown is not None
        assert (
            'The fixture value for "tmp_path" is not available during '
            "teardown because it was not previously requested." in teardown.longreprtext
        )

    def test_getfixturevalue(self, tmp_path: Path) -> None:
        @pytest.fixture
        def something(request):
            return 1

        # A module-level ``values`` list of the original: a source function
        # would read the *host* module's globals, so it becomes a closure.
        values = [2]

        @pytest.fixture
        def other(request):
            return values.pop()

        def test_func(something): ...

        with Ensemble(something, other, test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            assert isinstance(item, Function)
            req = item._request

            # Execute item's setup.
            item.session._setupstate.setup(item)

            with pytest.raises(pytest.FixtureLookupError):
                req.getfixturevalue("notexists")
            val = req.getfixturevalue("something")
            assert val == 1
            val = req.getfixturevalue("something")
            assert val == 1
            val2 = req.getfixturevalue("other")
            assert val2 == 2
            val2 = req.getfixturevalue("other")  # see about caching
            assert val2 == 2
            assert item.funcargs["something"] == 1
            assert len(get_public_names(item.funcargs)) == 2
            assert "request" in item.funcargs

    def test_request_addfinalizer(self, tmp_path: Path) -> None:
        teardownlist: list[int] = []

        @pytest.fixture
        def something(request):
            request.addfinalizer(lambda: teardownlist.append(1))

        def test_func(something): ...

        module = build_module(
            "test_addfinalizer",
            something=something,
            test_func=test_func,
            teardownlist=teardownlist,
        )
        with Ensemble(module, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            assert isinstance(item, Function)
            item.session._setupstate.setup(item)
            item._request._fillfixtures()
            # successively check finalization calls
            parent = item.getparent(pytest.Module)
            assert parent is not None
            assert parent.obj.teardownlist is teardownlist
            ss = item.session._setupstate
            assert not teardownlist
            ss.teardown_exact(None)
            print(ss.stack)
            assert teardownlist == [1]

    def test_request_addfinalizer_failing_setup(self, tmp_path: Path) -> None:
        values = [1]

        @pytest.fixture
        def myfix(request):
            request.addfinalizer(values.pop)
            assert 0

        def test_fix(myfix):
            pass

        def test_finalizer_ran():
            assert not values

        record = run_tests(myfix, test_fix, test_finalizer_ran, rootpath=tmp_path)
        # A failing *setup* is an error in terminal categories, where the
        # original's assertoutcome(failed=1) counted the failed report.
        record.assert_outcomes(errors=1, passed=1)

    def test_request_addfinalizer_failing_setup_module(self, tmp_path: Path) -> None:
        values = [1, 2]

        @pytest.fixture(scope="module")
        def myfix(request):
            request.addfinalizer(values.pop)
            request.addfinalizer(values.pop)
            assert 0

        def test_fix(myfix):
            pass

        run_tests(myfix, test_fix, rootpath=tmp_path)
        assert not values

    def test_request_addfinalizer_partial_setup_failure(self, tmp_path: Path) -> None:
        values: list[None] = []

        @pytest.fixture
        def something(request):
            request.addfinalizer(lambda: values.append(None))

        def test_func(something, missingarg):
            pass

        def test_second():
            assert len(values) == 1

        record = run_tests(something, test_func, test_second, rootpath=tmp_path)
        record.assert_outcomes(errors=1, passed=1)
        assert record["test_func"].outcome == "error"
        assert record["test_second"].passed

    def test_request_subrequest_addfinalizer_exceptions(self, tmp_path: Path) -> None:
        """
        Ensure exceptions raised during teardown by finalizers are suppressed
        until all finalizers are called, then re-raised together in an
        exception group (#2440)
        """
        values: list[int] = []

        def _excepts(where):
            raise Exception(f"Error in {where} fixture")

        @pytest.fixture
        def subrequest(request):
            return request

        @pytest.fixture
        def something(subrequest):
            subrequest.addfinalizer(lambda: values.append(1))
            subrequest.addfinalizer(lambda: values.append(2))
            subrequest.addfinalizer(lambda: _excepts("something"))

        @pytest.fixture
        def excepts(subrequest):
            subrequest.addfinalizer(lambda: _excepts("excepts"))
            subrequest.addfinalizer(lambda: values.append(3))

        def test_first(something, excepts):
            pass

        def test_second():
            assert values == [3, 2, 1]

        record = run_tests(
            subrequest,
            something,
            excepts,
            test_first,
            test_second,
            rootpath=tmp_path,
            capture_output=True,
        )
        record.assert_outcomes(passed=2, errors=1)
        record.stdout.fnmatch_lines(
            [
                '  | *ExceptionGroup: errors while tearing down fixture "subrequest" of <Function test_first> (2 sub-exceptions)',  # noqa: E501
                "  +-+---------------- 1 ----------------",
                "    | Exception: Error in something fixture",
                "    +---------------- 2 ----------------",
                "    | Exception: Error in excepts fixture",
                "    +------------------------------------",
            ],
        )

    def test_request_getmodulepath(self, tmp_path: Path) -> None:
        def test_somefunc(): ...

        with Ensemble(test_somefunc, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            assert isinstance(item, Function)
            modcol = item.getparent(pytest.Module)
            assert modcol is not None
            req = TopRequest(item, _ispytest=True)
            assert req.path == modcol.path

    def test_request_fixturenames(self, tmp_path: Path) -> None:
        @pytest.fixture
        def arg1():
            pass

        @pytest.fixture
        def farg(arg1):
            pass

        @pytest.fixture(autouse=True)
        def sarg(tmp_path):
            pass

        def test_function(request, farg):
            assert set(get_public_names(request.fixturenames)) == {
                "sarg",
                "arg1",
                "request",
                "farg",
                "tmp_path",
                "tmp_path_factory",
            }

        record = run_tests(arg1, farg, sarg, test_function, rootpath=tmp_path)
        record.assert_outcomes(passed=1)

    def test_request_fixturenames_dynamic_fixture(self) -> None:
        """Regression test for #3057"""
        example = EXAMPLES / "fixtures/test_getfixturevalue_dynamic.py"
        record = run_tests(module_from_path(example), rootpath=example.parent)
        record.assert_outcomes(passed=1)
        assert record["test_getfixturevalue_dynamic.py::test"].passed

    def test_setupdecorator_and_xunit(self, tmp_path: Path) -> None:
        values: list[str] = []

        @pytest.fixture(scope="module", autouse=True)
        def setup_module():
            values.append("module")

        @pytest.fixture(autouse=True)
        def setup_function():
            values.append("function")

        def test_func():
            pass

        class TestClass:
            @pytest.fixture(scope="class", autouse=True)
            @classmethod
            def setup_class(cls):
                values.append("class")

            @pytest.fixture(autouse=True)
            def setup_method(self):
                values.append("method")

            def test_method(self):
                pass

        record = run_tests(
            setup_module, setup_function, test_func, TestClass, rootpath=tmp_path
        )
        record.assert_outcomes(passed=2)
        assert values == ["module", "function", "class", "function", "method"]

    # ensemble: --fixtures runs through pytest_cmdline_main, which an
    # ensemble never reaches, and the subject is a subdirectory conftest.
    def test_fixtures_sub_subdir_normalize_sep(self, pytester: Pytester) -> None:
        # this tests that normalization of nodeids takes place
        b = pytester.path.joinpath("tests", "unit")
        b.mkdir(parents=True)
        b.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def arg1():
                    pass
                """
            ),
            encoding="utf-8",
        )
        p = b.joinpath("test_module.py")
        p.write_text("def test_func(arg1): pass", encoding="utf-8")
        result = pytester.runpytest(p, "--fixtures")
        assert result.ret == 0
        result.stdout.fnmatch_lines(
            """
            *fixtures defined*conftest*
            *arg1*
        """
        )

    # ensemble: --fixtures runs through pytest_cmdline_main, unreachable.
    def test_show_fixtures_color_yes(self, pytester: Pytester) -> None:
        pytester.makepyfile("def test_this(): assert 1")
        result = pytester.runpytest("--color=yes", "--fixtures")
        assert "\x1b[32mtmp_path" in result.stdout.str()

    def test_newstyle_with_request(self, tmp_path: Path) -> None:
        @pytest.fixture
        def arg(request):
            pass

        def test_1(arg):
            pass

        run_tests(arg, test_1, rootpath=tmp_path).assert_outcomes(passed=1)

    def test_setupcontext_no_param(self, tmp_path: Path) -> None:
        @pytest.fixture(params=[1, 2])
        def arg(request):
            return request.param

        @pytest.fixture(autouse=True)
        def mysetup(request, arg):
            assert not hasattr(request, "param")

        def test_1(arg):
            assert arg in (1, 2)

        run_tests(arg, mysetup, test_1, rootpath=tmp_path).assert_outcomes(passed=2)


class TestRequestSessionScoped:
    @pytest.fixture(scope="session")
    def session_request(self, request):
        return request

    @pytest.mark.parametrize("name", ["path", "module"])
    def test_session_scoped_unavailable_attributes(self, session_request, name):
        with pytest.raises(
            AttributeError,
            match=f"{name} not available in session-scoped context",
        ):
            getattr(session_request, name)


class TestRequestMarking:
    def test_applymarker(self, tmp_path: Path) -> None:
        @pytest.fixture
        def something(request):
            pass

        class TestClass:
            def test_func1(self, something):
                pass

            def test_func2(self, something):
                pass

        with Ensemble(something, TestClass, rootpath=tmp_path) as ensemble:
            item1, _item2 = ensemble.collect()
            assert isinstance(item1, Function)
            req1 = TopRequest(item1, _ispytest=True)
            assert "xfail" not in item1.keywords
            req1.applymarker(pytest.mark.xfail)
            assert "xfail" in item1.keywords
            assert "skipif" not in item1.keywords
            req1.applymarker(pytest.mark.skipif)
            assert "skipif" in item1.keywords
            with pytest.raises(ValueError):
                req1.applymarker(42)  # type: ignore[arg-type]

    def test_accesskeywords(self, tmp_path: Path) -> None:
        @pytest.fixture
        def keywords(request):
            return request.keywords

        @unregistered_mark("XYZ")
        def test_function(keywords):
            assert keywords["XYZ"]
            assert "abc" not in keywords

        run_tests(keywords, test_function, rootpath=tmp_path).assert_outcomes(passed=1)

    def test_accessmarker_dynamic(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            @pytest.fixture
            def keywords(self, request):
                return request.keywords

            @pytest.fixture(scope="class", autouse=True)
            def marking(self, request):
                request.applymarker(pytest.mark.XYZ("hello"))

        def test_fun1(keywords):
            assert keywords["XYZ"] is not None
            assert "abc" not in keywords

        def test_fun2(keywords):
            assert keywords["XYZ"] is not None
            assert "abc" not in keywords

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        run_tests(test_fun1, test_fun2, spec=spec).assert_outcomes(passed=2)


class TestFixtureUsages:
    def test_noargfixturedec(self, tmp_path: Path) -> None:
        @pytest.fixture
        def arg1():
            return 1

        def test_func(arg1):
            assert arg1 == 1

        run_tests(arg1, test_func, rootpath=tmp_path).assert_outcomes(passed=1)

    def test_receives_funcargs(self, tmp_path: Path) -> None:
        @pytest.fixture
        def arg1():
            return 1

        @pytest.fixture
        def arg2(arg1):
            return arg1 + 1

        def test_add(arg2):
            assert arg2 == 2

        def test_all(arg1, arg2):
            assert arg1 == 1
            assert arg2 == 2

        record = run_tests(arg1, arg2, test_add, test_all, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    # ensemble: asserts the fixtures' file:line, which is host-anchored for
    # in-memory sources (see test_receives_funcargs_scope_mismatch_issue660
    # for the same failure asserted without locations).
    def test_receives_funcargs_scope_mismatch(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="function")
            def arg1():
                return 1

            @pytest.fixture(scope="module")
            def arg2(arg1):
                return arg1 + 1

            def test_add(arg2):
                assert arg2 == 2
        """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ScopeMismatch*Requesting fixture stack*",
                "test_receives_funcargs_scope_mismatch.py:6:  def arg2(arg1)",
                "Requested fixture:",
                "test_receives_funcargs_scope_mismatch.py:2:  def arg1()",
                "*1 error*",
            ]
        )

    def test_receives_funcargs_scope_mismatch_issue660(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="function")
        def arg1():
            return 1

        @pytest.fixture(scope="module")
        def arg2(arg1):
            return arg1 + 1

        def test_add(arg1, arg2):
            assert arg2 == 2

        record = run_tests(arg1, arg2, test_add, rootpath=tmp_path, capture_output=True)
        record.stdout.fnmatch_lines(
            [
                "*ScopeMismatch*Requesting fixture stack*",
                "* def arg2(arg1)",
                "Requested fixture:",
                "* def arg1()",
                "*1 error*",
            ],
        )

    def test_invalid_scope(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="functions")  # type: ignore[call-overload]
        def badscope():
            pass

        def test_nothing(badscope):
            pass

        record = run_tests(
            build_module("test_invalid_scope", badscope, test_nothing),
            rootpath=tmp_path,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            "*Fixture 'badscope' from test_invalid_scope.py got an unexpected scope value 'functions'"
        )

    @pytest.mark.parametrize("scope", ["function", "session"])
    def test_parameters_without_eq_semantics(self, scope, tmp_path: Path) -> None:
        class NoEq1:  # fails on `a == b` statement
            def __eq__(self, _):
                raise RuntimeError

        class NoEq2:  # fails on `if a == b:` statement
            def __eq__(self, _):
                class NoBool:
                    def __bool__(self):
                        raise RuntimeError

                return NoBool()

        @pytest.fixture(params=[NoEq1(), NoEq2()], scope=scope)
        def no_eq(request):
            return request.param

        def test1(no_eq):
            pass

        def test2(no_eq):
            pass

        record = run_tests(no_eq, test1, test2, rootpath=tmp_path)
        record.assert_outcomes(passed=4)

    def test_funcarg_parametrized_and_used_twice(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(params=[1, 2])
        def arg1(request):
            values.append(1)
            return request.param

        @pytest.fixture
        def arg2(arg1):
            return arg1 + 1

        def test_add(arg1, arg2):
            assert arg2 == arg1 + 1
            assert len(values) == arg1

        record = run_tests(arg1, arg2, test_add, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_factory_uses_unknown_funcarg_as_dependency_error(
        self, tmp_path: Path
    ) -> None:
        @pytest.fixture
        def fail(missing):
            return

        @pytest.fixture
        def call_fail(fail):
            return

        def test_missing(call_fail):
            pass

        # ``fail`` carries the fixture wrapper's name, so it is passed by
        # keyword to keep its own name in the synthesized module.
        record = run_tests(
            build_module(
                "test_unknown_dependency",
                call_fail,
                test_missing,
                fail=fail,
            ),
            rootpath=tmp_path,
            capture_output=True,
        )
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            [
                "*pytest.fixture*",
                "*def call_fail(fail)*",
                "*pytest.fixture*",
                "*def fail*",
                "*fixture*'missing'*not found*",
            ]
        )

    # ensemble: the subject is an exception raised while *importing* the test
    # module; ensembles serve a preset module object and never import.
    def test_factory_setup_as_classes_fails(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            class arg1(object):
                def __init__(self, request):
                    self.x = 1
            arg1 = pytest.fixture()(arg1)

        """
        )
        reprec = pytester.inline_run()
        values = reprec.getfailedcollections()
        assert len(values) == 1

    def test_usefixtures_marker(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(scope="class")
        def myfix(request):
            request.cls.hello = "world"
            values.append(1)

        class TestClass:
            hello: str  # set by the ``myfix`` fixture

            def test_one(self):
                assert self.hello == "world"
                assert len(values) == 1

            def test_two(self):
                assert self.hello == "world"
                assert len(values) == 1

        pytest.mark.usefixtures("myfix")(TestClass)

        record = run_tests(myfix, TestClass, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_empty_usefixtures_marker(self, tmp_path: Path) -> None:
        """Empty usefixtures() marker issues a warning (#12439)."""

        @pytest.mark.usefixtures()
        def test_one():
            assert 1 == 1

        spec = ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["always"]})
        record = run_tests(
            build_module("test_empty_usefixtures_marker", test_one), spec=spec
        )
        record.assert_outcomes(passed=1, warnings=1)
        assert str(record.warnings[0].message) == (
            "usefixtures() in test_empty_usefixtures_marker.py::test_one"
            " without arguments has no effect"
        )

    def test_usefixtures_ini(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            @pytest.fixture(scope="class")
            def myfix(self, request):
                request.cls.hello = "world"

        class TestClass:
            hello: str  # set by the ``myfix`` fixture

            def test_one(self):
                assert self.hello == "world"

            def test_two(self):
                assert self.hello == "world"

        spec = ConfigSpec(
            rootpath=tmp_path,
            inicfg={"usefixtures": ["myfix"]},
            extra_plugins=(ConftestPlugin(),),
        )
        run_tests(TestClass, spec=spec).assert_outcomes(passed=2)

    # ensemble: --markers runs through pytest_cmdline_main, unreachable.
    def test_usefixtures_seen_in_showmarkers(self, pytester: Pytester) -> None:
        result = pytester.runpytest("--markers")
        result.stdout.fnmatch_lines(
            """
            *usefixtures(fixturename1*mark tests*fixtures*
        """
        )

    def test_request_instance_issue203(self, tmp_path: Path) -> None:
        class TestClass:
            @pytest.fixture
            def setup1(self, request):
                assert self == request.instance
                self.arg1 = 1

            def test_hello(self, setup1):
                assert self.arg1 == 1

        run_tests(TestClass, rootpath=tmp_path).assert_outcomes(passed=1)

    def test_fixture_parametrized_with_iterator(self, tmp_path: Path) -> None:
        values: list[int] = []

        def f():
            yield 1
            yield 2

        dec = pytest.fixture(scope="module", params=f())

        @dec
        def arg(request):
            return request.param

        @dec
        def arg2(request):
            return request.param

        def test_1(arg):
            values.append(arg)

        def test_2(arg2):
            values.append(arg2 * 10)

        record = run_tests(arg, arg2, test_1, test_2, rootpath=tmp_path)
        record.assert_outcomes(passed=4)
        assert values == [1, 2, 10, 20]

    def test_setup_functions_as_fixtures(self, tmp_path: Path) -> None:
        """Ensure setup_* methods obey fixture scope rules (#517, #3094)."""
        # The original's module global becomes a one-element list, since a
        # source function would see the host module's globals.
        db_initialized: list[bool | None] = [None]

        @pytest.fixture(scope="session", autouse=True)
        def db():
            db_initialized[0] = True
            yield
            db_initialized[0] = False

        def setup_module():
            assert db_initialized[0]

        def teardown_module():
            assert db_initialized[0]

        class TestClass:
            def setup_method(self, method):
                assert db_initialized[0]

            def teardown_method(self, method):
                assert db_initialized[0]

            def test_printer_1(self):
                pass

            def test_printer_2(self):
                pass

        module = build_module(
            "test_setup_functions_as_fixtures",
            db,
            TestClass,
            setup_module=setup_module,
            teardown_module=teardown_module,
        )
        run_tests(module, rootpath=tmp_path).assert_outcomes(passed=2)

    def test_parameterized_fixture_caching(self, tmp_path: Path) -> None:
        """Regression test for #12600."""
        from itertools import count

        cache_misses = count(0)

        def pytest_generate_tests(metafunc):
            if "my_fixture" in metafunc.fixturenames:
                # Use unique objects for parametrization (as opposed to small strings
                # and small integers which are singletons).
                metafunc.parametrize("my_fixture", [[1], [2]], indirect=True)

        @pytest.fixture(scope="session")
        def my_fixture(request):
            next(cache_misses)

        def test1(my_fixture):
            pass

        def test2(my_fixture):
            pass

        def teardown_module():
            assert next(cache_misses) == 2

        module = build_module(
            "test_parameterized_fixture_caching",
            my_fixture,
            test1,
            test2,
            pytest_generate_tests=pytest_generate_tests,
            teardown_module=teardown_module,
        )
        # A failing teardown_module would show up as an error, so asserting
        # the exact outcomes subsumes the original's "no ERROR at teardown".
        run_tests(module, rootpath=tmp_path).assert_outcomes(passed=4)

    def test_unwrapping_pytest_fixture(self, tmp_path: Path) -> None:
        """Ensure the unwrap method on `FixtureFunctionDefinition` correctly wraps and unwraps methods and functions"""
        import inspect

        class FixtureFunctionDefTestClass:
            def __init__(self) -> None:
                self.i = 10

            @pytest.fixture
            def fixture_function_def_test_method(self):
                return self.i

        @pytest.fixture
        def fixture_function_def_test_func():
            return 9

        def test_get_wrapped_func_returns_method():
            obj = FixtureFunctionDefTestClass()
            wrapped_function_result = (
                obj.fixture_function_def_test_method._get_wrapped_function()
            )
            assert inspect.ismethod(wrapped_function_result)
            assert wrapped_function_result() == 10

        def test_get_wrapped_func_returns_function():
            assert fixture_function_def_test_func._get_wrapped_function()() == 9

        record = run_tests(
            test_get_wrapped_func_returns_method,
            test_get_wrapped_func_returns_function,
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=2)

    def test_fixture_wrapped_looks_liked_wrapped_function(self, tmp_path: Path) -> None:
        """Ensure that `FixtureFunctionDefinition` behaves like the function it wrapped."""

        @pytest.fixture
        def fixture_function_def_test_func():
            return 9

        fixture_function_def_test_func.__doc__ = "documentation"

        def test_fixture_has_same_doc():
            assert fixture_function_def_test_func.__doc__ == "documentation"

        record = run_tests(
            fixture_function_def_test_func, test_fixture_has_same_doc, rootpath=tmp_path
        )
        record.assert_outcomes(passed=1)


class TestFixtureManagerParseFactories:
    @pytest.fixture
    def pytester(self, pytester: Pytester) -> Pytester:
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture
            def hello(request):
                return "conftest"

            @pytest.fixture
            def fm(request):
                return request._fixturemanager

            @pytest.fixture
            def item(request):
                return request._pyfuncitem
        """
        )
        return pytester

    @pytest.fixture
    def spec(self, tmp_path: Path) -> ConfigSpec:
        """The rootdir conftest of this class, as a plugin object."""

        class ConftestPlugin:
            @pytest.fixture
            def hello(self, request):
                return "conftest"

            @pytest.fixture
            def fm(self, request):
                return request._fixturemanager

            @pytest.fixture
            def item(self, request):
                return request._pyfuncitem

        return ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))

    def test_parsefactories_evil_objects_issue214(self, spec: ConfigSpec) -> None:
        class A:
            def __call__(self):
                pass

            def __getattr__(self, name):
                raise RuntimeError()

        def test_hello():
            pass

        record = run_tests(build_module("test_evil", test_hello, a=A()), spec=spec)
        record.assert_outcomes(passed=1, failed=0)

    def test_parsefactories_conftest(self, spec: ConfigSpec) -> None:
        def test_hello(item, fm):
            for name in ("fm", "hello", "item"):
                faclist = fm.getfixturedefs(name, item)
                assert len(faclist) == 1
                fac = faclist[0]
                assert fac.func.__name__ == name

        run_tests(test_hello, spec=spec).assert_outcomes(passed=1)

    def test_parsefactories_conftest_and_module_and_class(
        self, spec: ConfigSpec
    ) -> None:
        @pytest.fixture
        def hello(request):
            return "module"

        class TestClass:
            @pytest.fixture
            def hello(self, request):
                return "class"

            def test_hello(self, item, fm):
                faclist = fm.getfixturedefs("hello", item)
                print(faclist)
                assert len(faclist) == 3

                assert faclist[0].func(item._request) == "conftest"
                assert faclist[1].func(item._request) == "module"
                assert faclist[2].func(item._request) == "class"

        record = run_tests(
            build_module("test_three_levels", TestClass, hello=hello), spec=spec
        )
        record.assert_outcomes(passed=1)

    def test_register_fixture_ordered_by_visibility(self, tmp_path: Path) -> None:
        """A fixturedef registered for a more specific node takes precedence
        over one registered for a more general (ancestor) node, regardless of
        the order in which they were registered (#14513)."""

        class ConftestPlugin:
            @pytest.hookimpl(wrapper=True)
            def pytest_collection(self, session):
                result = yield
                item = session.items[0]
                pytest.register_fixture(
                    name="fix", func=lambda: "session1", node=session
                )
                # For coverage; can be removed once nodeid= deprecation is over.
                fm = session._fixturemanager
                fm._register_fixture(
                    name="fix", func=lambda: "session-legacy", nodeid=""
                )
                fm._register_fixture(
                    name="fix", func=lambda: "broken-legacy", nodeid="broken"
                )
                pytest.register_fixture(
                    name="fix", func=lambda fix: f"item1-{fix}", node=item
                )
                pytest.register_fixture(
                    name="fix", func=lambda fix: f"item2-{fix}", node=item
                )
                pytest.register_fixture(
                    name="fix", func=lambda: "session2", node=session
                )
                return result

        def test(fix):
            assert fix == "item2-item1-session2"

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        run_tests(test, spec=spec).assert_outcomes(passed=1)

    # ensemble: conftests in sibling directories, run from a third one.
    def test_parsefactories_relative_node_ids(
        self, pytester: Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        # example mostly taken from:
        # https://mail.python.org/pipermail/pytest-dev/2014-September/002617.html
        runner = pytester.mkdir("runner")
        package = pytester.mkdir("package")
        package.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
            import pytest
            @pytest.fixture
            def one():
                return 1
            """
            ),
            encoding="utf-8",
        )
        package.joinpath("test_x.py").write_text(
            textwrap.dedent(
                """\
                def test_x(one):
                    assert one == 1
                """
            ),
            encoding="utf-8",
        )
        sub = package.joinpath("sub")
        sub.mkdir()
        sub.joinpath("__init__.py").touch()
        sub.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def one():
                    return 2
                """
            ),
            encoding="utf-8",
        )
        sub.joinpath("test_y.py").write_text(
            textwrap.dedent(
                """\
                def test_x(one):
                    assert one == 2
                """
            ),
            encoding="utf-8",
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)
        with monkeypatch.context() as mp:
            mp.chdir(runner)
            reprec = pytester.inline_run("..")
            reprec.assertoutcome(passed=2)

    # ensemble: package layout (__init__.py, relative imports).
    def test_package_xunit_fixture(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            __init__="""\
            values = []
        """
        )
        package = pytester.mkdir("package")
        package.joinpath("__init__.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def setup_module():
                    values.append("package")
                def teardown_module():
                    values[:] = []
                """
            ),
            encoding="utf-8",
        )
        package.joinpath("test_x.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def test_x():
                    assert values == ["package"]
                """
            ),
            encoding="utf-8",
        )
        package = pytester.mkdir("package2")
        package.joinpath("__init__.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def setup_module():
                    values.append("package2")
                def teardown_module():
                    values[:] = []
                """
            ),
            encoding="utf-8",
        )
        package.joinpath("test_x.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def test_x():
                    assert values == ["package2"]
                """
            ),
            encoding="utf-8",
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    # ensemble: package layout and package-scoped fixtures.
    def test_package_fixture_complex(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            __init__="""\
            values = []
        """
        )
        pytester.syspathinsert(pytester.path.name)
        package = pytester.mkdir("package")
        package.joinpath("__init__.py").write_text("", encoding="utf-8")
        package.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                from .. import values
                @pytest.fixture(scope="package")
                def one():
                    values.append("package")
                    yield values
                    values.pop()
                @pytest.fixture(scope="package", autouse=True)
                def two():
                    values.append("package-auto")
                    yield values
                    values.pop()
                """
            ),
            encoding="utf-8",
        )
        package.joinpath("test_x.py").write_text(
            textwrap.dedent(
                """\
                from .. import values
                def test_package_autouse():
                    assert values == ["package-auto"]
                def test_package(one):
                    assert values == ["package-auto", "package"]
                """
            ),
            encoding="utf-8",
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    # ensemble: the example is a directory tree with a conftest defining
    # custom collectors for non-python files. The items under test are not
    # python at all, so there is no module to run in place.
    def test_collect_custom_items(self, pytester: Pytester) -> None:
        pytester.copy_example("fixtures/custom_item")
        result = pytester.runpytest("foo")
        result.stdout.fnmatch_lines(["*passed*"])


class TestAutouseDiscovery:
    @pytest.fixture
    def pytester(self, pytester: Pytester) -> Pytester:
        pytester.makeconftest(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def perfunction(request, tmp_path):
                pass

            @pytest.fixture()
            def arg1(tmp_path):
                pass
            @pytest.fixture(autouse=True)
            def perfunction2(arg1):
                pass

            @pytest.fixture
            def fm(request):
                return request._fixturemanager

            @pytest.fixture
            def item(request):
                return request._pyfuncitem
        """
        )
        return pytester

    @pytest.fixture
    def spec(self, tmp_path: Path) -> ConfigSpec:
        """The rootdir conftest of this class, as a plugin object."""

        class ConftestPlugin:
            @pytest.fixture(autouse=True)
            def perfunction(self, request, tmp_path):
                pass

            @pytest.fixture
            def arg1(self, tmp_path):
                pass

            @pytest.fixture(autouse=True)
            def perfunction2(self, arg1):
                pass

            @pytest.fixture
            def fm(self, request):
                return request._fixturemanager

            @pytest.fixture
            def item(self, request):
                return request._pyfuncitem

        return ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))

    def test_parsefactories_conftest(self, spec: ConfigSpec) -> None:
        def test_check_setup(item, fm):
            autousenames = list(fm._getautousenames(item))
            assert len(get_public_names(autousenames)) == 2
            assert "perfunction2" in autousenames
            assert "perfunction" in autousenames

        run_tests(test_check_setup, spec=spec).assert_outcomes(passed=1)

    def test_two_classes_separated_autouse(self, tmp_path: Path) -> None:
        class TestA:
            values: list[int] = []

            @pytest.fixture(autouse=True)
            def setup1(self):
                self.values.append(1)

            def test_setup1(self):
                assert self.values == [1]

        class TestB:
            values: list[int] = []

            @pytest.fixture(autouse=True)
            def setup2(self):
                self.values.append(1)

            def test_setup2(self):
                assert self.values == [1]

        run_tests(TestA, TestB, rootpath=tmp_path).assert_outcomes(passed=2)

    def test_setup_at_classlevel(self, tmp_path: Path) -> None:
        class TestClass:
            funcname: str  # set by the ``permethod`` fixture

            @pytest.fixture(autouse=True)
            def permethod(self, request):
                request.instance.funcname = request.function.__name__

            def test_method1(self):
                assert self.funcname == "test_method1"

            def test_method2(self):
                assert self.funcname == "test_method2"

        run_tests(TestClass, rootpath=tmp_path).assert_outcomes(passed=2)

    # ensemble: `pytest.fixture(enabled=...)` is rejected at decoration time
    # by the host, so the unimplemented feature cannot be spelled inline.
    @pytest.mark.xfail(reason="'enabled' feature not implemented")
    def test_setup_enabled_functionnode(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def enabled(parentnode, markers):
                return "needsdb" in markers

            @pytest.fixture(params=[1,2])
            def db(request):
                return request.param

            @pytest.fixture(enabled=enabled, autouse=True)
            def createdb(db):
                pass

            def test_func1(request):
                assert "db" not in request.fixturenames

            @pytest.mark.needsdb
            def test_func2(request):
                assert "db" in request.fixturenames
        """
        )
        reprec = pytester.inline_run("-s")
        reprec.assertoutcome(passed=2)

    def test_callables_nocode(self, tmp_path: Path) -> None:
        """An imported mock.call would break setup/factory discovery due to
        it being callable and __code__ not being a code object."""

        class _call(tuple[object, ...]):
            def __call__(self, *k, **kw):
                pass

            def __getattr__(self, k):
                return self

        # collect_tests raises rather than returning an empty list if
        # collection blew up, so this really means "collected nothing".
        assert (
            collect_tests(
                build_module("test_callables_nocode", call=_call()), rootpath=tmp_path
            )
            == []
        )

    # ensemble: an autouse fixture in one subdirectory's conftest must not
    # reach a sibling directory - that is directory scoping.
    def test_autouse_in_conftests(self, pytester: Pytester) -> None:
        a = pytester.mkdir("a")
        b = pytester.mkdir("a1")
        conftest = pytester.makeconftest(
            """
            import pytest
            @pytest.fixture(autouse=True)
            def hello():
                xxx
        """
        )
        conftest.rename(a.joinpath(conftest.name))
        a.joinpath("test_something.py").write_text(
            "def test_func(): pass", encoding="utf-8"
        )
        b.joinpath("test_otherthing.py").write_text(
            "def test_func(): pass", encoding="utf-8"
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            """
            *1 passed*1 error*
        """
        )

    def test_autouse_in_module_and_two_classes(self, tmp_path: Path) -> None:
        values: list[str] = []

        @pytest.fixture(autouse=True)
        def append1():
            values.append("module")

        def test_x():
            assert values == ["module"]

        class TestA:
            @pytest.fixture(autouse=True)
            def append2(self):
                values.append("A")

            def test_hello(self):
                assert values == ["module", "module", "A"], values

        class TestA2:
            def test_world(self):
                assert values == ["module", "module", "A", "module"], values

        record = run_tests(append1, test_x, TestA, TestA2, rootpath=tmp_path)
        record.assert_outcomes(passed=3)


class TestAutouseManagement:
    # ensemble: the conftest sits in a directory between rootdir and the test.
    def test_autouse_conftest_mid_directory(self, pytester: Pytester) -> None:
        pkgdir = pytester.mkpydir("xyz123")
        pkgdir.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture(autouse=True)
                def app():
                    import sys
                    sys._myapp = "hello"
                """
            ),
            encoding="utf-8",
        )
        sub = pkgdir.joinpath("tests")
        sub.mkdir()
        t = sub.joinpath("test_app.py")
        t.touch()
        t.write_text(
            textwrap.dedent(
                """\
                import sys
                def test_app():
                    assert sys._myapp == "hello"
                """
            ),
            encoding="utf-8",
        )
        reprec = pytester.inline_run("-s")
        reprec.assertoutcome(passed=1)

    def test_funcarg_and_setup(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(scope="module")
        def arg():
            values.append(1)
            return 0

        @pytest.fixture(scope="module", autouse=True)
        def something(arg):
            values.append(2)

        def test_hello(arg):
            assert len(values) == 2
            assert values == [1, 2]
            assert arg == 0

        def test_hello2(arg):
            assert len(values) == 2
            assert values == [1, 2]
            assert arg == 0

        record = run_tests(arg, something, test_hello, test_hello2, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_uses_parametrized_resource(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(params=[1, 2])
        def arg(request):
            return request.param

        @pytest.fixture(autouse=True)
        def something(arg):
            values.append(arg)

        def test_hello():
            if len(values) == 1:
                assert values == [1]
            elif len(values) == 2:
                assert values == [1, 2]
            else:
                0 / 0  # noqa: B018

        record = run_tests(arg, something, test_hello, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_session_parametrized_function(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(scope="session", params=[1, 2])
        def arg(request):
            return request.param

        @pytest.fixture(scope="function", autouse=True)
        def append(request, arg):
            if request.function.__name__ == "test_some":
                values.append(arg)

        def test_some():
            pass

        def test_result(arg):
            assert len(values) == arg
            assert values[:arg] == [1, 2][:arg]

        record = run_tests(arg, append, test_some, test_result, rootpath=tmp_path)
        record.assert_outcomes(passed=4)

    def test_class_function_parametrization_finalization(self, tmp_path: Path) -> None:
        values: list[str] = []

        class ConftestPlugin:
            @pytest.fixture(scope="function", params=[1, 2])
            def farg(self, request):
                return request.param

            @pytest.fixture(scope="class", params=list("ab"))
            def carg(self, request):
                return request.param

            @pytest.fixture(scope="function", autouse=True)
            def append(self, request, farg, carg):
                def fin():
                    values.append(f"fin_{carg}{farg}")

                request.addfinalizer(fin)

        class TestClass:
            def test_1(self):
                pass

        class TestClass2:
            def test_2(self):
                pass

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(TestClass, TestClass2, spec=spec)
        record.assert_outcomes(passed=8)
        assert values == ["fin_a1", "fin_a2", "fin_b1", "fin_b2"] * 2

    def test_scope_ordering(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(scope="function", autouse=True)
        def fappend2():
            values.append(2)

        @pytest.fixture(scope="class", autouse=True)
        def classappend3():
            values.append(3)

        @pytest.fixture(scope="module", autouse=True)
        def mappend():
            values.append(1)

        class TestHallo:
            def test_method(self):
                assert values == [1, 3, 2]

        record = run_tests(
            fappend2, classappend3, mappend, TestHallo, rootpath=tmp_path
        )
        record.assert_outcomes(passed=1)

    def test_parametrization_setup_teardown_ordering(self, tmp_path: Path) -> None:
        values: list[str] = []

        def pytest_generate_tests(metafunc):
            if metafunc.cls is None:
                assert metafunc.function is test_finish
            if metafunc.cls is not None:
                metafunc.parametrize("item", [1, 2], scope="class")

        class TestClass:
            @pytest.fixture(scope="class", autouse=True)
            @classmethod
            def setup_teardown(cls, item):
                values.append(f"setup-{item}")
                yield
                values.append(f"teardown-{item}")

            def test_step1(self, item):
                values.append(f"step1-{item}")

            def test_step2(self, item):
                values.append(f"step2-{item}")

        def test_finish():
            assert values == [
                "setup-1",
                "step1-1",
                "step2-1",
                "teardown-1",
                "setup-2",
                "step1-2",
                "step2-2",
                "teardown-2",
            ]

        module = build_module(
            "test_setup_teardown_ordering",
            TestClass,
            test_finish,
            pytest_generate_tests=pytest_generate_tests,
        )
        run_tests(module, rootpath=tmp_path).assert_outcomes(passed=5)

    def test_ordering_autouse_before_explicit(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(autouse=True)
        def fix1():
            values.append(1)

        @pytest.fixture
        def arg1():
            values.append(2)

        def test_hello(arg1):
            assert values == [1, 2]

        run_tests(fix1, arg1, test_hello, rootpath=tmp_path).assert_outcomes(passed=1)

    @pytest.mark.parametrize("param1", [{}, {"params": [1]}], ids=["p00", "p01"])
    @pytest.mark.parametrize("param2", [{}, {"params": [1]}], ids=["p10", "p11"])
    def test_ordering_dependencies_torndown_first(
        self, tmp_path: Path, param1, param2
    ) -> None:
        """#226"""
        values: list[str] = []

        @pytest.fixture(**param1)
        def arg1(request):
            request.addfinalizer(lambda: values.append("fin1"))
            values.append("new1")

        @pytest.fixture(**param2)
        def arg2(request, arg1):
            request.addfinalizer(lambda: values.append("fin2"))
            values.append("new2")

        def test_arg(arg2):
            pass

        def test_check():
            assert values == ["new1", "new2", "fin2", "fin1"]

        record = run_tests(arg1, arg2, test_arg, test_check, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_reordering_catastrophic_performance(self, tmp_path: Path) -> None:
        """Check that a certain high-scope parametrization pattern doesn't cause
        a catasrophic slowdown.

        Regression test for #12355.
        """
        params = tuple("abcdefghijklmnopqrstuvwxyz")

        @pytest.mark.parametrize(params, [range(len(params))] * 3, scope="module")
        def test_parametrize(
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            i,
            j,
            k,
            l,  # noqa: E741
            m,
            n,
            o,
            p,
            q,
            r,
            s,
            t,
            u,
            v,
            w,
            x,
            y,
            z,
        ):
            pass

        run_tests(test_parametrize, rootpath=tmp_path).assert_outcomes(passed=3)


class TestFixtureMarker:
    def test_parametrize(self, tmp_path: Path) -> None:
        values: list[str] = []

        @pytest.fixture(params=["a", "b", "c"])
        def arg(request):
            return request.param

        def test_param(arg):
            values.append(arg)

        def test_result():
            assert values == list("abc")

        record = run_tests(arg, test_param, test_result, rootpath=tmp_path)
        record.assert_outcomes(passed=4)

    def test_multiple_parametrization_issue_736(self, tmp_path: Path) -> None:
        @pytest.fixture(params=[1, 2, 3])
        def foo(request):
            return request.param

        @pytest.mark.parametrize("foobar", [4, 5, 6])
        def test_issue(foo, foobar):
            assert foo in [1, 2, 3]
            assert foobar in [4, 5, 6]

        run_tests(foo, test_issue, rootpath=tmp_path).assert_outcomes(passed=9)

    @pytest.mark.parametrize(
        "param_args",
        ["fixt, val", "fixt,val", ["fixt", "val"], ("fixt", "val")],
    )
    def test_override_parametrized_fixture_issue_979(
        self, tmp_path: Path, param_args
    ) -> None:
        """Make sure a parametrized argument can override a parametrized fixture.

        This was a regression introduced in the fix for #736.
        """

        @pytest.fixture(params=[1, 2])
        def fixt(request):
            return request.param

        @pytest.mark.parametrize(param_args, [(3, "x"), (4, "x")])
        def test_foo(fixt, val):
            pass

        run_tests(fixt, test_foo, rootpath=tmp_path).assert_outcomes(passed=2)

    def test_override_parametrized_fixture_with_indirect(self, tmp_path: Path) -> None:
        """Make sure a parametrized argument can override a parametrized fixture.

        This was a regression introduced in the fix for #736.
        """

        @pytest.fixture(params=["a"])
        def fixt(request):
            return request.param * 2

        def test_fixt(fixt):
            assert fixt == "aa"

        @pytest.mark.parametrize("fixt", ["b"], indirect=True)
        def test_indirect(fixt):
            assert fixt == "bb"

        record = run_tests(fixt, test_fixt, test_indirect, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_scope_session(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(scope="module")
        def arg():
            values.append(1)
            return 1

        def test_1(arg):
            assert arg == 1

        def test_2(arg):
            assert arg == 1
            assert len(values) == 1

        class TestClass:
            def test3(self, arg):
                assert arg == 1
                assert len(values) == 1

        record = run_tests(arg, test_1, test_2, TestClass, rootpath=tmp_path)
        record.assert_outcomes(passed=3)

    def test_scope_session_exc(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(scope="session")
        def fix():
            values.append(1)
            pytest.skip("skipping")

        def test_1(fix):
            pass

        def test_2(fix):
            pass

        def test_last():
            assert values == [1]

        record = run_tests(fix, test_1, test_2, test_last, rootpath=tmp_path)
        record.assert_outcomes(skipped=2, passed=1)

    def test_scope_session_exc_two_fix(self, tmp_path: Path) -> None:
        values: list[int] = []
        m: list[int] = []

        @pytest.fixture(scope="session")
        def a():
            values.append(1)
            pytest.skip("skipping")

        @pytest.fixture(scope="session")
        def b(a):
            m.append(1)

        def test_1(b):
            pass

        def test_2(b):
            pass

        def test_last():
            assert values == [1]
            assert m == []

        record = run_tests(a, b, test_1, test_2, test_last, rootpath=tmp_path)
        record.assert_outcomes(skipped=2, passed=1)

    def test_scope_exc(self, tmp_path: Path) -> None:
        reqs: list[int] = []

        class ConftestPlugin:
            @pytest.fixture(scope="session")
            def fix(self, request):
                reqs.append(1)
                pytest.skip()

            @pytest.fixture
            def req_list(self):
                return reqs

        def test_foo(fix):
            pass

        def test_bar(fix):
            pass

        def test_last(req_list):
            assert req_list == [1]

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_foo", test_foo),
            build_module("test_bar", test_bar),
            build_module("test_real", test_last),
            spec=spec,
        )
        record.assert_outcomes(skipped=2, passed=1)

    def test_scope_module_uses_session(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(scope="module")
        def arg():
            values.append(1)
            return 1

        def test_1(arg):
            assert arg == 1

        def test_2(arg):
            assert arg == 1
            assert len(values) == 1

        class TestClass:
            def test3(self, arg):
                assert arg == 1
                assert len(values) == 1

        record = run_tests(arg, test_1, test_2, TestClass, rootpath=tmp_path)
        record.assert_outcomes(passed=3)

    def test_scope_module_and_finalizer(self, tmp_path: Path) -> None:
        finalized_list: list[int] = []
        created_list: list[int] = []

        class ConftestPlugin:
            @pytest.fixture(scope="module")
            def arg(self, request):
                created_list.append(1)
                assert request.scope == "module"
                request.addfinalizer(lambda: finalized_list.append(1))

            @pytest.fixture
            def created(self, request):
                return len(created_list)

            @pytest.fixture
            def finalized(self, request):
                return len(finalized_list)

        def test_1(arg, created, finalized):
            assert created == 1
            assert finalized == 0

        def test_2(arg, created, finalized):
            assert created == 1
            assert finalized == 0

        def test_3(arg, created, finalized):
            assert created == 2
            assert finalized == 1

        def test_4(arg, created, finalized):
            assert created == 3
            assert finalized == 2

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_mod1", test_1, test_2),
            build_module("test_mod2", test_3),
            build_module("test_mode3", test_4),
            spec=spec,
        )
        record.assert_outcomes(passed=4)

    def test_scope_mismatch_various(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            @pytest.fixture(scope="function")
            def arg(self, request):
                pass

        @pytest.fixture(scope="session")
        def arg(request):
            request.getfixturevalue("arg")

        def test_1(arg):
            pass

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_mod1", arg=arg, test_1=test_1),
            spec=spec,
            capture_output=True,
        )
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            ["*ScopeMismatch*You tried*function*session*request*"]
        )

    # ensemble: asserts the fixtures' file:line, host-anchored for in-memory
    # sources.
    def test_scope_mismatch_already_computed_dynamic(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_it="""
                import pytest

                @pytest.fixture(scope="function")
                def fixfunc(): pass

                @pytest.fixture(scope="module")
                def fixmod(fixfunc): pass

                def test_it(request, fixfunc):
                    request.getfixturevalue("fixmod")
            """,
        )

        result = pytester.runpytest()
        assert result.ret == ExitCode.TESTS_FAILED
        result.stdout.fnmatch_lines(
            [
                "*ScopeMismatch*Requesting fixture stack*",
                "test_it.py:6:  def fixmod(fixfunc)",
                "Requested fixture:",
                "test_it.py:3:  def fixfunc()",
            ]
        )

    def test_dynamic_scope(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def dynamic_scope(fixture_name, config):
            if config.getoption("--extend-scope"):
                return "session"
            return "function"

        class ConftestPlugin:
            def pytest_addoption(self, parser):
                parser.addoption("--extend-scope", action="store_true", default=False)

            @pytest.fixture(scope=dynamic_scope)
            def dynamic_fixture(self):
                calls.append("call")
                return len(calls)

        def test_first(dynamic_fixture):
            assert dynamic_fixture == 1

        def test_second(dynamic_fixture):
            assert dynamic_fixture == 2

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(test_first, test_second, spec=spec)
        record.assert_outcomes(passed=2)

        calls.clear()
        record = run_tests(
            test_first, test_second, spec=spec.replace(args=("--extend-scope",))
        )
        record.assert_outcomes(passed=1, failed=1)

    def test_dynamic_scope_bad_return(self, tmp_path: Path) -> None:
        def dynamic_scope(**_):
            return "wrong-scope"

        @pytest.fixture(scope=dynamic_scope)  # type: ignore[arg-type]
        def fixture():
            pass

        record = run_tests(
            build_module("test_dynamic_scope_bad_return", fixture=fixture),
            rootpath=tmp_path,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            "*Fixture 'fixture' from test_dynamic_scope_bad_return.py "
            "got an unexpected scope value 'wrong-scope'*"
        )

    def test_register_only_with_mark(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            @pytest.fixture
            def arg(self):
                return 1

        @pytest.fixture
        def arg(arg):
            return arg + 1

        def test_1(arg):
            assert arg == 2

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(build_module("test_mod1", arg=arg, test_1=test_1), spec=spec)
        record.assert_outcomes(passed=1)

    def test_parametrize_and_scope(self, tmp_path: Path) -> None:
        values: list[str] = []

        @pytest.fixture(scope="module", params=["a", "b", "c"])
        def arg(request):
            return request.param

        def test_param(arg):
            values.append(arg)

        record = run_tests(arg, test_param, rootpath=tmp_path)
        record.assert_outcomes(passed=3)
        assert len(values) == 3
        assert "a" in values
        assert "b" in values
        assert "c" in values

    def test_scope_mismatch(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            @pytest.fixture(scope="function")
            def arg(self, request):
                pass

        @pytest.fixture(scope="session")
        def arg(arg):
            pass

        def test_mismatch(arg):
            pass

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(
            build_module("test_mismatch", arg=arg, test_mismatch=test_mismatch),
            spec=spec,
            capture_output=True,
        )
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(["*ScopeMismatch*", "*1 error*"])

    def test_parametrize_separated_order(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(scope="module", params=[1, 2])
        def arg(request):
            return request.param

        def test_1(arg):
            values.append(arg)

        def test_2(arg):
            values.append(arg)

        record = run_tests(arg, test_1, test_2, rootpath=tmp_path)
        record.assert_outcomes(passed=4)
        assert values == [1, 1, 2, 2]

    def test_module_parametrized_ordering(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            @pytest.fixture(scope="session", params="s1 s2".split())
            def sarg(self):
                pass

            @pytest.fixture(scope="module", params="m1 m2".split())
            def marg(self):
                pass

        def test_func(sarg):
            pass

        def test_func1(marg):
            pass

        def test_func2(sarg):
            pass

        def test_func3(sarg, marg):
            pass

        def test_func3b(sarg, marg):
            pass

        def test_func4(marg):
            pass

        spec = ConfigSpec(
            rootpath=tmp_path,
            args=("-v",),
            inicfg={"console_output_style": "classic"},
            extra_plugins=(ConftestPlugin(),),
        )
        record = run_tests(
            build_module("test_mod1", test_func, test_func1),
            build_module("test_mod2", test_func2, test_func3, test_func3b, test_func4),
            spec=spec,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            """
            test_mod1.py::test_func[s1] PASSED
            test_mod2.py::test_func2[s1] PASSED
            test_mod2.py::test_func3[s1-m1] PASSED
            test_mod2.py::test_func3b[s1-m1] PASSED
            test_mod2.py::test_func3[s1-m2] PASSED
            test_mod2.py::test_func3b[s1-m2] PASSED
            test_mod1.py::test_func[s2] PASSED
            test_mod2.py::test_func2[s2] PASSED
            test_mod2.py::test_func3[s2-m1] PASSED
            test_mod2.py::test_func3b[s2-m1] PASSED
            test_mod2.py::test_func4[m1] PASSED
            test_mod2.py::test_func3[s2-m2] PASSED
            test_mod2.py::test_func3b[s2-m2] PASSED
            test_mod2.py::test_func4[m2] PASSED
            test_mod1.py::test_func1[m1] PASSED
            test_mod1.py::test_func1[m2] PASSED
        """
        )

    def test_dynamic_parametrized_ordering(self, tmp_path: Path) -> None:
        class ConftestPlugin:
            def pytest_configure(self, config):
                class DynamicFixturePlugin:
                    @pytest.fixture(scope="session", params=["flavor1", "flavor2"])
                    def flavor(self, request):
                        return request.param

                config.pluginmanager.register(DynamicFixturePlugin(), "flavor-fixture")

            @pytest.fixture(scope="session", params=["vxlan", "vlan"])
            def encap(self, request):
                return request.param

            @pytest.fixture(scope="session", autouse="True")  # type: ignore[call-overload]
            def reprovision(self, request, flavor, encap):
                pass

        def test(reprovision):
            pass

        def test2(reprovision):
            pass

        spec = ConfigSpec(
            rootpath=tmp_path,
            args=("-v",),
            inicfg={"console_output_style": "classic"},
            extra_plugins=(ConftestPlugin(),),
        )
        record = run_tests(
            build_module("test_dynamic_parametrized_ordering", test, test2),
            spec=spec,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            """
            test_dynamic_parametrized_ordering.py::test[flavor1-vxlan] PASSED
            test_dynamic_parametrized_ordering.py::test2[flavor1-vxlan] PASSED
            test_dynamic_parametrized_ordering.py::test[flavor1-vlan] PASSED
            test_dynamic_parametrized_ordering.py::test2[flavor1-vlan] PASSED
            test_dynamic_parametrized_ordering.py::test[flavor2-vlan] PASSED
            test_dynamic_parametrized_ordering.py::test2[flavor2-vlan] PASSED
            test_dynamic_parametrized_ordering.py::test[flavor2-vxlan] PASSED
            test_dynamic_parametrized_ordering.py::test2[flavor2-vxlan] PASSED
        """
        )

    def test_class_ordering(self, tmp_path: Path) -> None:
        values: list[str] = []

        class ConftestPlugin:
            @pytest.fixture(scope="function", params=[1, 2])
            def farg(self, request):
                return request.param

            @pytest.fixture(scope="class", params=list("ab"))
            def carg(self, request):
                return request.param

            @pytest.fixture(scope="function", autouse=True)
            def append(self, request, farg, carg):
                def fin():
                    values.append(f"fin_{carg}{farg}")

                request.addfinalizer(fin)

        class TestClass2:
            def test_1(self):
                pass

            def test_2(self):
                pass

        class TestClass:
            def test_3(self):
                pass

        spec = ConfigSpec(
            rootpath=tmp_path,
            args=("-v",),
            inicfg={"console_output_style": "classic"},
            extra_plugins=(ConftestPlugin(),),
        )
        record = run_tests(
            build_module("test_class_ordering", TestClass2, TestClass),
            spec=spec,
            capture_output=True,
        )
        record.stdout.re_match_lines(
            r"""
            test_class_ordering.py::TestClass2::test_1\[a-1\] PASSED
            test_class_ordering.py::TestClass2::test_1\[a-2\] PASSED
            test_class_ordering.py::TestClass2::test_2\[a-1\] PASSED
            test_class_ordering.py::TestClass2::test_2\[a-2\] PASSED
            test_class_ordering.py::TestClass2::test_1\[b-1\] PASSED
            test_class_ordering.py::TestClass2::test_1\[b-2\] PASSED
            test_class_ordering.py::TestClass2::test_2\[b-1\] PASSED
            test_class_ordering.py::TestClass2::test_2\[b-2\] PASSED
            test_class_ordering.py::TestClass::test_3\[a-1\] PASSED
            test_class_ordering.py::TestClass::test_3\[a-2\] PASSED
            test_class_ordering.py::TestClass::test_3\[b-1\] PASSED
            test_class_ordering.py::TestClass::test_3\[b-2\] PASSED
        """
        )

    def test_parametrize_separated_order_higher_scope_first(
        self, tmp_path: Path
    ) -> None:
        values: list[str] = []

        @pytest.fixture(scope="function", params=[1, 2])
        def arg(request):
            param = request.param
            request.addfinalizer(lambda: values.append(f"fin:{param}"))
            values.append(f"create:{param}")
            return request.param

        @pytest.fixture(scope="module", params=["mod1", "mod2"])
        def modarg(request):
            param = request.param
            request.addfinalizer(lambda: values.append(f"fin:{param}"))
            values.append(f"create:{param}")
            return request.param

        def test_1(arg):
            values.append("test1")

        def test_2(modarg):
            values.append("test2")

        def test_3(arg, modarg):
            values.append("test3")

        def test_4(modarg, arg):
            values.append("test4")

        record = run_tests(
            arg, modarg, test_1, test_2, test_3, test_4, rootpath=tmp_path
        )
        record.assert_outcomes(passed=12)
        expected = [
            "create:1",
            "test1",
            "fin:1",
            "create:2",
            "test1",
            "fin:2",
            "create:mod1",
            "test2",
            "create:1",
            "test3",
            "fin:1",
            "create:2",
            "test3",
            "fin:2",
            "create:1",
            "test4",
            "fin:1",
            "create:2",
            "test4",
            "fin:2",
            "fin:mod1",
            "create:mod2",
            "test2",
            "create:1",
            "test3",
            "fin:1",
            "create:2",
            "test3",
            "fin:2",
            "create:1",
            "test4",
            "fin:1",
            "create:2",
            "test4",
            "fin:2",
            "fin:mod2",
        ]
        import pprint

        pprint.pprint(list(zip_longest(values, expected)))
        assert values == expected

    def test_parametrized_fixture_teardown_order(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(params=[1, 2], scope="class")
        def param1(request):
            return request.param

        class TestClass:
            @pytest.fixture(scope="class", autouse=True)
            @classmethod
            def setup1(cls, request, param1):
                values.append(1)
                request.addfinalizer(cls.teardown1)

            @classmethod
            def teardown1(self):
                assert values.pop() == 1

            @pytest.fixture(scope="class", autouse=True)
            @classmethod
            def setup2(cls, request, param1):
                values.append(2)
                request.addfinalizer(cls.teardown2)

            @classmethod
            def teardown2(cls):
                assert values.pop() == 2

            def test(self):
                pass

        def test_finish():
            assert not values

        record = run_tests(param1, TestClass, test_finish, rootpath=tmp_path)
        record.assert_outcomes(passed=3)

    # ensemble: the subject is the finalizer of a subdirectory module's
    # override reaching the rootdir conftest fixture, plus -s stdout.
    def test_fixture_finalizer(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest
            import sys

            @pytest.fixture
            def browser(request):

                def finalize():
                    sys.stdout.write_text('Finalized', encoding='utf-8')
                request.addfinalizer(finalize)
                return {}
        """
        )
        b = pytester.mkdir("subdir")
        b.joinpath("test_overridden_fixture_finalizer.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                @pytest.fixture
                def browser(browser):
                    browser['visited'] = True
                    return browser

                def test_browser(browser):
                    assert browser['visited'] is True
                """
            ),
            encoding="utf-8",
        )
        reprec = pytester.runpytest("-s")
        for test in ["test_browser"]:
            reprec.stdout.fnmatch_lines(["*Finalized*"])

    def test_class_scope_with_normal_tests(self, tmp_path: Path) -> None:
        class Box:
            value = 0

        @pytest.fixture(scope="class")
        def a(request):
            Box.value += 1
            return Box.value

        def test_a(a):
            assert a == 1

        class Test1:
            def test_b(self, a):
                assert a == 2

        class Test2:
            def test_c(self, a):
                assert a == 3

        record = run_tests(a, test_a, Test1, Test2, rootpath=tmp_path)
        for test in ["test_a", "test_b", "test_c"]:
            assert record[test].passed

    def test_request_is_clean(self, tmp_path: Path) -> None:
        values: list[int] = []

        @pytest.fixture(params=[1, 2])
        def fix(request):
            request.addfinalizer(lambda: values.append(request.param))

        def test_fix(fix):
            pass

        run_tests(fix, test_fix, rootpath=tmp_path)
        assert values == [1, 2]

    def test_parametrize_separated_lifecycle(self, tmp_path: Path) -> None:
        values: list[object] = []

        @pytest.fixture(scope="module", params=[1, 2])
        def arg(request):
            x = request.param
            request.addfinalizer(lambda: values.append(f"fin{x}"))
            return request.param

        def test_1(arg):
            values.append(arg)

        def test_2(arg):
            values.append(arg)

        record = run_tests(arg, test_1, test_2, rootpath=tmp_path)
        record.assert_outcomes(passed=4)
        import pprint

        pprint.pprint(values)
        # assert len(values) == 6
        assert values[0] == values[1] == 1
        assert values[2] == "fin1"
        assert values[3] == values[4] == 2
        assert values[5] == "fin2"

    def test_parametrize_function_scoped_finalizers_called(
        self, tmp_path: Path
    ) -> None:
        values: list[object] = []

        @pytest.fixture(scope="function", params=[1, 2])
        def arg(request):
            x = request.param
            request.addfinalizer(lambda: values.append(f"fin{x}"))
            return request.param

        def test_1(arg):
            values.append(arg)

        def test_2(arg):
            values.append(arg)

        def test_3():
            assert len(values) == 8
            assert values == [1, "fin1", 2, "fin2", 1, "fin1", 2, "fin2"]

        record = run_tests(arg, test_1, test_2, test_3, rootpath=tmp_path)
        record.assert_outcomes(passed=5)

    @pytest.mark.parametrize("scope", ["session", "function", "module"])
    def test_finalizer_order_on_parametrization(self, scope, tmp_path: Path) -> None:
        """#246"""
        values: list[str] = []

        @pytest.fixture(scope=scope, params=["1"])
        def fix1(request):
            return request.param

        @pytest.fixture(scope=scope)
        def fix2(request, base):
            def cleanup_fix2():
                assert not values, "base should not have been finalized"

            request.addfinalizer(cleanup_fix2)

        @pytest.fixture(scope=scope)
        def base(request, fix1):
            def cleanup_base():
                values.append("fin_base")
                print("finalizing base")

            request.addfinalizer(cleanup_base)

        def test_begin():
            pass

        def test_baz(base, fix2):
            pass

        def test_other():
            pass

        record = run_tests(
            fix1, fix2, base, test_begin, test_baz, test_other, rootpath=tmp_path
        )
        record.assert_outcomes(passed=3)

    def test_class_scope_parametrization_ordering(self, tmp_path: Path) -> None:
        """#396"""
        values: list[str] = []

        @pytest.fixture(params=["John", "Doe"], scope="class")
        def human(request):
            request.addfinalizer(lambda: values.append(f"fin {request.param}"))
            return request.param

        class TestGreetings:
            def test_hello(self, human):
                values.append("test_hello")

        class TestMetrics:
            def test_name(self, human):
                values.append("test_name")

            def test_population(self, human):
                values.append("test_population")

        record = run_tests(human, TestGreetings, TestMetrics, rootpath=tmp_path)
        record.assert_outcomes(passed=6)
        assert values == [
            "test_hello",
            "fin John",
            "test_hello",
            "fin Doe",
            "test_name",
            "test_population",
            "fin John",
            "test_name",
            "test_population",
            "fin Doe",
        ]

    def test_parametrize_setup_function(self, tmp_path: Path) -> None:
        values: list[object] = []

        @pytest.fixture(scope="module", params=[1, 2])
        def arg(request):
            return request.param

        @pytest.fixture(scope="module", autouse=True)
        def mysetup(request, arg):
            request.addfinalizer(lambda: values.append(f"fin{arg}"))
            values.append(f"setup{arg}")

        def test_1(arg):
            values.append(arg)

        def test_2(arg):
            values.append(arg)

        def test_3():
            import pprint

            pprint.pprint(values)
            # ``arg`` is the fixture object here, exactly as in the original
            # module-level source: neither branch is ever taken.
            arg_value: object = arg
            if arg_value == 1:
                assert values == ["setup1", 1, 1]
            elif arg_value == 2:
                assert values == ["setup1", 1, 1, "fin1", "setup2", 2, 2]

        record = run_tests(arg, mysetup, test_1, test_2, test_3, rootpath=tmp_path)
        record.assert_outcomes(passed=6)

    def test_fixture_marked_function_not_collected_as_test(
        self, tmp_path: Path
    ) -> None:
        @pytest.fixture
        def test_app():
            return 1

        def test_something(test_app):
            assert test_app == 1

        record = run_tests(test_app, test_something, rootpath=tmp_path)
        record.assert_outcomes(passed=1)

    def test_params_and_ids(self, tmp_path: Path) -> None:
        @pytest.fixture(params=[object(), object()], ids=["alpha", "beta"])
        def fix(request):
            return request.param

        def test_foo(fix):
            assert 1

        items = collect_tests(fix, test_foo, rootpath=tmp_path)
        assert [item.name for item in items] == ["test_foo[alpha]", "test_foo[beta]"]

    def test_params_and_ids_yieldfixture(self, tmp_path: Path) -> None:
        @pytest.fixture(params=[object(), object()], ids=["alpha", "beta"])
        def fix(request):
            yield request.param

        def test_foo(fix):
            assert 1

        items = collect_tests(fix, test_foo, rootpath=tmp_path)
        assert [item.name for item in items] == ["test_foo[alpha]", "test_foo[beta]"]

    # ensemble: needs two subprocess runs with different PYTHONHASHSEED.
    def test_deterministic_fixture_collection(
        self, pytester: Pytester, monkeypatch
    ) -> None:
        """#920"""
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="module",
                            params=["A",
                                    "B",
                                    "C"])
            def A(request):
                return request.param

            @pytest.fixture(scope="module",
                            params=["DDDDDDDDD", "EEEEEEEEEEEE", "FFFFFFFFFFF", "banansda"])
            def B(request, A):
                return request.param

            def test_foo(B):
                # Something funky is going on here.
                # Despite specified seeds, on what is collected,
                # sometimes we get unexpected passes. hashing B seems
                # to help?
                assert hash(B) or True
            """
        )
        monkeypatch.setenv("PYTHONHASHSEED", "1")
        out1 = pytester.runpytest_subprocess("-v")
        monkeypatch.setenv("PYTHONHASHSEED", "2")
        out2 = pytester.runpytest_subprocess("-v")
        output1 = [
            line
            for line in out1.outlines
            if line.startswith("test_deterministic_fixture_collection.py::test_foo")
        ]
        output2 = [
            line
            for line in out2.outlines
            if line.startswith("test_deterministic_fixture_collection.py::test_foo")
        ]
        assert len(output1) == 12
        assert output1 == output2


class TestRequestScopeAccess:
    pytestmark = pytest.mark.parametrize(
        ("scope", "ok", "error"),
        [
            ["session", "", "path class function module"],
            ["module", "module path", "cls function"],
            ["class", "module path cls", "function"],
            ["function", "module path cls function", ""],
        ],
    )

    def test_setup(self, tmp_path: Path, scope, ok, error) -> None:
        @pytest.fixture(scope=scope, autouse=True)
        def myscoped(request):
            for x in ok.split():
                assert hasattr(request, x)
            for x in error.split():
                with pytest.raises(AttributeError):
                    getattr(request, x)
            assert request.session
            assert request.config

        def test_func():
            pass

        run_tests(myscoped, test_func, rootpath=tmp_path).assert_outcomes(passed=1)

    def test_funcarg(self, tmp_path: Path, scope, ok, error) -> None:
        @pytest.fixture(scope=scope)
        def arg(request):
            for x in ok.split():
                assert hasattr(request, x)
            for x in error.split():
                with pytest.raises(AttributeError):
                    getattr(request, x)
            assert request.session
            assert request.config

        def test_func(arg):
            pass

        run_tests(arg, test_func, rootpath=tmp_path).assert_outcomes(passed=1)


class TestErrors:
    def test_subfactory_missing_funcarg(self, tmp_path: Path) -> None:
        @pytest.fixture
        def gen(qwe123):
            return 1

        def test_something(gen):
            pass

        record = run_tests(gen, test_something, rootpath=tmp_path, capture_output=True)
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            ["*def gen(qwe123):*", "*fixture*qwe123*not found*", "*1 error*"]
        )

    def test_issue498_fixture_finalizer_failing(self, tmp_path: Path) -> None:
        values: list[object] = []

        @pytest.fixture
        def fix1(request):
            def f():
                raise KeyError

            request.addfinalizer(f)
            return object()

        def test_1(fix1):
            values.append(fix1)

        def test_2(fix1):
            values.append(fix1)

        def test_3():
            assert values[0] != values[1]

        record = run_tests(
            fix1, test_1, test_2, test_3, rootpath=tmp_path, capture_output=True
        )
        record.assert_outcomes(passed=3, errors=2)
        record.stdout.fnmatch_lines(
            """
            *ERROR*teardown*test_1*
            *KeyError*
            *ERROR*teardown*test_2*
            *KeyError*
            *3 pass*2 errors*
        """
        )

    def test_setupfunc_missing_funcarg(self, tmp_path: Path) -> None:
        @pytest.fixture(autouse=True)
        def gen(qwe123):
            return 1

        def test_something():
            pass

        record = run_tests(gen, test_something, rootpath=tmp_path, capture_output=True)
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            ["*def gen(qwe123):*", "*fixture*qwe123*not found*", "*1 error*"]
        )

    def test_cached_exception_doesnt_get_longer(self, tmp_path: Path) -> None:
        """Regression test for #12204."""

        @pytest.fixture(scope="session")
        def bad():
            1 / 0  # noqa: B018

        def test_1(bad): ...

        def test_2(bad): ...

        def test_3(bad): ...

        # --tb is registered by the terminal plugin, so it has to be loaded;
        # capture_output keeps what it renders away from the outer stdout.
        spec = ConfigSpec(rootpath=tmp_path, args=("--tb=native",)).with_plugins(
            "terminal"
        )
        record = run_tests(bad, test_1, test_2, test_3, spec=spec, capture_output=True)
        record.assert_outcomes(errors=3)
        failures = [report for report in record.reports if report.failed]
        assert len(failures) == 3
        lines1 = failures[1].longrepr.reprtraceback.reprentries[0].lines  # type: ignore[union-attr]
        lines2 = failures[2].longrepr.reprtraceback.reprentries[0].lines  # type: ignore[union-attr]
        assert len(lines1) == len(lines2)


# ensemble: every test here drives ``--fixtures``, which is implemented as a
# ``pytest_cmdline_main`` hook and renders fixture *definition* locations; an
# ensemble neither reaches cmdline_main nor has non-host source locations.
class TestShowFixtures:
    def test_funcarg_compat(self, tmp_path: Path) -> None:
        spec = ConfigSpec(rootpath=tmp_path, args=("--funcargs",))
        with Ensemble(spec=spec) as ensemble:
            assert ensemble.config.option.showfixtures

    def test_show_help(self, pytester: Pytester) -> None:
        result = pytester.runpytest("--fixtures", "--help")
        assert not result.ret

    def test_show_fixtures(self, pytester: Pytester) -> None:
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            [
                "tmp_path_factory [[]session scope[]] -- .../_pytest/tmpdir.py:*",
                "*for the test session*",
                "tmp_path -- .../_pytest/tmpdir.py:*",
                "*temporary directory*",
            ]
        )

    def test_show_fixtures_verbose(self, pytester: Pytester) -> None:
        result = pytester.runpytest("--fixtures", "-v")
        result.stdout.fnmatch_lines(
            [
                "tmp_path_factory [[]session scope[]] -- .../_pytest/tmpdir.py:*",
                "*for the test session*",
                "tmp_path -- .../_pytest/tmpdir.py:*",
                "*temporary directory*",
            ]
        )

    def test_show_fixtures_testmodule(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            '''
            import pytest
            @pytest.fixture
            def _arg0():
                """ hidden """
            @pytest.fixture
            def arg1():
                """  hello world """
        '''
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            """
            *tmp_path -- *
            *fixtures defined from*
            *arg1 -- test_show_fixtures_testmodule.py:6*
            *hello world*
        """
        )
        result.stdout.no_fnmatch_line("*arg0*")

    @pytest.mark.parametrize("testmod", [True, False])
    def test_show_fixtures_conftest(self, pytester: Pytester, testmod) -> None:
        pytester.makeconftest(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """  hello world """
        '''
        )
        if testmod:
            pytester.makepyfile(
                """
                def test_hello():
                    pass
            """
            )
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            *tmp_path*
            *fixtures defined from*conftest*
            *arg1*
            *hello world*
        """
        )

    def test_show_fixtures_trimmed_doc(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                @pytest.fixture
                def arg1():
                    """
                    line1
                    line2

                    """
                @pytest.fixture
                def arg2():
                    """
                    line1
                    line2

                    """
                '''
            )
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_trimmed_doc *
                arg2 -- test_show_fixtures_trimmed_doc.py:10
                    line1
                    line2
                arg1 -- test_show_fixtures_trimmed_doc.py:3
                    line1
                    line2
                """
            )
        )

    def test_show_fixtures_indented_doc(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                @pytest.fixture
                def fixture1():
                    """
                    line1
                        indented line
                    """
                '''
            )
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_doc *
                fixture1 -- test_show_fixtures_indented_doc.py:3
                    line1
                        indented line
                """
            )
        )

    def test_show_fixtures_indented_doc_first_line_unindented(
        self, pytester: Pytester
    ) -> None:
        p = pytester.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                @pytest.fixture
                def fixture1():
                    """line1
                    line2
                        indented line
                    """
                '''
            )
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_doc_first_line_unindented *
                fixture1 -- test_show_fixtures_indented_doc_first_line_unindented.py:3
                    line1
                    line2
                        indented line
                """
            )
        )

    def test_show_fixtures_indented_in_class(self, pytester: Pytester) -> None:
        p = pytester.makepyfile(
            textwrap.dedent(
                '''\
                import pytest
                class TestClass(object):
                    @pytest.fixture
                    def fixture1(self):
                        """line1
                        line2
                            indented line
                        """
                '''
            )
        )
        result = pytester.runpytest("--fixtures", p)
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                * fixtures defined from test_show_fixtures_indented_in_class *
                fixture1 -- test_show_fixtures_indented_in_class.py:4
                    line1
                    line2
                        indented line
                """
            )
        )

    def test_show_fixtures_different_files(self, pytester: Pytester) -> None:
        """`--fixtures` only shows fixtures from first file (#833)."""
        pytester.makepyfile(
            test_a='''
            import pytest

            @pytest.fixture
            def fix_a():
                """Fixture A"""
                pass

            def test_a(fix_a):
                pass
        '''
        )
        pytester.makepyfile(
            test_b='''
            import pytest

            @pytest.fixture
            def fix_b():
                """Fixture B"""
                pass

            def test_b(fix_b):
                pass
        '''
        )
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            * fixtures defined from test_a *
            fix_a -- test_a.py:4
                Fixture A

            * fixtures defined from test_b *
            fix_b -- test_b.py:4
                Fixture B
        """
        )

    def test_show_fixtures_with_same_name(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """Hello World in conftest.py"""
                return "Hello World"
        '''
        )
        pytester.makepyfile(
            """
            def test_foo(arg1):
                assert arg1 == "Hello World"
        """
        )
        pytester.makepyfile(
            '''
            import pytest
            @pytest.fixture
            def arg1():
                """Hi from test module"""
                return "Hi"
            def test_bar(arg1):
                assert arg1 == "Hi"
        '''
        )
        result = pytester.runpytest("--fixtures")
        result.stdout.fnmatch_lines(
            """
            * fixtures defined from conftest *
            arg1 -- conftest.py:3
                Hello World in conftest.py

            * fixtures defined from test_show_fixtures_with_same_name *
            arg1 -- test_show_fixtures_with_same_name.py:3
                Hi from test module
        """
        )

    def test_fixture_disallow_twice(self):
        """Test that applying @pytest.fixture twice generates an error (#2334)."""
        with pytest.raises(ValueError):

            @pytest.fixture
            @pytest.fixture
            def foo():
                raise NotImplementedError()


class TestContextManagerFixtureFuncs:
    def test_simple(self, tmp_path: Path) -> None:
        # The original watched the ordering through printed output under -s;
        # recording the events directly asserts the same ordering without
        # depending on capture, which an ensemble does not provide.
        events: list[str] = []

        @pytest.fixture
        def arg1():
            events.append("setup")
            yield 1
            events.append("teardown")

        def test_1(arg1):
            events.append(f"test1 {arg1}")

        def test_2(arg1):
            events.append(f"test2 {arg1}")
            assert 0

        record = run_tests(arg1, test_1, test_2, rootpath=tmp_path)
        record.assert_outcomes(passed=1, failed=1)
        assert events == [
            "setup",
            "test1 1",
            "teardown",
            "setup",
            "test2 1",
            "teardown",
        ]

    def test_scoped(self, tmp_path: Path) -> None:
        events: list[str] = []

        @pytest.fixture(scope="module")
        def arg1():
            events.append("setup")
            yield 1
            events.append("teardown")

        def test_1(arg1):
            events.append(f"test1 {arg1}")

        def test_2(arg1):
            events.append(f"test2 {arg1}")

        record = run_tests(arg1, test_1, test_2, rootpath=tmp_path)
        record.assert_outcomes(passed=2)
        assert events == ["setup", "test1 1", "test2 1", "teardown"]

    def test_setup_exception(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="module")
        def arg1():
            pytest.fail("setup")
            yield 1  # type: ignore[unreachable]

        def test_1(arg1):
            pass

        record = run_tests(arg1, test_1, rootpath=tmp_path)
        record.assert_outcomes(errors=1)
        setup = record["test_1"].setup
        assert setup is not None
        assert "Failed: setup" in setup.longreprtext

    def test_teardown_exception(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="module")
        def arg1():
            yield 1
            pytest.fail("teardown")

        def test_1(arg1):
            pass

        record = run_tests(arg1, test_1, rootpath=tmp_path)
        record.assert_outcomes(passed=1, errors=1)
        teardown = record["test_1"].teardown
        assert teardown is not None
        assert "Failed: teardown" in teardown.longreprtext

    # ensemble: asserts the offending fixture's file:line, host-anchored for
    # in-memory sources.
    def test_yields_more_than_one(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.fixture(scope="module")
            def arg1():
                yield 1
                yield 2
            def test_1(arg1):
                pass
        """
        )
        result = pytester.runpytest("-s")
        result.stdout.fnmatch_lines(
            """
            *fixture function*
            *test_yields*:2*
        """
        )

    def test_custom_name(self, tmp_path: Path) -> None:
        seen: list[str] = []

        @pytest.fixture(name="meow")
        def arg1():
            return "mew"

        def test_1(meow):
            seen.append(meow)

        record = run_tests(arg1, test_1, rootpath=tmp_path)
        record.assert_outcomes(passed=1)
        assert seen == ["mew"]


# ensemble: every test here asserts where the requested fixture is *defined*
# and where it was requested from, as file:line; for in-memory sources both
# resolve into this host file.
class TestParameterizedSubRequest:
    def test_call_from_fixture(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_call_from_fixture="""
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param

            @pytest.fixture
            def get_named_fixture(request):
                return request.getfixturevalue('fix_with_param')

            def test_foo(request, get_named_fixture):
                pass
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_call_from_fixture.py::test_foo",
                "Requested fixture 'fix_with_param' defined in:",
                "test_call_from_fixture.py:4",
                "Requested here:",
                "test_call_from_fixture.py:9",
                "*1 error in*",
            ]
        )

    def test_call_from_test(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_call_from_test="""
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param

            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_call_from_test.py::test_foo",
                "Requested fixture 'fix_with_param' defined in:",
                "test_call_from_test.py:4",
                "Requested here:",
                "test_call_from_test.py:8",
                "*1 failed*",
            ]
        )

    def test_external_fixture(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(params=[0, 1, 2])
            def fix_with_param(request):
                return request.param
            """
        )

        pytester.makepyfile(
            test_external_fixture="""
            def test_foo(request):
                request.getfixturevalue('fix_with_param')
            """
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_external_fixture.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                "conftest.py:4",
                "Requested here:",
                "test_external_fixture.py:2",
                "*1 failed*",
            ]
        )

    def test_non_relative_path(self, pytester: Pytester) -> None:
        tests_dir = pytester.mkdir("tests")
        fixdir = pytester.mkdir("fixtures")
        fixfile = fixdir.joinpath("fix.py")
        fixfile.write_text(
            textwrap.dedent(
                """\
                import pytest

                @pytest.fixture(params=[0, 1, 2])
                def fix_with_param(request):
                    return request.param
                """
            ),
            encoding="utf-8",
        )

        testfile = tests_dir.joinpath("test_foos.py")
        testfile.write_text(
            textwrap.dedent(
                """\
                from fix import fix_with_param

                def test_foo(request):
                    request.getfixturevalue('fix_with_param')
                """
            ),
            encoding="utf-8",
        )

        os.chdir(tests_dir)
        pytester.syspathinsert(fixdir)
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_foos.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                f"{fixfile}:4",
                "Requested here:",
                "test_foos.py:4",
                "*1 failed*",
            ]
        )

        # With non-overlapping rootdir, passing tests_dir.
        rootdir = pytester.mkdir("rootdir")
        os.chdir(rootdir)
        result = pytester.runpytest("--rootdir", rootdir, tests_dir)
        result.stdout.fnmatch_lines(
            [
                "The requested fixture has no parameter defined for test:",
                "    test_foos.py::test_foo",
                "",
                "Requested fixture 'fix_with_param' defined in:",
                f"{fixfile}:4",
                "Requested here:",
                f"{testfile}:4",
                "*1 failed*",
            ]
        )


# ensemble: the subject is a rootdir conftest and a subdirectory conftest
# both implementing the hook, and the order they run in.
def test_pytest_fixture_setup_and_post_finalizer_hook(pytester: Pytester) -> None:
    pytester.makeconftest(
        """
        def pytest_fixture_setup(fixturedef, request):
            print('ROOT setup hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
        def pytest_fixture_post_finalizer(fixturedef, request):
            print('ROOT finalizer hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
    """
    )
    pytester.makepyfile(
        **{
            "tests/conftest.py": """
            def pytest_fixture_setup(fixturedef, request):
                print('TESTS setup hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
            def pytest_fixture_post_finalizer(fixturedef, request):
                print('TESTS finalizer hook called for {0} from {1}'.format(fixturedef.argname, request.node.name))
        """,
            "tests/test_hooks.py": """
            import pytest

            @pytest.fixture()
            def my_fixture():
                return 'some'

            def test_func(my_fixture):
                print('TEST test_func')
                assert my_fixture == 'some'
        """,
        }
    )
    result = pytester.runpytest("-s")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "*TESTS setup hook called for my_fixture from test_func*",
            "*ROOT setup hook called for my_fixture from test_func*",
            "*TEST test_func*",
            "*TESTS finalizer hook called for my_fixture from test_func*",
            "*ROOT finalizer hook called for my_fixture from test_func*",
        ]
    )


def test_fixture_post_finalizer_called_once(tmp_path: Path) -> None:
    """Test that pytest_fixture_post_finalizer is called only once per fixture teardown.

    When a fixture depends on multiple parametrized fixtures and all their parameters
    change at the same time, the dependent fixture should be torn down only once,
    and pytest_fixture_post_finalizer should be called only once for it.
    """
    finalizer_calls: list[str] = []

    class ConftestPlugin:
        def pytest_fixture_post_finalizer(self, fixturedef, request):
            finalizer_calls.append(fixturedef.argname)

        @pytest.fixture(autouse=True)
        def check_finalizer_calls(self, request):
            yield
            # After each test, verify no duplicate finalizer calls.
            if finalizer_calls:
                assert len(finalizer_calls) == len(set(finalizer_calls)), (
                    f"Duplicate finalizer calls detected: {finalizer_calls}"
                )
                finalizer_calls.clear()

    @pytest.fixture(scope="session")
    def foo(request):
        return request.param

    @pytest.fixture(scope="session")
    def bar(request):
        return request.param

    @pytest.fixture(scope="session")
    def baz(foo, bar):
        return f"{foo}-{bar}"

    @pytest.mark.parametrize("foo,bar", [(1, 1)], indirect=True)
    def test_first(foo, bar, baz):
        assert foo == 1
        assert bar == 1
        assert baz == "1-1"

    @pytest.mark.parametrize("foo,bar", [(2, 2)], indirect=True)
    def test_second(foo, bar, baz):
        assert foo == 2
        assert bar == 2
        assert baz == "2-2"

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
    module = build_module("test_fixtures", foo, bar, baz, test_first, test_second)
    record = run_tests(module, spec=spec)
    # The test passes, which means no duplicate finalizer calls were detected
    # by the check_finalizer_calls autouse fixture.
    record.assert_outcomes(passed=2)


def test_fixture_post_finalizer_hook_exception(tmp_path: Path) -> None:
    """Test that exceptions in pytest_fixture_post_finalizer hook are caught.

    Also verifies that the fixture cache is properly reset even when the
    post_finalizer hook raises an exception, so the fixture can be rebuilt
    in subsequent tests.
    """

    class ConftestPlugin:
        def pytest_fixture_post_finalizer(self, fixturedef, request):
            if "test_first" in request.node.nodeid:
                raise RuntimeError("Error in post finalizer hook")

        @pytest.fixture
        def my_fixture(self, request):
            yield request.node.nodeid

    def test_first(my_fixture):
        assert "test_first" in my_fixture

    def test_second(my_fixture):
        assert "test_second" in my_fixture

    spec = ConfigSpec(
        rootpath=tmp_path,
        args=("-v", "--setup-show"),
        extra_plugins=(ConftestPlugin(),),
    ).with_plugins("setuponly")
    record = run_tests(
        build_module("test_fixtures", test_first, test_second),
        spec=spec,
        capture_output=True,
    )
    record.assert_outcomes(passed=2, errors=1)
    record.stdout.fnmatch_lines(
        [
            "*test_first*PASSED",
            "*test_first*ERROR",
            "*RuntimeError: Error in post finalizer hook*",
        ]
    )
    # Verify fixture is setup twice (rebuilt for test_second despite error).
    record.stdout.fnmatch_lines(
        [
            "test_fixtures.py::test_first ",
            "        SETUP    F my_fixture",
            "        test_fixtures.py::test_first (fixtures used: my_fixture, request) PASSED",
            "test_fixtures.py::test_first ERROR",
            "test_fixtures.py::test_second ",
            "        SETUP    F my_fixture",
            "        test_fixtures.py::test_second (fixtures used: my_fixture, request) PASSED",
            "        TEARDOWN F my_fixture",
        ],
        consecutive=True,
    )


class TestParamValueKey:
    """Unit tests for the equivalence key used by `reorder_items` (#8914)."""

    def test_equal_hashable_values(self) -> None:
        # Build equal-but-not-identical values to exercise the ``==`` path
        # rather than the identity shortcut.
        v1, v2 = tuple([1, 2]), tuple([1, 2])
        assert v1 is not v2
        k1, k2 = ParamValueKey(v1, 0), ParamValueKey(v2, 1)
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_identical_value(self) -> None:
        value = object()
        assert ParamValueKey(value, 0) == ParamValueKey(value, 1)

    def test_unequal_hashable_values(self) -> None:
        assert ParamValueKey("a", 0) != ParamValueKey("b", 0)

    def test_equal_values_of_different_type(self) -> None:
        # 1 == True == 1.0 in Python, but grouping them could change which
        # value an adjacent test's fixture is set up with, so the key keeps
        # them apart.
        assert ParamValueKey(1, 0) != ParamValueKey(True, 0)
        assert ParamValueKey(1, 0) != ParamValueKey(1.0, 0)

    def test_value_key_never_equals_index_key(self) -> None:
        # hash(0) == hash(ParamValueKey({}, 0)._key) here, so these could
        # collide in a dict bucket; they must still compare unequal.
        assert ParamValueKey(0, 0) != ParamValueKey({}, 0)
        assert ParamValueKey({}, 0) != ParamValueKey(0, 0)

    def test_unhashable_values_compare_by_index(self) -> None:
        assert ParamValueKey({"a": 1}, 0) == ParamValueKey({"b": 2}, 0)
        assert ParamValueKey({"a": 1}, 0) != ParamValueKey({"a": 1}, 1)

    def test_exotic_eq(self) -> None:
        class Exotic:
            def __eq__(self, other: object) -> bool:
                raise ValueError("cannot compare")

            def __hash__(self) -> int:
                return 0

        assert ParamValueKey(Exotic(), 0) != ParamValueKey(Exotic(), 0)

    def test_other_types(self) -> None:
        assert ParamValueKey("a", 0) != "a"
        assert ParamValueKey("a", 0).__eq__("a") is NotImplemented

    def test_repr(self) -> None:
        assert repr(ParamValueKey("a", 0)) == "ParamValueKey(value='a')"
        assert repr(ParamValueKey({}, 3)) == "ParamValueKey(index=3)"


class TestScopeOrdering:
    """Class of tests that ensure fixtures are ordered based on their scopes (#2405)"""

    @pytest.mark.parametrize("variant", ["mark", "autouse"])
    def test_func_closure_module_auto(self, tmp_path: Path, variant) -> None:
        """Semantically identical to the example posted in #2405 when ``use_mark=True``"""

        @pytest.fixture(scope="module", autouse=variant == "autouse")
        def m1():
            pass

        @pytest.fixture(scope="function", autouse=True)
        def f1():
            pass

        def test_func(m1):
            pass

        module = build_module(
            "test_func_closure_module_auto",
            m1,
            f1,
            test_func,
            pytestmark=pytest.mark.usefixtures("m1") if variant == "mark" else [],
        )
        with Ensemble(module, rootpath=tmp_path) as ensemble:
            items = ensemble.collect()
            assert isinstance(items[0], Function)
            request = TopRequest(items[0], _ispytest=True)
            assert request.fixturenames == "m1 f1".split()

    # ensemble: the closure under test contains a package-scoped fixture, and
    # an ensemble has no Package node for it to bind to.
    def test_func_closure_with_native_fixtures(self, pytester: Pytester) -> None:
        """Sanity check that verifies the order returned by the closures and the
        actual fixture execution order: the execution order may differ because
        of fixture inter-dependencies."""
        pytester.makepyfile(
            """
            import pytest

            fixture_order = []

            @pytest.fixture(scope="session")
            def s1():
                fixture_order.append("s1")

            @pytest.fixture(scope="package")
            def p1():
                fixture_order.append("p1")

            @pytest.fixture(scope="module")
            def m1():
                fixture_order.append("m1")

            @pytest.fixture(scope="session")
            def my_tmp_path_factory():
                fixture_order.append("my_tmp_path_factory")

            @pytest.fixture
            def my_tmp_path(my_tmp_path_factory):
                fixture_order.append("my_tmp_path")

            @pytest.fixture
            def f1(my_tmp_path):
                fixture_order.append("f1")

            @pytest.fixture
            def f2():
                fixture_order.append("f2")

            def test_foo(f1, p1, m1, f2, s1):
                # Actual fixture execution differs from static order: dependent
                # fixtures must be created first ("my_tmp_path").
                assert fixture_order == [
                    "my_tmp_path_factory",
                    "s1",
                    "p1",
                    "m1",
                    "my_tmp_path",
                    "f1",
                    "f2",
                ]
        """
        )
        items, _ = pytester.inline_genitems()
        assert isinstance(items[0], Function)
        request = TopRequest(items[0], _ispytest=True)
        # Static order of fixtures based on their scope and position in the
        # parameter list.
        assert request.fixturenames == [
            "my_tmp_path_factory",
            "s1",
            "p1",
            "m1",
            "f1",
            "my_tmp_path",
            "f2",
        ]
        result = pytester.runpytest("-vv")
        result.assert_outcomes(passed=1)

    def test_func_closure_module(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="module")
        def m1():
            pass

        @pytest.fixture(scope="function")
        def f1():
            pass

        def test_func(f1, m1):
            pass

        with Ensemble(m1, f1, test_func, rootpath=tmp_path) as ensemble:
            items = ensemble.collect()
            assert isinstance(items[0], Function)
            request = TopRequest(items[0], _ispytest=True)
            assert request.fixturenames == "m1 f1".split()

    def test_func_closure_scopes_reordered(self, tmp_path: Path) -> None:
        """Test ensures that fixtures are ordered by scope regardless of the order of the parameters, although
        fixtures of same scope keep the declared order
        """

        @pytest.fixture(scope="session")
        def s1():
            pass

        @pytest.fixture(scope="module")
        def m1():
            pass

        @pytest.fixture(scope="function")
        def f1():
            pass

        @pytest.fixture(scope="function")
        def f2():
            pass

        class Test:
            @pytest.fixture(scope="class")
            def c1(cls):
                pass

            def test_func(self, f2, f1, c1, m1, s1):
                pass

        # Fixture *definition* order matters here, and in an ensemble that is
        # the order the members are passed in.
        with Ensemble(s1, m1, f1, f2, Test, rootpath=tmp_path) as ensemble:
            items = ensemble.collect()
            assert isinstance(items[0], Function)
            request = TopRequest(items[0], _ispytest=True)
            assert request.fixturenames == "s1 m1 c1 f2 f1".split()

    # ensemble: conftests in nested directories, one of them package-scoped.
    def test_func_closure_same_scope_closer_root_first(
        self, pytester: Pytester
    ) -> None:
        """Auto-use fixtures of same scope are ordered by closer-to-root first"""
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(scope='module', autouse=True)
            def m_conf(): pass
        """
        )
        pytester.makepyfile(
            **{
                "sub/conftest.py": """
                import pytest

                @pytest.fixture(scope='package', autouse=True)
                def p_sub(): pass

                @pytest.fixture(scope='module', autouse=True)
                def m_sub(): pass
            """,
                "sub/__init__.py": "",
                "sub/test_func.py": """
                import pytest

                @pytest.fixture(scope='module', autouse=True)
                def m_test(): pass

                @pytest.fixture(scope='function')
                def f1(): pass

                def test_func(m_test, f1):
                    pass
        """,
            }
        )
        items, _ = pytester.inline_genitems()
        assert isinstance(items[0], Function)
        request = TopRequest(items[0], _ispytest=True)
        assert request.fixturenames == "p_sub m_conf m_sub m_test f1".split()

    # ensemble: the closure under test contains a package-scoped fixture.
    def test_func_closure_all_scopes_complex(self, pytester: Pytester) -> None:
        """Complex test involving all scopes and mixing autouse with normal fixtures"""
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(scope='session')
            def s1(): pass

            @pytest.fixture(scope='package', autouse=True)
            def p1(): pass
        """
        )
        pytester.makepyfile(**{"__init__.py": ""})
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(scope='module', autouse=True)
            def m1(): pass

            @pytest.fixture(scope='module')
            def m2(s1): pass

            @pytest.fixture(scope='function')
            def f1(): pass

            @pytest.fixture(scope='function')
            def f2(): pass

            class Test:

                @pytest.fixture(scope='class', autouse=True)
                def c1(self):
                    pass

                def test_func(self, f2, f1, m2):
                    pass
        """
        )
        items, _ = pytester.inline_genitems()
        assert isinstance(items[0], Function)
        request = TopRequest(items[0], _ispytest=True)
        assert request.fixturenames == "s1 p1 m1 m2 c1 f2 f1".split()

    # ensemble: package-scoped fixture, and a package layout.
    def test_parametrized_package_scope_reordering(self, pytester: Pytester) -> None:
        """A parameterized package-scoped fixture correctly reorders items to
        minimize setups & teardowns.

        Regression test for #12328.
        """
        pytester.makepyfile(
            __init__="",
            conftest="""
                import pytest
                @pytest.fixture(scope="package", params=["a", "b"])
                def fix(request):
                    return request.param
            """,
            test_1="def test1(fix): pass",
            test_2="def test2(fix): pass",
        )

        result = pytester.runpytest("--setup-plan")
        assert result.ret == ExitCode.OK
        result.stdout.fnmatch_lines(
            [
                "  SETUP    P fix['a']",
                "        test_1.py::test1[a] (fixtures used: fix, request)",
                "        test_2.py::test2[a] (fixtures used: fix, request)",
                "  TEARDOWN P fix['a']",
                "  SETUP    P fix['b']",
                "        test_1.py::test1[b] (fixtures used: fix, request)",
                "        test_2.py::test2[b] (fixtures used: fix, request)",
                "  TEARDOWN P fix['b']",
            ],
        )

    def test_reorder_by_param_value_across_parametrize_calls(
        self, tmp_path: Path
    ) -> None:
        """Items parametrized by separate parametrize() calls are grouped by
        the *value* of higher-scoped parameters, so that equal values share a
        single fixture setup.

        Regression test for #8914.
        """

        @pytest.fixture(scope="session")
        def prepare(request):
            return request.param

        @pytest.mark.parametrize("prepare", ["dina"], indirect=True, scope="session")
        def test_1(prepare): ...

        @pytest.mark.parametrize("prepare", ["more"], indirect=True, scope="session")
        def test_2(prepare): ...

        @pytest.mark.parametrize("prepare", ["dina"], indirect=True, scope="session")
        def test_3(prepare): ...

        spec = ConfigSpec(rootpath=tmp_path, args=("--setup-plan",)).with_plugins(
            "setupplan", "setuponly"
        )
        record = run_tests(
            build_module("test_8914", prepare, test_1, test_2, test_3),
            spec=spec,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "SETUP    S prepare['dina']",
                "        test_8914.py::test_1[dina] (fixtures used: prepare, request)",
                "        test_8914.py::test_3[dina] (fixtures used: prepare, request)",
                "TEARDOWN S prepare['dina']",
                "SETUP    S prepare['more']",
                "        test_8914.py::test_2[more] (fixtures used: prepare, request)",
                "TEARDOWN S prepare['more']",
            ],
        )

    def test_reorder_unhashable_params_fall_back_to_index(self, tmp_path: Path) -> None:
        """Unhashable parameter values are grouped by their index within their
        parametrize() call, as they were before #8914 was fixed.
        """

        @pytest.fixture(scope="module")
        def fix(request):
            return request.param

        @pytest.mark.parametrize(
            "fix", [{"a": 1}, {"b": 2}], indirect=True, scope="module"
        )
        def test_1(fix): ...

        @pytest.mark.parametrize(
            "fix", [{"a": 1}, {"b": 2}], indirect=True, scope="module"
        )
        def test_2(fix): ...

        spec = ConfigSpec(rootpath=tmp_path, args=("--setup-plan",)).with_plugins(
            "setupplan", "setuponly"
        )
        record = run_tests(
            build_module("test_unhashable", fix, test_1, test_2),
            spec=spec,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "    SETUP    M fix[{'a': 1}]",
                "        test_unhashable.py::test_1[fix0] (fixtures used: fix, request)",
                "        test_unhashable.py::test_2[fix0] (fixtures used: fix, request)",
                "    TEARDOWN M fix[{'a': 1}]",
                "    SETUP    M fix[{'b': 2}]",
                "        test_unhashable.py::test_1[fix1] (fixtures used: fix, request)",
                "        test_unhashable.py::test_2[fix1] (fixtures used: fix, request)",
                "    TEARDOWN M fix[{'b': 2}]",
            ],
        )

    def test_reorder_mixed_hashable_unhashable_params(self, tmp_path: Path) -> None:
        """Hashable and unhashable values parametrizing the same fixture only
        group with their own kind: values with values, unhashables by index.
        """

        @pytest.fixture(scope="module")
        def fix(request):
            return request.param

        @pytest.mark.parametrize("fix", [{"a": 1}], indirect=True, scope="module")
        def test_1(fix): ...

        @pytest.mark.parametrize("fix", ["x"], indirect=True, scope="module")
        def test_2(fix): ...

        @pytest.mark.parametrize("fix", ["x"], indirect=True, scope="module")
        def test_3(fix): ...

        spec = ConfigSpec(rootpath=tmp_path, args=("--setup-plan",)).with_plugins(
            "setupplan", "setuponly"
        )
        record = run_tests(
            build_module("test_mixed", fix, test_1, test_2, test_3),
            spec=spec,
            capture_output=True,
        )
        record.stdout.fnmatch_lines(
            [
                "    SETUP    M fix[{'a': 1}]",
                "        test_mixed.py::test_1[fix0] (fixtures used: fix, request)",
                "    TEARDOWN M fix[{'a': 1}]",
                "    SETUP    M fix['x']",
                "        test_mixed.py::test_2[x] (fixtures used: fix, request)",
                "        test_mixed.py::test_3[x] (fixtures used: fix, request)",
                "    TEARDOWN M fix['x']",
            ],
        )

    def test_reorder_params_with_exotic_eq(self, tmp_path: Path) -> None:
        """Parameter values whose ``__eq__`` raises or returns non-booleans
        (e.g. numpy arrays) do not break collection or reordering (#6497).
        """

        class Exotic:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                raise ValueError("cannot compare")

            def __hash__(self):
                return 0

        @pytest.fixture(scope="module")
        def fix(request):
            return request.param

        @pytest.mark.parametrize("fix", [Exotic(1)], indirect=True, scope="module")
        def test_1(fix): ...

        @pytest.mark.parametrize("fix", [Exotic(2)], indirect=True, scope="module")
        def test_2(fix): ...

        record = run_tests(fix, test_1, test_2, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    # ensemble: package layout with package-scoped fixtures in two packages.
    def test_multiple_packages(self, pytester: Pytester) -> None:
        """Complex test involving multiple package fixtures. Make sure teardowns
        are executed in order.
        .
        └── root
            ├── __init__.py
            ├── sub1
            │   ├── __init__.py
            │   ├── conftest.py
            │   └── test_1.py
            └── sub2
                ├── __init__.py
                ├── conftest.py
                └── test_2.py
        """
        root = pytester.mkdir("root")
        root.joinpath("__init__.py").write_text("values = []", encoding="utf-8")
        sub1 = root.joinpath("sub1")
        sub1.mkdir()
        sub1.joinpath("__init__.py").touch()
        sub1.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
            import pytest
            from .. import values
            @pytest.fixture(scope="package")
            def fix():
                values.append("pre-sub1")
                yield values
                assert values.pop() == "pre-sub1"
        """
            ),
            encoding="utf-8",
        )
        sub1.joinpath("test_1.py").write_text(
            textwrap.dedent(
                """\
            from .. import values
            def test_1(fix):
                assert values == ["pre-sub1"]
        """
            ),
            encoding="utf-8",
        )
        sub2 = root.joinpath("sub2")
        sub2.mkdir()
        sub2.joinpath("__init__.py").touch()
        sub2.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
            import pytest
            from .. import values
            @pytest.fixture(scope="package")
            def fix():
                values.append("pre-sub2")
                yield values
                assert values.pop() == "pre-sub2"
        """
            ),
            encoding="utf-8",
        )
        sub2.joinpath("test_2.py").write_text(
            textwrap.dedent(
                """\
            from .. import values
            def test_2(fix):
                assert values == ["pre-sub2"]
        """
            ),
            encoding="utf-8",
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=2)

    def test_class_fixture_self_instance(self, tmp_path: Path) -> None:
        """Check that plugin classes which implement fixtures receive the plugin instance
        as self (see #2270).
        """

        class MyPlugin:
            def __init__(self):
                self.arg = 1

            @pytest.fixture(scope="function")
            def myfix(self):
                assert isinstance(self, MyPlugin)
                return self.arg

        class TestClass:
            def test_1(self, myfix):
                assert myfix == 1

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(MyPlugin(),))
        run_tests(TestClass, spec=spec).assert_outcomes(passed=1)


def test_call_fixture_function_error():
    """Check if an error is raised if a fixture function is called directly (#4545)"""

    @pytest.fixture
    def fix():
        raise NotImplementedError()

    with pytest.raises(pytest.fail.Exception):
        assert fix() == 1


# ensemble: the double decoration raises at decoration time, so it can only
# be observed as a *module import* failure (see test_fixture_disallow_twice
# for the direct form).
def test_fixture_double_decorator(pytester: Pytester) -> None:
    """Check if an error is raised when using @pytest.fixture twice."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        @pytest.fixture
        def fixt():
            pass
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        [
            "E * ValueError: @pytest.fixture is being applied more than once to the same function 'fixt'"
        ]
    )


# ensemble: `@pytest.fixture` on a class raises at decoration time, so it can
# only be observed as a module import failure.
def test_fixture_class(pytester: Pytester) -> None:
    """Check if an error is raised when using @pytest.fixture on a class."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        class A:
            pass
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(errors=1)


def test_fixture_param_shadowing(tmp_path: Path) -> None:
    """Parametrized arguments would be shadowed if a fixture with the same name also exists (#5036)"""

    @pytest.fixture(params=["a", "b"])
    def argroot(request):
        return request.param

    @pytest.fixture
    def arg(argroot):
        return argroot

    # This should only be parametrized directly
    @pytest.mark.parametrize("arg", [1])
    def test_direct(arg):
        assert arg == 1

    # This should be parametrized based on the fixtures
    def test_normal_fixture(arg):
        assert isinstance(arg, str)

    # Indirect should still work:

    @pytest.fixture
    def arg2(request):
        return 2 * request.param

    @pytest.mark.parametrize("arg2", [1], indirect=True)
    def test_indirect(arg2):
        assert arg2 == 2

    module = build_module(
        "test_fixture_param_shadowing",
        argroot,
        test_direct,
        test_normal_fixture,
        test_indirect,
        arg=arg,
        arg2=arg2,
    )
    record = run_tests(module, rootpath=tmp_path)
    # Only one test should have run
    record.assert_outcomes(passed=4)
    assert sorted(record.by_test) == [
        "test_fixture_param_shadowing.py::test_direct[1]",
        "test_fixture_param_shadowing.py::test_indirect[1]",
        "test_fixture_param_shadowing.py::test_normal_fixture[a]",
        "test_fixture_param_shadowing.py::test_normal_fixture[b]",
    ]


def test_fixture_named_request() -> None:
    # The reserved name is rejected by the decorator, so importing the example
    # is what raises - there is nothing left to collect afterwards. Importing
    # it as itself is what makes the reported location the example's own line;
    # an inlined copy would name this file instead.
    example = EXAMPLES / "fixtures/test_fixture_named_request.py"
    with pytest.raises(pytest.fail.Exception) as excinfo:
        module_from_path(example)
    LineMatcher(str(excinfo.value).splitlines()).fnmatch_lines(
        [
            "*'request' is a reserved word for fixtures, use another name:",
            "  *test_fixture_named_request.py:8",
        ]
    )


def test_indirect_fixture_does_not_break_scope(tmp_path: Path) -> None:
    """Ensure that fixture scope is respected when using indirect fixtures (#570)"""
    instantiated: list[tuple[str, str]] = []

    @pytest.fixture(scope="session")
    def fixture_1(request):
        instantiated.append(("fixture_1", request.param))

    @pytest.fixture(scope="session")
    def fixture_2(request):
        instantiated.append(("fixture_2", request.param))

    scenarios = [
        ("A", "a1"),
        ("A", "a2"),
        ("B", "b1"),
        ("B", "b2"),
        ("C", "c1"),
        ("C", "c2"),
    ]

    @pytest.mark.parametrize(
        "fixture_1,fixture_2", scenarios, indirect=["fixture_1", "fixture_2"]
    )
    def test_create_fixtures(fixture_1, fixture_2):
        pass

    def test_check_fixture_instantiations():
        assert instantiated == [
            ("fixture_1", "A"),
            ("fixture_2", "a1"),
            ("fixture_2", "a2"),
            ("fixture_1", "B"),
            ("fixture_2", "b1"),
            ("fixture_2", "b2"),
            ("fixture_1", "C"),
            ("fixture_2", "c1"),
            ("fixture_2", "c2"),
        ]

    record = run_tests(
        fixture_1,
        fixture_2,
        test_create_fixtures,
        test_check_fixture_instantiations,
        rootpath=tmp_path,
    )
    record.assert_outcomes(passed=7)


def test_fixture_parametrization_nparray(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")

    @pytest.fixture(params=numpy.linspace(1, 10, 10))
    def value(request):
        return request.param

    def test_bug(value):
        assert value == value

    run_tests(value, test_bug, rootpath=tmp_path).assert_outcomes(passed=10)


def test_fixture_arg_ordering(tmp_path: Path) -> None:
    """
    This test describes how fixtures in the same scope but without explicit dependencies
    between them are created. While users should make dependencies explicit, often
    they rely on this order, so this test exists to catch regressions in this regard.
    See #6540 and #6492.
    """
    suffixes: list[str] = []

    @pytest.fixture
    def fix_1():
        suffixes.append("fix_1")

    @pytest.fixture
    def fix_2():
        suffixes.append("fix_2")

    @pytest.fixture
    def fix_3():
        suffixes.append("fix_3")

    @pytest.fixture
    def fix_4():
        suffixes.append("fix_4")

    @pytest.fixture
    def fix_5():
        suffixes.append("fix_5")

    @pytest.fixture
    def fix_combined(fix_1, fix_2, fix_3, fix_4, fix_5):
        pass

    def test_suffix(fix_combined):
        assert suffixes == ["fix_1", "fix_2", "fix_3", "fix_4", "fix_5"]

    record = run_tests(
        fix_1, fix_2, fix_3, fix_4, fix_5, fix_combined, test_suffix, rootpath=tmp_path
    )
    record.assert_outcomes(passed=1)


def test_yield_fixture_with_no_value(tmp_path: Path) -> None:
    @pytest.fixture(name="custom")
    def empty_yield():
        if False:
            yield  # type: ignore[unreachable]

    def test_fixt(custom):
        pass

    record = run_tests(empty_yield, test_fixt, rootpath=tmp_path)
    record.assert_outcomes(errors=1)
    setup = record["test_fixt"].setup
    assert setup is not None
    assert "ValueError: custom did not yield a value" in setup.longreprtext


def test_deduplicate_names() -> None:
    items = deduplicate_names("abacd")
    assert items == ("a", "b", "c", "d")
    items = deduplicate_names((*items, "g", "f", "g", "e", "b"))
    assert items == ("a", "b", "c", "d", "g", "f", "e")


def test_staticmethod_classmethod_fixture_instance(tmp_path: Path) -> None:
    """Ensure that static and class methods get and have access to a fresh
    instance.

    This also ensures `setup_method` works well with static and class methods.

    Regression test for #12065.
    """

    class Test:
        ran_setup_method = False
        ran_fixture = False

        def setup_method(self):
            assert not self.ran_setup_method
            self.ran_setup_method = True

        @pytest.fixture(autouse=True)
        def fixture(self):
            assert not self.ran_fixture
            self.ran_fixture = True

        def test_method(self):
            assert self.ran_setup_method
            assert self.ran_fixture

        @staticmethod
        def test_1(request):
            assert request.instance.ran_setup_method
            assert request.instance.ran_fixture

        @classmethod
        def test_2(cls, request):
            assert request.instance.ran_setup_method
            assert request.instance.ran_fixture

    run_tests(Test, rootpath=tmp_path).assert_outcomes(passed=3)


def test_scoped_fixture_caching(tmp_path: Path) -> None:
    """Make sure setup and finalization is only run once when using scoped fixture
    multiple times."""
    executed: list[str] = []

    @pytest.fixture(scope="class")
    def fixture_1():
        executed.append("fix setup")
        yield
        executed.append("fix teardown")

    class TestFixtureCaching:
        def test_1(self, fixture_1: None) -> None:
            assert executed == ["fix setup"]

        def test_2(self, fixture_1: None) -> None:
            assert executed == ["fix setup"]

    def test_expected_setup_and_teardown() -> None:
        assert executed == ["fix setup", "fix teardown"]

    record = run_tests(
        fixture_1,
        TestFixtureCaching,
        test_expected_setup_and_teardown,
        rootpath=tmp_path,
    )
    record.assert_outcomes(passed=3)


def test_scoped_fixture_caching_exception(tmp_path: Path) -> None:
    """Make sure setup & finalization is only run once for scoped fixture, with a cached exception."""
    executed_crash: list[str] = []

    @pytest.fixture(scope="class")
    def fixture_crash(request: pytest.FixtureRequest) -> None:
        executed_crash.append("fix_crash setup")

        def my_finalizer() -> None:
            executed_crash.append("fix_crash teardown")

        request.addfinalizer(my_finalizer)

        raise Exception("foo")

    class TestFixtureCachingException:
        @pytest.mark.xfail
        def test_crash_1(self, fixture_crash: None) -> None: ...

        @pytest.mark.xfail
        def test_crash_2(self, fixture_crash: None) -> None: ...

    def test_crash_expected_setup_and_teardown() -> None:
        assert executed_crash == ["fix_crash setup", "fix_crash teardown"]

    record = run_tests(
        fixture_crash,
        TestFixtureCachingException,
        test_crash_expected_setup_and_teardown,
        rootpath=tmp_path,
    )
    # The original only asserted a zero exit status; the two crashing tests
    # are xfail, so they count as xfailed rather than errors.
    record.assert_outcomes(passed=1, xfailed=2)


def test_scoped_fixture_teardown_order(tmp_path: Path) -> None:
    """
    Make sure teardowns happen in reverse order of setup with scoped fixtures, when
    a later test only depends on a subset of scoped fixtures.

    Regression test for https://github.com/pytest-dev/pytest/issues/1489
    """
    # The original's module global becomes a one-element list.
    last_executed = [""]

    @pytest.fixture(scope="module")
    def fixture_1():
        assert last_executed[0] == ""
        last_executed[0] = "fixture_1_setup"
        yield
        assert last_executed[0] == "fixture_2_teardown"
        last_executed[0] = "fixture_1_teardown"

    @pytest.fixture(scope="module")
    def fixture_2():
        assert last_executed[0] == "fixture_1_setup"
        last_executed[0] = "fixture_2_setup"
        yield
        assert last_executed[0] == "run_test"
        last_executed[0] = "fixture_2_teardown"

    def test_fixture_teardown_order(fixture_1: None, fixture_2: None) -> None:
        assert last_executed[0] == "fixture_2_setup"
        last_executed[0] = "run_test"

    def test_2(fixture_1: None) -> None:
        # This would previously queue an additional teardown of fixture_1,
        # despite fixture_1's value being cached, which caused fixture_1 to be
        # torn down before fixture_2 - violating the rule that teardowns should
        # happen in reverse order of setup.
        pass

    record = run_tests(
        fixture_1, fixture_2, test_fixture_teardown_order, test_2, rootpath=tmp_path
    )
    record.assert_outcomes(passed=2)
    assert last_executed[0] == "fixture_1_teardown"


def test_subfixture_teardown_order(tmp_path: Path) -> None:
    """
    Make sure fixtures don't re-register their finalization in parent fixtures multiple
    times, causing ordering failure in their teardowns.

    Regression test for #12135
    """
    execution_order: list[str] = []

    @pytest.fixture(scope="class")
    def fixture_1(): ...

    @pytest.fixture(scope="class")
    def fixture_2(fixture_1):
        execution_order.append("setup 2")
        yield
        execution_order.append("teardown 2")

    @pytest.fixture(scope="class")
    def fixture_3(fixture_1):
        execution_order.append("setup 3")
        yield
        execution_order.append("teardown 3")

    class TestFoo:
        def test_initialize_fixtures(self, fixture_2, fixture_3): ...

        # This would previously reschedule fixture_2's finalizer in the parent fixture,
        # causing it to be torn down before fixture 3.
        def test_reschedule_fixture_2(self, fixture_2): ...

        # Force finalization directly on fixture_1
        # Otherwise the cleanup would sequence 3&2 before 1 as normal.
        @pytest.mark.parametrize("fixture_1", [None], indirect=["fixture_1"])
        def test_finalize_fixture_1(self, fixture_1): ...

    def test_result():
        assert execution_order == ["setup 2", "setup 3", "teardown 3", "teardown 2"]

    record = run_tests(
        fixture_1, fixture_2, fixture_3, TestFoo, test_result, rootpath=tmp_path
    )
    record.assert_outcomes(passed=4)


def test_parametrized_fixture_scope_allowed(tmp_path: Path) -> None:
    """
    Make sure scope from parametrize does not affect fixture's ability to be
    depended upon.

    Regression test for #13248
    """

    @pytest.fixture(scope="session")
    def my_fixture(request):
        return getattr(request, "param", None)

    @pytest.fixture(scope="session")
    def another_fixture(my_fixture):
        return my_fixture

    @pytest.mark.parametrize("my_fixture", ["a value"], indirect=True, scope="function")
    def test_foo(another_fixture):
        assert another_fixture == "a value"

    record = run_tests(my_fixture, another_fixture, test_foo, rootpath=tmp_path)
    record.assert_outcomes(passed=1)


def test_collect_positional_only(tmp_path: Path) -> None:
    """Support the collection of tests with positional-only arguments (#13376)."""

    class Test:
        @pytest.fixture
        def fix(self):
            return 1

        def test_method(self, /, fix):
            assert fix == 1

    run_tests(Test, rootpath=tmp_path).assert_outcomes(passed=1)


def test_parametrization_dependency_pruning(tmp_path: Path) -> None:
    """Test that when a fixture is dynamically shadowed by parameterization, it
    is properly pruned and not executed."""

    # This fixture should never run because shadowed_fixture is parametrized.
    @pytest.fixture
    def boom():
        raise RuntimeError("BOOM!")

    # This fixture is shadowed by metafunc.parametrize in pytest_generate_tests.
    @pytest.fixture
    def shadowed_fixture(boom):
        return "fixture_value"

    # Dynamically parametrize shadowed_fixture, replacing the fixture with direct values.
    def pytest_generate_tests(metafunc):
        if "shadowed_fixture" in metafunc.fixturenames:
            metafunc.parametrize("shadowed_fixture", ["param1", "param2"])

    # This test should receive shadowed_fixture as a parametrized value, and
    # boom should not explode.
    def test_shadowed(shadowed_fixture):
        assert shadowed_fixture in ["param1", "param2"]

    module = build_module(
        "test_parametrization_dependency_pruning",
        boom,
        shadowed_fixture,
        test_shadowed,
        pytest_generate_tests=pytest_generate_tests,
    )
    run_tests(module, rootpath=tmp_path).assert_outcomes(passed=2)


def test_fixture_closure_with_overrides(tmp_path: Path) -> None:
    """Test that an item's static fixture closure properly includes transitive
    dependencies through overridden fixtures (#13773)."""

    class ConftestPlugin:
        @pytest.fixture
        def db(self): ...

        @pytest.fixture
        def app(self, db): ...

    # Overrides conftest-level `app` and requests it.
    @pytest.fixture
    def app(app): ...

    class TestClass:
        # Overrides module-level `app` and requests it.
        @pytest.fixture
        def app(self, app): ...

        def test_something(self, request, app):
            # Both dynamic and static fixture closures should include 'db'.
            assert "db" in request.fixturenames
            assert "db" in request.node.fixturenames
            # No dynamic dependencies, should be equal.
            assert set(request.fixturenames) == set(request.node.fixturenames)

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
    record = run_tests(build_module("test_overrides", TestClass, app=app), spec=spec)
    record.assert_outcomes(passed=1)


def test_fixture_closure_with_overrides_and_intermediary(tmp_path: Path) -> None:
    """Test that an item's static fixture closure properly includes transitive
    dependencies through overridden fixtures (#13773).

    A more complicated case than test_fixture_closure_with_overrides, adds an
    intermediary so the override chain is not direct.
    """

    class ConftestPlugin:
        @pytest.fixture
        def db(self): ...

        @pytest.fixture
        def app(self, db): ...

        @pytest.fixture
        def intermediate(self, app): ...

    # Overrides conftest-level `app` and requests it.
    @pytest.fixture
    def app(intermediate): ...

    class TestClass:
        # Overrides module-level `app` and requests it.
        @pytest.fixture
        def app(self, app): ...

        def test_something(self, request, app):
            # Both dynamic and static fixture closures should include 'db'.
            assert "db" in request.fixturenames
            assert "db" in request.node.fixturenames
            # No dynamic dependencies, should be equal.
            assert set(request.fixturenames) == set(request.node.fixturenames)

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
    record = run_tests(build_module("test_intermediary", TestClass, app=app), spec=spec)
    record.assert_outcomes(passed=1)


def test_fixture_closure_with_overrides_and_parametrization(tmp_path: Path) -> None:
    """Test that an item's static fixture closure properly includes transitive
    dependencies through overridden fixtures (#13773) when also including
    parametrization (#14248)."""

    class ConftestPlugin:
        @pytest.fixture
        def db(self): ...

        @pytest.fixture
        def app(self, db): ...

    # Overrides conftest-level `app` and requests it.
    @pytest.fixture
    def app(app): ...

    class TestClass:
        # Overrides module-level `app` and requests it.
        @pytest.fixture
        def app(self, app): ...

        @pytest.mark.parametrize("a", [1])
        def test_something(self, request, app, a):
            # Both dynamic and static fixture closures should include 'db'.
            assert "db" in request.fixturenames
            assert "db" in request.node.fixturenames
            # No dynamic dependencies, should be equal.
            assert set(request.fixturenames) == set(request.node.fixturenames)

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
    record = run_tests(
        build_module("test_parametrized_override", TestClass, app=app), spec=spec
    )
    record.assert_outcomes(passed=1)


def test_fixture_closure_with_broken_override_chain(tmp_path: Path) -> None:
    """Test that an item's static fixture closure properly includes transitive
    dependencies through overridden fixtures (#13773).

    A more complicated case than test_fixture_closure_with_overrides, one of the
    fixtures in the chain doesn't call its super, so it shouldn't be included.
    """

    class ConftestPlugin:
        @pytest.fixture
        def db(self): ...

        @pytest.fixture
        def app(self, db): ...

    # Overrides conftest-level `app` and *doesn't* request it.
    @pytest.fixture
    def app(): ...

    class TestClass:
        # Overrides module-level `app` and requests it.
        @pytest.fixture
        def app(self, app): ...

        def test_something(self, request, app):
            # Both dynamic and static fixture closures should include 'db'.
            assert "db" not in request.fixturenames
            assert "db" not in request.node.fixturenames
            # No dynamic dependencies, should be equal.
            assert set(request.fixturenames) == set(request.node.fixturenames)

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
    record = run_tests(build_module("test_broken_chain", TestClass, app=app), spec=spec)
    record.assert_outcomes(passed=1)


def test_fixture_closure_handles_circular_dependencies(tmp_path: Path) -> None:
    """Test that getfixtureclosure properly handles circular dependencies.

    The test will error in the runtest phase due to the fixture loop,
    but the closure computation still completes.
    """

    # Direct circular dependency.
    @pytest.fixture
    def fix_a(fix_b): ...

    @pytest.fixture
    def fix_b(fix_a): ...

    # Indirect circular dependency through multiple fixtures.
    @pytest.fixture
    def fix_x(fix_y): ...

    @pytest.fixture
    def fix_y(fix_z): ...

    @pytest.fixture
    def fix_z(fix_x): ...

    def test_circular_deps(fix_a, fix_x):
        pass

    items = collect_tests(
        fix_a, fix_b, fix_x, fix_y, fix_z, test_circular_deps, rootpath=tmp_path
    )
    assert isinstance(items[0], Function)
    assert items[0].fixturenames == ["fix_a", "fix_b", "fix_x", "fix_y", "fix_z"]


def test_fixture_closure_handles_diamond_dependencies(tmp_path: Path) -> None:
    """Test that getfixtureclosure properly handles diamond dependencies."""

    @pytest.fixture
    def db(): ...

    @pytest.fixture
    def user(db): ...

    @pytest.fixture
    def session(db): ...

    @pytest.fixture
    def app(user, session): ...

    def test_diamond_deps(request, app):
        assert request.node.fixturenames == [
            "request",
            "app",
            "user",
            "db",
            "session",
        ]
        assert request.fixturenames == ["request", "app", "user", "db", "session"]

    record = run_tests(db, user, session, app, test_diamond_deps, rootpath=tmp_path)
    record.assert_outcomes(passed=1)


def test_fixture_closure_with_complex_override_and_shared_deps(
    tmp_path: Path,
) -> None:
    """Test that shared dependencies in override chains are processed only once."""

    class ConftestPlugin:
        @pytest.fixture
        def db(self): ...

        @pytest.fixture
        def cache(self): ...

        @pytest.fixture
        def settings(self): ...

        @pytest.fixture
        def app(self, db, cache, settings): ...

    # Override app, but also directly use cache and settings.
    # This creates multiple paths to the same fixtures.
    @pytest.fixture
    def app(app, cache, settings): ...

    class TestClass:
        # Another override that uses both app and cache.
        @pytest.fixture
        def app(self, app, cache): ...

        def test_shared_deps(self, request, app):
            assert request.node.fixturenames == [
                "request",
                "app",
                "db",
                "cache",
                "settings",
            ]

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
    record = run_tests(build_module("test_shared_deps", TestClass, app=app), spec=spec)
    record.assert_outcomes(passed=1)


def test_fixture_closure_with_parametrize_ignore(tmp_path: Path) -> None:
    """Test that getfixtureclosure properly handles parametrization argnames
    which override a fixture."""

    @pytest.fixture
    def fix1(fix2): ...

    @pytest.fixture
    def fix2(fix3): ...

    @pytest.fixture
    def fix3(): ...

    @pytest.mark.parametrize("fix2", ["2"])
    def test_it(request, fix1):
        assert request.node.fixturenames == ["request", "fix1", "fix2"]
        assert request.fixturenames == ["request", "fix1", "fix2"]

    record = run_tests(fix1, fix2, fix3, test_it, rootpath=tmp_path)
    record.assert_outcomes(passed=1)


def test_overridden_fixture_depends_on_parametrized(tmp_path: Path) -> None:
    """#11075"""

    @pytest.fixture(params=["foo"])
    def fixture_foo(request):
        yield request.param

    @pytest.fixture
    def fixture_bar(fixture_foo):
        yield fixture_foo

    class TestFoobar:
        @pytest.fixture
        def fixture_bar(self, fixture_bar):
            yield fixture_bar

        def test_foobar(self, fixture_bar):
            assert fixture_bar == "foo"

    record = run_tests(fixture_foo, fixture_bar, TestFoobar, rootpath=tmp_path)
    record.assert_outcomes(passed=1)


def _custom_deco(func):
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


#: An ensemble that must see fixture-discovery warnings: the ensemble's own
#: filters take precedence over the host suite's ``filterwarnings = error``.
_WARN_INICFG = {"filterwarnings": ["default::pytest.PytestWarning"]}


def test_custom_decorated_fixture_warning(tmp_path: Path) -> None:
    """Fixtures wrapped by custom decorators using functools.wraps warn."""

    class TestClass:
        @_custom_deco
        @pytest.fixture
        def my_fixture(self):
            return "fixture_value"

        def test_fixture_usage(self, my_fixture):
            assert my_fixture == "fixture_value"

    spec = ConfigSpec(rootpath=tmp_path, inicfg=_WARN_INICFG)
    record = run_tests(TestClass, spec=spec, capture_output=True)
    # The original matched the warning's file:line too, which is host-anchored
    # for an in-memory source.
    assert [str(w.message) for w in record.warnings] == [
        "cannot discover fixture 'my_fixture' due to being wrapped in decorators"
    ]
    record.stdout.fnmatch_lines(["*fixture 'my_fixture' not found*"])
    record.assert_outcomes(errors=1, warnings=1)


def test_custom_decorated_fixture_above_classmethod_warning(tmp_path: Path) -> None:
    """Warn when wraps hides a fixture that itself wraps @classmethod.

    The fixture definition stores the classmethod descriptor; warning emission
    peels it to reach the underlying function for warn_explicit_for.
    """

    class TestClass:
        @_custom_deco
        @pytest.fixture(scope="class")
        @classmethod
        def my_fixture(cls):
            return "fixture_value"

        def test_fixture_usage(self, my_fixture):
            assert my_fixture == "fixture_value"

    spec = ConfigSpec(rootpath=tmp_path, inicfg=_WARN_INICFG)
    record = run_tests(TestClass, spec=spec, capture_output=True)
    assert [str(w.message) for w in record.warnings] == [
        "cannot discover fixture 'my_fixture' due to being wrapped in decorators"
    ]
    record.stdout.fnmatch_lines(["*fixture 'my_fixture' not found*"])
    record.assert_outcomes(errors=1, warnings=1)


def test_classmethod_above_fixture_warning(tmp_path: Path) -> None:
    """@classmethod above @pytest.fixture hides the fixture (#13507)."""

    class TestFixture:
        @classmethod
        @pytest.fixture(scope="class")
        def fixt(cls):
            return 1

        def test_fixt(self, fixt):
            assert fixt == 1

    spec = ConfigSpec(rootpath=tmp_path, inicfg=_WARN_INICFG)
    record = run_tests(TestFixture, spec=spec, capture_output=True)
    assert [str(w.message) for w in record.warnings] == [
        "cannot discover fixture 'fixt' because it is wrapped by @classmethod; "
        "place @pytest.fixture above @classmethod"
    ]
    record.stdout.fnmatch_lines(["*fixture 'fixt' not found*"])
    record.assert_outcomes(errors=1, warnings=1)


def test_staticmethod_above_fixture_warning(tmp_path: Path) -> None:
    """@staticmethod above @pytest.fixture always warns.

    Unlike ``classmethod``, discovery still finds the fixture via
    ``staticmethod.__get__``, so the test can pass; a leading ``self``/``cls``
    already fails as a missing fixture without special-casing here.
    """

    class TestFixture:
        @staticmethod
        @pytest.fixture
        def fixt():
            return 1

        def test_fixt(self, fixt):
            assert fixt == 1

    spec = ConfigSpec(rootpath=tmp_path, inicfg=_WARN_INICFG)
    record = run_tests(TestFixture, spec=spec)
    assert [str(w.message) for w in record.warnings] == [
        "fixture 'fixt' is wrapped by @staticmethod above @pytest.fixture; "
        "place @pytest.fixture above @staticmethod"
    ]
    record.assert_outcomes(passed=1, warnings=1)


def test_fixture_above_classmethod_still_works(tmp_path: Path) -> None:
    """Documented order @pytest.fixture above @classmethod remains discoverable."""

    class TestFixture:
        @pytest.fixture(scope="class")
        @classmethod
        def fixt(cls):
            return 1

        def test_fixt(self, fixt):
            assert fixt == 1

    run_tests(TestFixture, rootpath=tmp_path).assert_outcomes(passed=1)


def test_fixture_above_staticmethod_still_works(tmp_path: Path) -> None:
    """@pytest.fixture above @staticmethod remains discoverable without warning."""

    class TestFixture:
        @pytest.fixture
        @staticmethod
        def fixt():
            return 1

        def test_fixt(self, fixt):
            assert fixt == 1

    # -W error::pytest.PytestWarning of the original.
    spec = ConfigSpec(
        rootpath=tmp_path, inicfg={"filterwarnings": ["error::pytest.PytestWarning"]}
    )
    run_tests(TestFixture, spec=spec).assert_outcomes(passed=1, warnings=0)


def test_classmethod_above_fixture_warning_inherited(tmp_path: Path) -> None:
    """MRO ``__dict__`` lookup finds @classmethod wrappers on a base class."""

    class Base:
        @classmethod
        @pytest.fixture(scope="class")
        def fixt(cls):
            return 1

    class TestFixture(Base):
        def test_fixt(self, fixt):
            assert fixt == 1

    spec = ConfigSpec(rootpath=tmp_path, inicfg=_WARN_INICFG)
    record = run_tests(TestFixture, spec=spec, capture_output=True)
    assert [str(w.message) for w in record.warnings] == [
        "cannot discover fixture 'fixt' because it is wrapped by @classmethod; "
        "place @pytest.fixture above @classmethod"
    ]
    record.stdout.fnmatch_lines(["*fixture 'fixt' not found*"])
    record.assert_outcomes(errors=1, warnings=1)

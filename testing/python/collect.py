# mypy: allow-untyped-defs
from __future__ import annotations

import os
from pathlib import Path
import sys
import textwrap
from typing import Any

import _pytest._code
from _pytest.ensemble import build_module
from _pytest.ensemble import collect_tests
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import Ensemble
from _pytest.ensemble import run_tests
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Node
from _pytest.pytester import Pytester
from _pytest.python import Class
from _pytest.python import Function
import pytest


# ensemble: this whole class is about the module *import* chokepoint - real
# files on disk, import modes, import errors and their tracebacks - which is
# exactly what EnsembleModule bypasses by serving a preset object.
class TestModule:
    def test_failing_import(self, pytester: Pytester) -> None:
        modcol = pytester.getmodulecol("import alksdjalskdjalkjals")
        with pytest.raises(Collector.CollectError):
            modcol.collect()

    def test_import_duplicate(self, pytester: Pytester) -> None:
        a = pytester.mkdir("a")
        b = pytester.mkdir("b")
        p1 = a.joinpath("test_whatever.py")
        p1.touch()
        p2 = b.joinpath("test_whatever.py")
        p2.touch()
        # ensure we don't have it imported already
        sys.modules.pop(p1.stem, None)

        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*import*mismatch*",
                "*imported*test_whatever*",
                f"*{p1}*",
                "*not the same*",
                f"*{p2}*",
                "*HINT*",
            ]
        )

    def test_import_prepend_append(
        self, pytester: Pytester, monkeypatch: MonkeyPatch
    ) -> None:
        root1 = pytester.mkdir("root1")
        root2 = pytester.mkdir("root2")
        root1.joinpath("x456.py").touch()
        root2.joinpath("x456.py").touch()
        p = root2.joinpath("test_x456.py")
        monkeypatch.syspath_prepend(str(root1))
        p.write_text(
            textwrap.dedent(
                f"""\
                import x456
                def test():
                    assert x456.__file__.startswith({str(root2)!r})
                """
            ),
            encoding="utf-8",
        )
        with monkeypatch.context() as mp:
            mp.chdir(root2)
            reprec = pytester.inline_run("--import-mode=append")
            reprec.assertoutcome(passed=0, failed=1)
            reprec = pytester.inline_run()
            reprec.assertoutcome(passed=1)

    def test_syntax_error_in_module(self, pytester: Pytester) -> None:
        modcol = pytester.getmodulecol("this is a syntax error")
        with pytest.raises(modcol.CollectError):
            modcol.collect()
        with pytest.raises(modcol.CollectError):
            modcol.collect()

    def test_module_considers_pluginmanager_at_import(self, pytester: Pytester) -> None:
        modcol = pytester.getmodulecol("pytest_plugins='xasdlkj',")
        with pytest.raises(ImportError):
            modcol.obj()

    def test_invalid_test_module_name(self, pytester: Pytester) -> None:
        a = pytester.mkdir("a")
        a.joinpath("test_one.part1.py").touch()
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "ImportError while importing test module*test_one.part1*",
                "Hint: make sure your test modules/packages have valid Python names.",
            ]
        )

    @pytest.mark.parametrize("verbose", [0, 1, 2])
    def test_show_traceback_import_error(
        self, pytester: Pytester, verbose: int
    ) -> None:
        """Import errors when collecting modules should display the traceback (#1976).

        With low verbosity we omit pytest and internal modules, otherwise show all traceback entries.
        """
        pytester.makepyfile(
            foo_traceback_import_error="""
               from bar_traceback_import_error import NOT_AVAILABLE
           """,
            bar_traceback_import_error="",
        )
        pytester.makepyfile(
            """
               import foo_traceback_import_error
        """
        )
        args = ("-v",) * verbose
        result = pytester.runpytest(*args)
        result.stdout.fnmatch_lines(
            [
                "ImportError while importing test module*",
                "Traceback:",
                "*from bar_traceback_import_error import NOT_AVAILABLE",
                "*cannot import name *NOT_AVAILABLE*",
            ]
        )
        assert result.ret == 2

        stdout = result.stdout.str()
        if verbose == 2:
            assert "_pytest" in stdout
        else:
            assert "_pytest" not in stdout

    def test_show_traceback_import_error_unicode(self, pytester: Pytester) -> None:
        """Check test modules collected which raise ImportError with unicode messages
        are handled properly (#2336).
        """
        pytester.makepyfile("raise ImportError('Something bad happened ☺')")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(
            [
                "ImportError while importing test module*",
                "Traceback:",
                "*raise ImportError*Something bad happened*",
            ]
        )
        assert result.ret == 2


class TestClass:
    def test_class_with_init_warning(self, tmp_path: Path) -> None:
        class TestClass1:
            def __init__(self):
                pass

        module = build_module("test_class_with_init_warning", TestClass1)
        record = run_tests(
            module,
            spec=ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["always"]}),
        )
        record.assert_outcomes()
        assert [str(w.message) for w in record.warnings] == [
            "cannot collect test class 'TestClass1' because it has "
            "a __init__ constructor (from: test_class_with_init_warning.py)"
        ]

    def test_class_with_new_warning(self, tmp_path: Path) -> None:
        class TestClass1:
            def __new__(self):  # noqa: PLW0211
                pass

        module = build_module("test_class_with_new_warning", TestClass1)
        record = run_tests(
            module,
            spec=ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["always"]}),
        )
        record.assert_outcomes()
        assert [str(w.message) for w in record.warnings] == [
            "cannot collect test class 'TestClass1' because it has "
            "a __new__ constructor (from: test_class_with_new_warning.py)"
        ]

    def test_class_subclassobject(self, tmp_path: Path) -> None:
        class test:
            pass

        assert collect_tests(test, rootpath=tmp_path) == []

    def test_class_from_parent_without_obj_resolves_by_name(
        self, tmp_path: Path
    ) -> None:
        """Without an explicit obj, the class is looked up on the parent's object."""

        class TestGroup:
            def test_method(self):
                pass

        module = build_module("test_from_parent", TestGroup)
        with Ensemble(module, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            modcol = item.getparent(pytest.Module)
            assert modcol is not None
            cls = pytest.Class.from_parent(modcol, name="TestGroup")
            assert cls.obj is modcol.obj.TestGroup

    def test_static_method(self, tmp_path: Path) -> None:
        """Support for collecting staticmethod tests (#2528, #2699)"""

        class Test:
            @staticmethod
            def test_something():
                pass

            @pytest.fixture
            def fix(self):
                return 1

            @staticmethod
            def test_fix(fix):
                assert fix == 1

        record = run_tests(Test, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_setup_teardown_class_as_classmethod(self, tmp_path: Path) -> None:
        events = []

        class TestClassMethod:
            @classmethod
            def setup_class(cls):
                events.append("setup")

            def test_1(self):
                pass

            @classmethod
            def teardown_class(cls):
                events.append("teardown")

        record = run_tests(TestClassMethod, rootpath=tmp_path)
        record.assert_outcomes(passed=1)
        assert events == ["setup", "teardown"]

    def test_issue1035_obj_has_getattr(self, tmp_path: Path) -> None:
        class Chameleon:
            def __getattr__(self, name):
                return True

        module = build_module("test_issue1035", Chameleon, chameleon=Chameleon())
        assert collect_tests(module, rootpath=tmp_path) == []

    def test_issue1579_namedtuple(self, tmp_path: Path) -> None:
        import collections

        TestCase = collections.namedtuple("TestCase", ["a"])  # noqa: PYI024

        module = build_module("test_issue1579_namedtuple", TestCase)
        record = run_tests(
            module,
            spec=ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["always"]}),
        )
        record.assert_outcomes()
        assert [str(w.message) for w in record.warnings] == [
            "cannot collect test class 'TestCase' because it has "
            "a __new__ constructor (from: test_issue1579_namedtuple.py)"
        ]

    def test_issue2234_property(self, tmp_path: Path) -> None:
        class TestCase:
            @property
            def prop(self):
                raise NotImplementedError

        assert collect_tests(TestCase, rootpath=tmp_path) == []

    def test_does_not_discover_properties(self, tmp_path: Path) -> None:
        """Regression test for #12446."""

        class TestCase:
            @property
            def oops(self):
                raise SystemExit("do not call me!")

        assert collect_tests(TestCase, rootpath=tmp_path) == []

    def test_does_not_discover_instance_descriptors(self, tmp_path: Path) -> None:
        """Regression test for #12446."""

        # not `@property`, but it acts like one
        # this should cover the case of things like `@cached_property` / etc.
        class MyProperty:
            def __init__(self, func):
                self._func = func

            def __get__(self, inst, owner):
                if inst is None:
                    return self
                else:
                    return self._func.__get__(inst, owner)()

        class TestCase:
            @MyProperty
            def oops(self):
                raise SystemExit("do not call me!")

        assert collect_tests(TestCase, rootpath=tmp_path) == []

    def test_does_not_eval_properties_when_collecting_tests(
        self, tmp_path: Path
    ) -> None:
        """Regression test for #2568.

        Properties on a test class must only be evaluated when a test accesses
        them, not during collection or fixture parsing.
        """
        calls: list[int] = []

        class TestCase:
            @property
            def prop(self):
                calls.append(1)
                return len(calls)

            def test_prop(self):
                assert self.prop == 1

        record = run_tests(TestCase, rootpath=tmp_path)
        record.assert_outcomes(passed=1)
        assert calls == [1]

    def test_abstract_class_is_not_collected(self, tmp_path: Path) -> None:
        """Regression test for #12275 (non-unittest version)."""
        import abc

        class TestBase(abc.ABC):
            @abc.abstractmethod
            def abstract1(self): ...

            @abc.abstractmethod
            def abstract2(self): ...

            def test_it(self): ...  # noqa: B027

        class TestPartial(TestBase):
            def abstract1(self): ...

        class TestConcrete(TestPartial):
            def abstract2(self): ...

        record = run_tests(
            TestBase, TestPartial, TestConcrete, rootpath=tmp_path, name="test_abstract"
        )
        record.assert_outcomes(passed=1)
        assert list(record.by_test) == ["test_abstract.py::TestConcrete::test_it"]


class TestFunction:
    def test_getmodulecollector(self, tmp_path: Path) -> None:
        def test_func():
            pass

        (item,) = collect_tests(test_func, rootpath=tmp_path)
        modcol = item.getparent(pytest.Module)
        assert isinstance(modcol, pytest.Module)
        assert hasattr(modcol.obj, "test_func")

    def test_function_as_object_instance_ignored(self, tmp_path: Path) -> None:
        class A:
            def __call__(self, tmp_path):
                0 / 0  # noqa: B018

        # ensemble: the warning's file:line location is host-anchored, so only
        # the message itself is asserted on.
        module = build_module("test_function_as_object_instance_ignored", A, test_a=A())
        record = run_tests(
            module,
            spec=ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["always"]}),
        )
        record.assert_outcomes()
        assert [str(w.message) for w in record.warnings] == [
            "cannot collect 'test_a' because it is not a function."
        ]

    @staticmethod
    def make_function(tmp_path: Path, **kwargs: Any) -> Any:
        from _pytest.ensemble import ConfigSpec
        from _pytest.ensemble import configured
        from _pytest.ensemble import running_session

        with configured(ConfigSpec(rootpath=tmp_path)) as config:
            with running_session(config) as session:
                return pytest.Function.from_parent(parent=session, **kwargs)

    def test_function_equality(self, tmp_path: Path) -> None:
        def func1():
            pass

        def func2():
            pass

        f1 = self.make_function(tmp_path, name="name", callobj=func1)
        assert f1 == f1
        f2 = self.make_function(
            tmp_path, name="name", callobj=func2, originalname="foobar"
        )
        assert f1 != f2

    def test_repr_produces_actual_test_id(self, tmp_path: Path) -> None:
        f = self.make_function(
            tmp_path, name=r"test[\xe5]", callobj=self.test_repr_produces_actual_test_id
        )
        assert repr(f) == r"<Function test[\xe5]>"

    def test_issue197_parametrize_emptyset(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize("arg", [])
        def test_function(arg):
            pass

        run_tests(test_function, rootpath=tmp_path).assert_outcomes(skipped=1)

    def test_single_tuple_unwraps_values(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(("arg",), [(1,)])
        def test_function(arg):
            assert arg == 1

        run_tests(test_function, rootpath=tmp_path).assert_outcomes(passed=1)

    def test_issue213_parametrize_value_no_equal(self, tmp_path: Path) -> None:
        class A:
            def __eq__(self, other):
                raise ValueError("not possible")

        @pytest.mark.parametrize("arg", [A()])
        def test_function(arg):
            assert arg.__class__.__name__ == "A"

        run_tests(test_function, rootpath=tmp_path).assert_outcomes(passed=1)

    def test_parametrize_with_non_hashable_values(self, tmp_path: Path) -> None:
        """Test parametrization with non-hashable values."""
        archival_mapping = {
            "1.0": {"tag": "1.0"},
            "1.2.2a1": {"tag": "release-1.2.2a1"},
        }

        @pytest.mark.parametrize("key value".split(), archival_mapping.items())
        def test_archival_to_version(key, value):
            assert key in archival_mapping
            assert value == archival_mapping[key]

        run_tests(test_archival_to_version, rootpath=tmp_path).assert_outcomes(passed=2)

    def test_parametrize_with_non_hashable_values_indirect(
        self, tmp_path: Path
    ) -> None:
        """Test parametrization with non-hashable values with indirect parametrization."""
        archival_mapping = {
            "1.0": {"tag": "1.0"},
            "1.2.2a1": {"tag": "release-1.2.2a1"},
        }

        @pytest.fixture
        def key(request):
            return request.param

        @pytest.fixture
        def value(request):
            return request.param

        @pytest.mark.parametrize(
            "key value".split(), archival_mapping.items(), indirect=True
        )
        def test_archival_to_version(key, value):
            assert key in archival_mapping
            assert value == archival_mapping[key]

        record = run_tests(key, value, test_archival_to_version, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_parametrize_overrides_fixture(self, tmp_path: Path) -> None:
        """Test parametrization when parameter overrides existing fixture with same name."""

        @pytest.fixture
        def value():
            return "value"

        @pytest.mark.parametrize("value", ["overridden"])
        def test_overridden_via_param(value):
            assert value == "overridden"

        @pytest.mark.parametrize("somevalue", ["overridden"])
        def test_not_overridden(value, somevalue):
            assert value == "value"
            assert somevalue == "overridden"

        @pytest.mark.parametrize("other,value", [("foo", "overridden")])
        def test_overridden_via_multiparam(other, value):
            assert other == "foo"
            assert value == "overridden"

        record = run_tests(
            value,
            test_overridden_via_param,
            test_not_overridden,
            test_overridden_via_multiparam,
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=3)

    def test_parametrize_overrides_parametrized_fixture(self, tmp_path: Path) -> None:
        """Test parametrization when parameter overrides existing parametrized fixture with same name."""

        @pytest.fixture(params=[1, 2])
        def value(request):
            return request.param

        @pytest.mark.parametrize("value", ["overridden"])
        def test_overridden_via_param(value):
            assert value == "overridden"

        record = run_tests(value, test_overridden_via_param, rootpath=tmp_path)
        record.assert_outcomes(passed=1)

    def test_parametrize_overrides_parametrized_fixture_with_unrelated_indirect(
        self, tmp_path: Path
    ) -> None:
        """Test parametrization when parameter overrides existing parametrized fixture with same name,
        and there is an unrelated indirect param.

        Regression test for #13974.
        """

        @pytest.fixture(params=["a", "b"])
        def target(request):
            return request.param

        @pytest.fixture
        def val(request):
            return int(request.param)

        @pytest.mark.parametrize(
            ["val", "target"],
            [
                ("1", 1),
                ("2", 2),
            ],
            indirect=["val"],
        )
        def test(val, target):
            assert val == target

        record = run_tests(target, val, test, rootpath=tmp_path)
        record.assert_outcomes(passed=2)

    def test_parametrize_overrides_indirect_dependency_fixture(
        self, tmp_path: Path
    ) -> None:
        """Test parametrization when parameter overrides a fixture that a test indirectly depends on"""
        fix3_instantiated = []

        @pytest.fixture
        def fix1(fix2):
            return fix2 + "1"

        @pytest.fixture
        def fix2(fix3):
            return fix3 + "2"

        @pytest.fixture
        def fix3():
            fix3_instantiated.append(True)
            return "3"

        @pytest.mark.parametrize("fix2", ["2"])
        def test_it(fix1):
            assert fix1 == "21"
            assert not fix3_instantiated

        record = run_tests(fix1, fix2, fix3, test_it, rootpath=tmp_path)
        record.assert_outcomes(passed=1)
        assert fix3_instantiated == []

    def test_parametrize_with_mark(self, tmp_path: Path) -> None:
        @pytest.mark.foo
        @pytest.mark.parametrize(
            "arg", [1, pytest.param(2, marks=[pytest.mark.baz, pytest.mark.bar])]
        )
        def test_function(arg):
            pass

        items = collect_tests(test_function, rootpath=tmp_path)
        keywords = [item.keywords for item in items]
        assert (
            "foo" in keywords[0]
            and "bar" not in keywords[0]
            and "baz" not in keywords[0]
        )
        assert "foo" in keywords[1] and "bar" in keywords[1] and "baz" in keywords[1]

    def test_parametrize_with_empty_string_arguments(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize("v", ("", " "))
        @pytest.mark.parametrize("w", ("", " "))
        def test(v, w): ...

        items = collect_tests(test, rootpath=tmp_path)
        names = {item.name for item in items}
        assert names == {"test[-]", "test[ -]", "test[- ]", "test[ - ]"}

    def test_function_equality_with_callspec(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize("arg", [1, 2])
        def test_function(arg):
            pass

        items = collect_tests(test_function, rootpath=tmp_path)
        assert items[0] != items[1]
        assert not (items[0] == items[1])

    def test_pyfunc_call(self, tmp_path: Path) -> None:
        def test_func():
            raise ValueError

        with Ensemble(test_func, rootpath=tmp_path) as ensemble:
            (item,) = ensemble.collect()
            config = ensemble.config

            class MyPlugin1:
                def pytest_pyfunc_call(self):
                    raise ValueError

            class MyPlugin2:
                def pytest_pyfunc_call(self):
                    return True

            config.pluginmanager.register(MyPlugin1())
            config.pluginmanager.register(MyPlugin2())
            config.hook.pytest_runtest_setup(item=item)
            config.hook.pytest_pyfunc_call(pyfuncitem=item)

    def test_multiple_parametrize(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize("x", [0, 1])
        @pytest.mark.parametrize("y", [2, 3])
        def test1(x, y):
            pass

        colitems = collect_tests(test1, rootpath=tmp_path)
        assert colitems[0].name == "test1[2-0]"
        assert colitems[1].name == "test1[2-1]"
        assert colitems[2].name == "test1[3-0]"
        assert colitems[3].name == "test1[3-1]"

    def test_issue751_multiple_parametrize_with_ids(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize("x", [0], ids=["c"])
        @pytest.mark.parametrize("y", [0, 1], ids=["a", "b"])
        class Test:
            def test1(self, x, y):
                pass

            def test2(self, x, y):
                pass

        colitems = collect_tests(Test, rootpath=tmp_path)
        assert colitems[0].name == "test1[a-c]"
        assert colitems[1].name == "test1[b-c]"
        assert colitems[2].name == "test2[a-c]"
        assert colitems[3].name == "test2[b-c]"

    def test_parametrize_skipif(self, tmp_path: Path) -> None:
        m = pytest.mark.skipif("True")

        @pytest.mark.parametrize("x", [0, 1, pytest.param(2, marks=m)])
        def test_skip_if(x):
            assert x < 2

        run_tests(test_skip_if, rootpath=tmp_path).assert_outcomes(passed=2, skipped=1)

    def test_parametrize_skip(self, tmp_path: Path) -> None:
        m = pytest.mark.skip("")

        @pytest.mark.parametrize("x", [0, 1, pytest.param(2, marks=m)])
        def test_skip(x):
            assert x < 2

        run_tests(test_skip, rootpath=tmp_path).assert_outcomes(passed=2, skipped=1)

    def test_parametrize_skipif_no_skip(self, tmp_path: Path) -> None:
        m = pytest.mark.skipif("False")

        @pytest.mark.parametrize("x", [0, 1, m(2)])
        def test_skipif_no_skip(x):
            assert x < 2

        run_tests(test_skipif_no_skip, rootpath=tmp_path).assert_outcomes(
            passed=2, failed=1
        )

    def test_parametrize_xfail(self, tmp_path: Path) -> None:
        m = pytest.mark.xfail("True")

        @pytest.mark.parametrize("x", [0, 1, pytest.param(2, marks=m)])
        def test_xfail(x):
            assert x < 2

        run_tests(test_xfail, rootpath=tmp_path).assert_outcomes(passed=2, xfailed=1)

    def test_parametrize_passed(self, tmp_path: Path) -> None:
        m = pytest.mark.xfail("True")

        @pytest.mark.parametrize("x", [0, 1, pytest.param(2, marks=m)])
        def test_xfail(x):
            pass

        run_tests(test_xfail, rootpath=tmp_path).assert_outcomes(passed=2, xpassed=1)

    def test_parametrize_xfail_passed(self, tmp_path: Path) -> None:
        m = pytest.mark.xfail("False")

        @pytest.mark.parametrize("x", [0, 1, m(2)])
        def test_passed(x):
            pass

        run_tests(test_passed, rootpath=tmp_path).assert_outcomes(passed=3)

    def test_function_originalname(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize("arg", [1, 2])
        def test_func(arg):
            pass

        def test_no_param():
            pass

        items = collect_tests(test_func, test_no_param, rootpath=tmp_path)
        originalnames = []
        for x in items:
            assert isinstance(x, pytest.Function)
            originalnames.append(x.originalname)
        assert originalnames == [
            "test_func",
            "test_func",
            "test_no_param",
        ]

    def test_function_with_square_brackets(self, tmp_path: Path) -> None:
        """Check that functions with square brackets don't cause trouble."""
        module = build_module(
            "test_function_with_square_brackets",
            **{"test_foo[name]": lambda: None},
        )
        record = run_tests(module, rootpath=tmp_path)
        record.assert_outcomes(passed=1)
        assert list(record.by_test) == [
            "test_function_with_square_brackets.py::test_foo[name]"
        ]


class TestSorting:
    def test_check_equality(self, tmp_path: Path) -> None:
        def test_pass():
            pass

        def test_fail():
            assert 0

        module = build_module("test_check_equality", test_pass, test_fail)
        with Ensemble(module, rootpath=tmp_path) as ensemble:
            fn1, fn3 = ensemble.collect()
            # collect() is idempotent, and pytester.collect_by_name memoized:
            # a second lookup of the same name is the very same node.
            fn2 = ensemble.collect()[0]
            assert fn2 is fn1
            # deliberately widened: comparing a Function to its Module is what
            # is under test here
            modcol: Node | None = fn1.getparent(pytest.Module)
            assert modcol is not None

        assert isinstance(fn1, pytest.Function)
        assert isinstance(fn2, pytest.Function)

        assert fn1 == fn2
        assert fn1 != modcol
        assert hash(fn1) == hash(fn2)

        assert isinstance(fn3, pytest.Function)
        assert not (fn1 == fn3)
        assert fn1 != fn3

        for fn in fn1, fn2, fn3:
            assert fn != 3  # type: ignore[comparison-overlap]
            assert fn != modcol
            assert fn != [1, 2, 3]  # type: ignore[comparison-overlap]
            assert [1, 2, 3] != fn  # type: ignore[comparison-overlap]
            assert modcol != fn

    def test_allow_sane_sorting_for_decorators(self, tmp_path: Path) -> None:
        def dec(f):
            def g():
                return f(2)

            g.place_as = f  # type: ignore[attr-defined]
            return g

        def test_b(y):
            pass

        def test_a(y):
            pass

        # the wrappers all carry the same name and line, so they have to be
        # given their module names explicitly
        module = build_module(
            "test_allow_sane_sorting_for_decorators",
            test_b=dec(test_b),
            test_a=dec(test_a),
        )
        colitems = collect_tests(module, rootpath=tmp_path)
        assert len(colitems) == 2
        assert [item.name for item in colitems] == ["test_b", "test_a"]

    def test_ordered_by_definition_order(self, tmp_path: Path) -> None:
        class Test1:
            def test_foo(self): ...

            def test_bar(self): ...

        class Test2:
            def test_foo(self): ...

            test_bar = Test1.test_bar

        class Test3(Test2):
            def test_baz(self): ...

        items = collect_tests(
            Test1, Test2, Test3, rootpath=tmp_path, name="test_ordered"
        )
        assert [item.nodeid for item in items] == [
            "test_ordered.py::Test1::test_foo",
            "test_ordered.py::Test1::test_bar",
            # previously the order was flipped due to Test1.test_bar reference
            "test_ordered.py::Test2::test_foo",
            "test_ordered.py::Test2::test_bar",
            "test_ordered.py::Test3::test_foo",
            "test_ordered.py::Test3::test_bar",
            "test_ordered.py::Test3::test_baz",
        ]


class TestConftestCustomization:
    # ensemble: pytest_pycollect_makemodule never fires for an ensemble -
    # collect_sources constructs the EnsembleModule directly.
    def test_pytest_pycollect_module(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest
            class MyModule(pytest.Module):
                pass
            def pytest_pycollect_makemodule(module_path, parent):
                if module_path.name == "test_xyz.py":
                    return MyModule.from_parent(path=module_path, parent=parent)
        """
        )
        pytester.makepyfile("def test_some(): pass")
        pytester.makepyfile(test_xyz="def test_func(): pass")
        result = pytester.runpytest("--collect-only")
        result.stdout.fnmatch_lines(["*<Module*test_pytest*", "*<MyModule*xyz*"])

    # ensemble: as above, plus this one is specifically about the conftest in a
    # subdirectory applying to that directory only.
    def test_customized_pymakemodule_issue205_subdir(self, pytester: Pytester) -> None:
        b = pytester.path.joinpath("a", "b")
        b.mkdir(parents=True)
        b.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                import pytest
                @pytest.hookimpl(wrapper=True)
                def pytest_pycollect_makemodule():
                    mod = yield
                    mod.obj.hello = "world"
                    return mod
                """
            ),
            encoding="utf-8",
        )
        b.joinpath("test_module.py").write_text(
            textwrap.dedent(
                """\
                def test_hello():
                    assert hello == "world"
                """
            ),
            encoding="utf-8",
        )
        reprec = pytester.inline_run()
        reprec.assertoutcome(passed=1)

    def test_customized_pymakeitem(self, tmp_path: Path) -> None:
        class MakeItemPlugin:
            @pytest.hookimpl(wrapper=True)
            def pytest_pycollect_makeitem(self):
                result = yield
                if result:
                    for func in result:
                        func._some123 = "world"
                return result

        @pytest.fixture
        def obj(request):
            return request.node._some123

        def test_hello(obj):
            assert obj == "world"

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(MakeItemPlugin(),))
        run_tests(obj, test_hello, spec=spec).assert_outcomes(passed=1)

    def test_pytest_pycollect_makeitem(self, tmp_path: Path) -> None:
        class MyFunction(pytest.Function):
            pass

        class MakeItemPlugin:
            def pytest_pycollect_makeitem(self, collector, name, obj):
                if name == "some":
                    return MyFunction.from_parent(name=name, parent=collector)

        def some():
            pass

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(MakeItemPlugin(),))
        items = collect_tests(some, spec=spec, name="test_makeitem")
        assert [type(item) for item in items] == [MyFunction]
        assert items[0].nodeid == "test_makeitem.py::some"

    # ensemble: pytest_collect_file never fires for an ensemble, and this needs a
    # subprocess for the sys.meta_path futzing anyway.
    def test_issue2369_collect_module_fileext(self, pytester: Pytester) -> None:
        """Ensure we can collect files with weird file extensions as Python
        modules (#2369)"""
        # Implement a little meta path finder to import files containing
        # Python source code whose file extension is ".narf".
        pytester.makeconftest(
            """
            import sys
            import os.path
            from importlib.util import spec_from_loader
            from importlib.machinery import SourceFileLoader
            from _pytest.python import Module

            class MetaPathFinder:
                def find_spec(self, fullname, path, target=None):
                    if os.path.exists(fullname + ".narf"):
                        return spec_from_loader(
                            fullname,
                            SourceFileLoader(fullname, fullname + ".narf"),
                        )
            sys.meta_path.append(MetaPathFinder())

            def pytest_collect_file(file_path, parent):
                if file_path.suffix == ".narf":
                    return Module.from_parent(path=file_path, parent=parent)
            """
        )
        pytester.makefile(
            ".narf",
            """\
            def test_something():
                assert 1 + 1 == 2""",
        )
        # Use runpytest_subprocess, since we're futzing with sys.meta_path.
        result = pytester.runpytest_subprocess()
        result.stdout.fnmatch_lines(["*1 passed*"])

    def test_early_ignored_attributes(self, tmp_path: Path) -> None:
        """Builtin attributes should be ignored early on, even if
        configuration would otherwise allow them.

        This tests a performance optimization, not correctness, really,
        although it tests PytestCollectionWarning is not raised, while
        it would have been raised otherwise.
        """

        class TestEmpty:
            pass

        def test_real():
            pass

        module = build_module(
            "test_early_ignored_attributes",
            TestEmpty,
            test_real,
            test_empty=TestEmpty(),
        )
        spec = ConfigSpec(
            rootpath=tmp_path,
            inicfg={
                "python_classes": ["*"],
                "python_functions": ["*"],
                "filterwarnings": ["always"],
            },
        )
        record = run_tests(module, spec=spec)
        # only test_real is collected, and none of the builtin module/class
        # attributes was ever offered up to warn about
        record.assert_outcomes(passed=1, warnings=0)
        assert list(record.by_test) == ["test_early_ignored_attributes.py::test_real"]


# ensemble: the subject is conftest hooks being scoped to their directory;
# ensembles never load conftest files.
def test_setup_only_available_in_subdir(pytester: Pytester) -> None:
    sub1 = pytester.mkpydir("sub1")
    sub2 = pytester.mkpydir("sub2")
    sub1.joinpath("conftest.py").write_text(
        textwrap.dedent(
            """\
            import pytest
            def pytest_runtest_setup(item):
                assert item.path.stem == "test_in_sub1"
            def pytest_runtest_call(item):
                assert item.path.stem == "test_in_sub1"
            def pytest_runtest_teardown(item):
                assert item.path.stem == "test_in_sub1"
            """
        ),
        encoding="utf-8",
    )
    sub2.joinpath("conftest.py").write_text(
        textwrap.dedent(
            """\
            import pytest
            def pytest_runtest_setup(item):
                assert item.path.stem == "test_in_sub2"
            def pytest_runtest_call(item):
                assert item.path.stem == "test_in_sub2"
            def pytest_runtest_teardown(item):
                assert item.path.stem == "test_in_sub2"
            """
        ),
        encoding="utf-8",
    )
    sub1.joinpath("test_in_sub1.py").write_text("def test_1(): pass", encoding="utf-8")
    sub2.joinpath("test_in_sub2.py").write_text("def test_2(): pass", encoding="utf-8")
    result = pytester.runpytest("-v", "-s")
    result.assert_outcomes(passed=2)


# ensemble: re-collects from a nodeid trail through perform_collect, which
# resolves against the filesystem; an ensemble serves preset collectors.
def test_modulecol_roundtrip(pytester: Pytester) -> None:
    modcol = pytester.getmodulecol("pass", withinit=False)
    trail = modcol.nodeid
    newcol = modcol.session.perform_collect([trail], genitems=0)[0]
    assert modcol.name == newcol.name


class TestTracebackCutting:
    def test_skip_simple(self):
        with pytest.raises(pytest.skip.Exception) as excinfo:
            pytest.skip("xxx")
        if sys.version_info >= (3, 11):
            assert excinfo.traceback[-1].frame.code.raw.co_qualname == "_Skip.__call__"
        assert excinfo.traceback[-1].ishidden(excinfo)
        assert excinfo.traceback[-2].frame.code.name == "test_skip_simple"
        assert not excinfo.traceback[-2].ishidden(excinfo)

    # ensemble: asserts on rendered tracebacks anchored at a conftest file and
    # its line numbers, plus --fulltrace, which lives in the terminal plugin.
    def test_traceback_argsetup(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture
            def hello(request):
                raise ValueError("xyz")
        """
        )
        p = pytester.makepyfile("def test(hello): pass")
        result = pytester.runpytest(p)
        assert result.ret != 0
        out = result.stdout.str()
        assert "xyz" in out
        assert "conftest.py:5: ValueError" in out
        numentries = out.count("_ _ _")  # separator for traceback entries
        assert numentries == 0

        result = pytester.runpytest("--fulltrace", p)
        out = result.stdout.str()
        assert "conftest.py:5: ValueError" in out
        numentries = out.count("_ _ _ _")  # separator for traceback entries
        assert numentries > 3

    # ensemble: about the traceback of a module *import* error.
    def test_traceback_error_during_import(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            x = 1
            x = 2
            x = 17
            asd
        """
        )
        result = pytester.runpytest()
        assert result.ret != 0
        out = result.stdout.str()
        assert "x = 1" not in out
        assert "x = 2" not in out
        result.stdout.fnmatch_lines([" *asd*", "E*NameError*"])
        result = pytester.runpytest("--fulltrace")
        out = result.stdout.str()
        assert "x = 1" in out
        assert "x = 2" in out
        result.stdout.fnmatch_lines([">*asd*", "E*NameError*"])

    def test_traceback_filter_error_during_fixture_collection(
        self, tmp_path: Path
    ) -> None:
        """Integration test for issue #995."""

        def fail_me(func):
            ns: dict[str, Any] = {}
            exec('def w(): raise ValueError("fail me")', ns)
            return ns["w"]

        @pytest.fixture(scope="class")
        @fail_me
        def fail_fixture():
            pass

        def test_failing_fixture(fail_fixture):
            pass

        # the fixture carries the name of the generated function it wraps, so
        # it has to be given its module name explicitly
        module = build_module(
            "test_traceback_filter",
            test_failing_fixture,
            fail_fixture=fail_fixture,
        )
        record = run_tests(module, rootpath=tmp_path, capture_output=True)
        record.assert_outcomes(errors=1)
        assert "INTERNALERROR>" not in record.output
        record.stdout.fnmatch_lines(["*ValueError: fail me*", "* 1 error in *"])

    def test_filter_traceback_generated_code(self) -> None:
        """Test that filter_traceback() works with the fact that
        _pytest._code.code.Code.path attribute might return an str object.

        In this case, one of the entries on the traceback was produced by
        dynamically generated code.
        See: https://bitbucket.org/pytest-dev/py/issues/71
        This fixes #995.
        """
        from _pytest._code import filter_traceback

        tb = None
        try:
            ns: dict[str, Any] = {}
            exec("def foo(): raise ValueError", ns)
            ns["foo"]()
        except ValueError:
            _, _, tb = sys.exc_info()

        assert tb is not None
        traceback = _pytest._code.Traceback(tb)
        assert isinstance(traceback[-1].path, str)
        assert not filter_traceback(traceback[-1])

    # ensemble: needs a real importable file that is then deleted from disk.
    def test_filter_traceback_path_no_longer_valid(self, pytester: Pytester) -> None:
        """Test that filter_traceback() works with the fact that
        _pytest._code.code.Code.path attribute might return an str object.

        In this case, one of the files in the traceback no longer exists.
        This fixes #1133.
        """
        from _pytest._code import filter_traceback

        pytester.syspathinsert()
        pytester.makepyfile(
            filter_traceback_entry_as_str="""
            def foo():
                raise ValueError
        """
        )
        tb = None
        try:
            import filter_traceback_entry_as_str

            filter_traceback_entry_as_str.foo()
        except ValueError:
            _, _, tb = sys.exc_info()

        assert tb is not None
        pytester.path.joinpath("filter_traceback_entry_as_str.py").unlink()
        traceback = _pytest._code.Traceback(tb)
        assert isinstance(traceback[-1].path, str)
        assert filter_traceback(traceback[-1])


class TestReportInfo:
    # ensemble: reportinfo/location of an ensemble item is anchored at the
    # *host* file that defines the source, so these three would either fail or
    # have to hard-code this file's line numbers.
    def test_itemreport_reportinfo(self, pytester: Pytester) -> None:
        pytester.makeconftest(
            """
            import pytest
            class MyFunction(pytest.Function):
                def reportinfo(self):
                    return "ABCDE", 42, "custom"
            def pytest_pycollect_makeitem(collector, name, obj):
                if name == "test_func":
                    return MyFunction.from_parent(name=name, parent=collector)
        """
        )
        item = pytester.getitem("def test_func(): pass")
        item.config.pluginmanager.getplugin("runner")
        assert item.location == ("ABCDE", 42, "custom")

    def test_func_reportinfo(self, pytester: Pytester) -> None:
        item = pytester.getitem("def test_func(): pass")
        path, lineno, modpath = item.reportinfo()
        assert os.fspath(path) == str(item.path)
        assert lineno == 0
        assert modpath == "test_func"

    def test_class_reportinfo(self, pytester: Pytester) -> None:
        modcol = pytester.getmodulecol(
            """
            # lineno 0
            class TestClass(object):
                def test_hello(self): pass
        """
        )
        classcol = pytester.collect_by_name(modcol, "TestClass")
        assert isinstance(classcol, Class)
        path, lineno, msg = classcol.reportinfo()
        assert os.fspath(path) == str(modcol.path)
        assert lineno == 1
        assert msg == "TestClass"

    @pytest.mark.filterwarnings(
        "ignore:usage of Generator.Function is deprecated, please use pytest.Function instead"
    )
    def test_reportinfo_with_nasty_getattr(self, tmp_path: Path) -> None:
        # https://github.com/pytest-dev/pytest/issues/1204
        class TestClass:
            def __getattr__(self, name):
                return "this is not an int"

            def __class_getattr__(cls, name):
                return "this is not an int"

            def intest_foo(self):
                pass

            def test_bar(self):
                pass

        (item,) = collect_tests(TestClass, rootpath=tmp_path)
        classcol = item.getparent(Class)
        assert isinstance(classcol, Class)
        _path, _lineno, _msg = classcol.reportinfo()
        func = next(iter(classcol.collect()))
        assert isinstance(func, Function)
        _path, _lineno, _msg = func.reportinfo()


# ensemble: the python_files half of custom discovery only exists for files on
# disk - an EnsembleModule is collected whatever it is called - and this is the
# only test covering it. See test_customized_python_discovery_functions for the
# ported half.
def test_customized_python_discovery(pytester: Pytester) -> None:
    pytester.makeini(
        """
        [pytest]
        python_files=check_*.py
        python_classes=Check
        python_functions=check
    """
    )
    p = pytester.makepyfile(
        """
        def check_simple():
            pass
        class CheckMyApp(object):
            def check_meth(self):
                pass
    """
    )
    p2 = p.with_name(p.name.replace("test", "check"))
    p.rename(p2)
    result = pytester.runpytest("--collect-only", "-s")
    result.stdout.fnmatch_lines(
        ["*check_customized*", "*check_simple*", "*CheckMyApp*", "*check_meth*"]
    )

    result = pytester.runpytest()
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*2 passed*"])


def test_customized_python_discovery_functions(tmp_path: Path) -> None:
    def _test_underscore():
        pass

    module = build_module(
        "test_customized_python_discovery_functions", _test_underscore
    )
    spec = ConfigSpec(rootpath=tmp_path, inicfg={"python_functions": ["_test"]})
    record = run_tests(module, spec=spec)
    record.assert_outcomes(passed=1)
    assert list(record.by_test) == [
        "test_customized_python_discovery_functions.py::_test_underscore"
    ]


def test_unorderable_types(tmp_path: Path) -> None:
    class TestJoinEmpty:
        pass

    def make_test():
        class Test:
            pass

        Test.__name__ = "TestFoo"
        return Test

    TestFoo = make_test()

    # a TypeError while ordering the collected classes would surface as a
    # collection error, which collect_tests refuses to swallow
    assert collect_tests(TestJoinEmpty, TestFoo, rootpath=tmp_path) == []


def test_dont_collect_non_function_callable(tmp_path: Path) -> None:
    """Test for issue https://github.com/pytest-dev/pytest/issues/331

    In this case an INTERNALERROR occurred trying to report the failure of
    a test like this one because pytest failed to get the source lines.
    """

    class Oh:
        def __call__(self):
            pass

    def test_real():
        pass

    # ensemble: the warning's file:line location is host-anchored, so only the
    # message itself is asserted on.
    module = build_module(
        "test_dont_collect_non_function_callable", Oh, test_real, test_a=Oh()
    )
    record = run_tests(
        module,
        spec=ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["always"]}),
    )
    record.assert_outcomes(passed=1, warnings=1)
    assert list(record.by_test) == [
        "test_dont_collect_non_function_callable.py::test_real"
    ]
    assert [str(w.message) for w in record.warnings] == [
        "cannot collect 'test_a' because it is not a function."
    ]


def test_class_injection_does_not_break_collection(tmp_path: Path) -> None:
    """Tests whether injection during collection time will terminate testing.

    In this case the error should not occur if the TestClass itself
    is modified during collection time, and the original method list
    is still used for collection.
    """

    class TestClass:
        def test_injection(self):
            """Test being parametrized."""

    class InjectPlugin:
        def pytest_generate_tests(self, metafunc):
            TestClass.changed_var = {}  # type: ignore[attr-defined]

    spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(InjectPlugin(),))
    # a "dictionary changed size during iteration" RuntimeError would surface
    # as a collection error, which collect_tests refuses to swallow
    record = run_tests(TestClass, spec=spec)
    record.assert_outcomes(passed=1)


# ensemble: about a SyntaxError raised while importing a module from disk.
def test_syntax_error_with_non_ascii_chars(pytester: Pytester) -> None:
    """Fix decoding issue while formatting SyntaxErrors during collection (#578)."""
    pytester.makepyfile("☃")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*ERROR collecting*", "*SyntaxError*", "*1 error in*"])


# ensemble: renders a collect error for a module that fails at import, with
# --fulltrace from the terminal plugin and file-anchored line numbers.
def test_collect_error_with_fulltrace(pytester: Pytester) -> None:
    pytester.makepyfile("assert 0")
    result = pytester.runpytest("--fulltrace")
    result.stdout.fnmatch_lines(
        [
            "collected 0 items / 1 error",
            "",
            "*= ERRORS =*",
            "*_ ERROR collecting test_collect_error_with_fulltrace.py _*",
            "",
            ">   assert 0",
            "E   assert 0",
            "",
            "test_collect_error_with_fulltrace.py:1: AssertionError",
            "*! Interrupted: 1 error during collection !*",
        ]
    )


# ensemble: duplicate *path arguments* on the command line; an ensemble is
# handed objects, not paths.
def test_skip_duplicates_by_default(pytester: Pytester) -> None:
    """Test for issue https://github.com/pytest-dev/pytest/issues/1609 (#1609)

    Ignore duplicate directories.
    """
    a = pytester.mkdir("a")
    fh = a.joinpath("test_a.py")
    fh.write_text(
        textwrap.dedent(
            """\
            import pytest
            def test_real():
                pass
            """
        ),
        encoding="utf-8",
    )
    result = pytester.runpytest(str(a), str(a))
    result.stdout.fnmatch_lines(["*collected 1 item*"])


# ensemble: --keep-duplicates over duplicate path arguments, as above.
def test_keep_duplicates(pytester: Pytester) -> None:
    """Test for issue https://github.com/pytest-dev/pytest/issues/1609 (#1609)

    Use --keep-duplicates to collect tests from duplicate directories.
    """
    a = pytester.mkdir("a")
    fh = a.joinpath("test_a.py")
    fh.write_text(
        textwrap.dedent(
            """\
            import pytest
            def test_real():
                pass
            """
        ),
        encoding="utf-8",
    )
    result = pytester.runpytest("--keep-duplicates", str(a), str(a))
    result.stdout.fnmatch_lines(["*collected 2 item*"])


# ensemble: everything below is about collecting real directory trees -
# packages, __init__.py files, path arguments and the resulting hierarchy -
# none of which an ensemble has.
def test_package_collection_infinite_recursion(pytester: Pytester) -> None:
    pytester.copy_example("collect/package_infinite_recursion")
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(["*1 passed*"])


def test_package_collection_init_given_as_argument(pytester: Pytester) -> None:
    """Regression test for #3749, #8976, #9263, #9313.

    Specifying an __init__.py file directly should collect only the __init__.py
    Module, not the entire package.
    """
    p = pytester.copy_example("collect/package_init_given_as_arg")
    items, _hookrecorder = pytester.inline_genitems(p / "pkg" / "__init__.py")
    assert len(items) == 1
    assert items[0].name == "test_init"


def test_package_with_modules(pytester: Pytester) -> None:
    """
    .
    └── root
        ├── __init__.py
        ├── sub1
        │   ├── __init__.py
        │   └── sub1_1
        │       ├── __init__.py
        │       └── test_in_sub1.py
        └── sub2
            └── test
                └── test_in_sub2.py

    """
    root = pytester.mkpydir("root")
    sub1 = root.joinpath("sub1")
    sub1_test = sub1.joinpath("sub1_1")
    sub1_test.mkdir(parents=True)
    for d in (sub1, sub1_test):
        d.joinpath("__init__.py").touch()

    sub2 = root.joinpath("sub2")
    sub2_test = sub2.joinpath("test")
    sub2_test.mkdir(parents=True)

    sub1_test.joinpath("test_in_sub1.py").write_text(
        "def test_1(): pass", encoding="utf-8"
    )
    sub2_test.joinpath("test_in_sub2.py").write_text(
        "def test_2(): pass", encoding="utf-8"
    )

    # Execute from .
    result = pytester.runpytest("-v", "-s")
    result.assert_outcomes(passed=2)

    # Execute from . with one argument "root"
    result = pytester.runpytest("-v", "-s", "root")
    result.assert_outcomes(passed=2)

    # Chdir into package's root and execute with no args
    os.chdir(root)
    result = pytester.runpytest("-v", "-s")
    result.assert_outcomes(passed=2)


def test_package_ordering(pytester: Pytester) -> None:
    """
    .
    └── root
        ├── Test_root.py
        ├── __init__.py
        ├── sub1
        │   ├── Test_sub1.py
        │   └── __init__.py
        └── sub2
            └── test
                └── test_sub2.py

    """
    pytester.makeini(
        """
        [pytest]
        python_files=*.py
    """
    )
    root = pytester.mkpydir("root")
    sub1 = root.joinpath("sub1")
    sub1.mkdir()
    sub1.joinpath("__init__.py").touch()
    sub2 = root.joinpath("sub2")
    sub2_test = sub2.joinpath("test")
    sub2_test.mkdir(parents=True)

    root.joinpath("Test_root.py").write_text("def test_1(): pass", encoding="utf-8")
    sub1.joinpath("Test_sub1.py").write_text("def test_2(): pass", encoding="utf-8")
    sub2_test.joinpath("test_sub2.py").write_text(
        "def test_3(): pass", encoding="utf-8"
    )

    # Execute from .
    result = pytester.runpytest("-v", "-s")
    result.assert_outcomes(passed=3)


def test_collection_hierarchy(pytester: Pytester) -> None:
    """A general test checking that a filesystem hierarchy is collected as
    expected in various scenarios.

    top/
    ├── aaa
    │   ├── pkg
    │   │   ├── __init__.py
    │   │   └── test_pkg.py
    │   └── test_aaa.py
    ├── test_a.py
    ├── test_b
    │   ├── __init__.py
    │   └── test_b.py
    ├── test_c.py
    └── zzz
        ├── dir
        │   └── test_dir.py
        ├── __init__.py
        └── test_zzz.py
    """
    pytester.makepyfile(
        **{
            "top/aaa/test_aaa.py": "def test_it(): pass",
            "top/aaa/pkg/__init__.py": "",
            "top/aaa/pkg/test_pkg.py": "def test_it(): pass",
            "top/test_a.py": "def test_it(): pass",
            "top/test_b/__init__.py": "",
            "top/test_b/test_b.py": "def test_it(): pass",
            "top/test_c.py": "def test_it(): pass",
            "top/zzz/__init__.py": "",
            "top/zzz/test_zzz.py": "def test_it(): pass",
            "top/zzz/dir/test_dir.py": "def test_it(): pass",
        }
    )

    full = [
        "<Dir test_collection_hierarchy*>",
        "  <Dir top>",
        "    <Dir aaa>",
        "      <Package pkg>",
        "        <Module test_pkg.py>",
        "          <Function test_it>",
        "      <Module test_aaa.py>",
        "        <Function test_it>",
        "    <Module test_a.py>",
        "      <Function test_it>",
        "    <Package test_b>",
        "      <Module test_b.py>",
        "        <Function test_it>",
        "    <Module test_c.py>",
        "      <Function test_it>",
        "    <Package zzz>",
        "      <Dir dir>",
        "        <Module test_dir.py>",
        "          <Function test_it>",
        "      <Module test_zzz.py>",
        "        <Function test_it>",
    ]
    result = pytester.runpytest("--collect-only")
    result.stdout.fnmatch_lines(full, consecutive=True)
    result = pytester.runpytest("top", "--collect-only")
    result.stdout.fnmatch_lines(full, consecutive=True)
    result = pytester.runpytest("top", "top", "--collect-only")
    result.stdout.fnmatch_lines(full, consecutive=True)

    result = pytester.runpytest(
        "top/aaa", "top/aaa/pkg", "--collect-only", "--keep-duplicates"
    )
    result.stdout.fnmatch_lines(
        [
            "<Dir test_collection_hierarchy*>",
            "  <Dir top>",
            "    <Dir aaa>",
            "      <Package pkg>",
            "        <Module test_pkg.py>",
            "          <Function test_it>",
            "      <Module test_aaa.py>",
            "        <Function test_it>",
            "      <Package pkg>",
            "        <Module test_pkg.py>",
            "          <Function test_it>",
        ],
        consecutive=True,
    )

    result = pytester.runpytest(
        "top/aaa/pkg", "top/aaa", "--collect-only", "--keep-duplicates"
    )
    result.stdout.fnmatch_lines(
        [
            "<Dir test_collection_hierarchy*>",
            "  <Dir top>",
            "    <Dir aaa>",
            "      <Package pkg>",
            "        <Module test_pkg.py>",
            "          <Function test_it>",
            "          <Function test_it>",
            "      <Module test_aaa.py>",
            "        <Function test_it>",
        ],
        consecutive=True,
    )

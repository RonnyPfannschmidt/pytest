# mypy: allow-untyped-defs
from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import re
import sys
import types
from typing import cast
from unittest import mock

from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import RegisteredMarker
from _pytest.config import UsageError
from _pytest.ensemble import build_module
from _pytest.ensemble import collect_tests
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import configured
from _pytest.ensemble import run_tests
from _pytest.ensemble import RunRecord
from _pytest.mark import _validate_marker_names
from _pytest.mark import MarkGenerator
from _pytest.mark.expression import Expression
from _pytest.mark.structures import _EmptyParameterSetMark
from _pytest.mark.structures import EMPTY_PARAMETERSET_OPTION
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.nodes import Collector
from _pytest.nodes import Node
from _pytest.pytester import Pytester
import pytest


def ensemble_mark(name: str, *args: object, **kwargs: object) -> MarkDecorator:
    """Build a mark decorator without consulting the *host* configuration.

    ``pytest.mark.<name>`` validates the name against whatever config is
    active at decoration time.  Sources written inline in this file are
    decorated while pytest's own suite (which runs with ``strict = true``)
    is the active config, long before the ensemble they are meant to run in
    exists, so marker names that are only registered inside an ensemble --
    or deliberately not registered anywhere -- have to bypass ``MARK_GEN``.
    """
    return MarkDecorator(Mark(name, args, kwargs, _ispytest=True), _ispytest=True)


def passed_names(record: RunRecord) -> list[str]:
    """The bare names of the tests that passed, in collection order."""
    return [
        nodeid.split("::")[-1] for nodeid, item in record.by_test.items() if item.passed
    ]


class TestMark:
    @pytest.mark.parametrize("attr", ["mark", "param"])
    def test_pytest_exists_in_namespace_all(self, attr: str) -> None:
        module = sys.modules["pytest"]
        assert attr in module.__all__

    def test_pytest_mark_notcallable(self) -> None:
        mark = MarkGenerator(_ispytest=True)
        with pytest.raises(TypeError):
            mark()  # type: ignore[operator]

    def test_mark_with_param(self):
        def some_function(abc):
            pass

        class SomeClass:
            pass

        assert pytest.mark.foo(some_function) is some_function
        marked_with_args = pytest.mark.foo.with_args(some_function)
        assert marked_with_args is not some_function

        assert pytest.mark.foo(SomeClass) is SomeClass
        assert pytest.mark.foo.with_args(SomeClass) is not SomeClass  # type: ignore[comparison-overlap]

    def test_pytest_mark_name_starts_with_underscore(self) -> None:
        mark = MarkGenerator(_ispytest=True)
        with pytest.raises(AttributeError):
            _ = mark._some_name


# ensemble: the subject is argument-level deduplication of the same file passed
# twice; ensemble collection is preset and has no path arguments to dedupe.
def test_marked_class_run_twice(pytester: Pytester) -> None:
    """Test fails file is run twice that contains marked class.
    See issue#683.
    """
    py_file = pytester.makepyfile(
        """
    import pytest
    @pytest.mark.parametrize('abc', [1, 2, 3])
    class Test1(object):
        def test_1(self, abc):
            assert abc in [1, 2, 3]
    """
    )
    file_name = os.path.basename(py_file)
    rec = pytester.inline_run("--keep-duplicates", file_name, file_name)
    rec.assertoutcome(passed=6)


def test_ini_markers(tmp_path: Path) -> None:
    def test_markers(pytestconfig):
        markers = pytestconfig.getini("markers")
        print(markers)
        assert len(markers) >= 2
        assert markers[0].startswith("a1:")
        assert markers[1].startswith("a2:")

    spec = ConfigSpec(
        rootpath=tmp_path,
        inicfg={
            "markers": [
                "a1: this is a webtest marker",
                "a2: this is a smoke marker",
            ]
        },
    )
    run_tests(test_markers, spec=spec).assert_outcomes(passed=1)


# ensemble: --markers is served from pytest_cmdline_main, which an ensemble
# (which starts from an already-parsed config) never reaches.
def test_markers_option(pytester: Pytester) -> None:
    pytester.makeini(
        """
        [pytest]
        markers =
            a1: this is a webtest marker
            a1some: another marker
            nodescription
    """
    )
    result = pytester.runpytest("--markers")
    result.stdout.fnmatch_lines(
        ["*a1*this is a webtest*", "*a1some*another marker", "*nodescription*"]
    )


def test_ini_markers_whitespace(tmp_path: Path) -> None:
    @ensemble_mark("a1")
    def test_markers():
        assert True

    spec = ConfigSpec(
        rootpath=tmp_path,
        inicfg={"markers": ["a1 : this is a whitespace marker"]},
        args=("--strict-markers", "-m", "a1"),
    )
    # --strict-markers also validates the '-m' expression against the
    # registered names, so this fails loudly if the whitespace is not stripped.
    run_tests(test_markers, spec=spec).assert_outcomes(passed=1)


# ensemble: needs a setup.cfg on disk and a conftest scoped to a subdirectory.
def test_marker_without_description(pytester: Pytester) -> None:
    pytester.makefile(
        ".cfg",
        setup="""
        [tool:pytest]
        markers=slow
    """,
    )
    pytester.makeconftest(
        """
        import pytest
        pytest.mark.xfail('FAIL')
    """
    )
    ftdir = pytester.mkdir("ft1_dummy")
    pytester.path.joinpath("conftest.py").replace(ftdir.joinpath("conftest.py"))
    rec = pytester.runpytest("--strict-markers")
    rec.assert_outcomes()


# ensemble: --markers again, plus a conftest that loads a plugin by module name
# from the current directory.
def test_markers_option_with_plugin_in_current_dir(pytester: Pytester) -> None:
    pytester.makeconftest('pytest_plugins = "flip_flop"')
    pytester.makepyfile(
        flip_flop="""\
        def pytest_configure(config):
            config.addinivalue_line("markers", "flip:flop")

        def pytest_generate_tests(metafunc):
            try:
                mark = metafunc.function.flipper
            except AttributeError:
                return
            metafunc.parametrize("x", (10, 20))"""
    )
    pytester.makepyfile(
        """\
        import pytest
        @pytest.mark.flipper
        def test_example(x):
            assert x"""
    )

    result = pytester.runpytest("--markers")
    result.stdout.fnmatch_lines(["*flip*flop*"])


def test_mark_on_pseudo_function(tmp_path: Path) -> None:
    @ensemble_mark("r", lambda x: 0 / 0)
    def test_hello():
        pass

    run_tests(test_hello, rootpath=tmp_path).assert_outcomes(passed=1)


# ensemble: the whole point is a ``@pytest.mark.unregisteredmark`` decorator
# resolved against the config under test; decorators on sources written here
# resolve against the *host* config at decoration time instead. The enforcement
# itself is covered through '-m' expression validation below.
@pytest.mark.parametrize(
    "option",
    [
        "--strict-markers",
        "--strict",
        "strict_markers = true",
        "strict = true",
        "addopts = --strict-markers",
    ],
)
def test_strict_prohibits_unregistered_markers(pytester: Pytester, option: str) -> None:
    pytester.makepyfile(
        """
        import pytest
        @pytest.mark.unregisteredmark
        def test_hello():
            pass
    """
    )
    if option.startswith("-"):
        result = pytester.runpytest(option)
    else:
        pytester.makeini(
            f"""
            [pytest]
            {option}
            """
        )
        result = pytester.runpytest()
    assert result.ret != 0
    result.stdout.fnmatch_lines(
        ["'unregisteredmark' not found in `markers` configuration option"]
    )


class TestValidateMarkerNames:
    """Tests for _validate_marker_names (issue #2781)."""

    class FakeConfig:
        def __init__(
            self,
            markers: list[str],
            strict_markers: bool | None = None,
            strict: bool = False,
        ) -> None:
            self._ini: dict[str, list[str] | bool | None] = {
                "markers": markers,
                "strict_markers": strict_markers,
                "strict": strict,
            }

        def getini(self, name: str) -> list[str] | bool | None:
            return self._ini[name]

        def _iter_registered_markers(self) -> Iterator[RegisteredMarker]:
            yield from Config._iter_registered_markers(cast(Config, self))

    def _make_config(
        self,
        strict_markers: bool | None = None,
        strict: bool = False,
    ) -> Config:
        return cast(
            Config,
            self.FakeConfig(
                markers=["registered: a registered marker"],
                strict_markers=strict_markers,
                strict=strict,
            ),
        )

    def test_unknown_marker_with_strict_markers(self) -> None:
        expr = Expression.compile("unknown_marker")

        with pytest.raises(UsageError, match=r"Unknown marker.*unknown_marker"):
            _validate_marker_names(expr, self._make_config(strict_markers=True))

    def test_unknown_marker_with_strict(self) -> None:
        expr = Expression.compile("unknown_marker")

        with pytest.raises(UsageError, match=r"Unknown marker.*unknown_marker"):
            _validate_marker_names(expr, self._make_config(strict=True))

    def test_registered_marker_passes(self) -> None:
        expr = Expression.compile("registered")

        _validate_marker_names(expr, self._make_config(strict_markers=True))

    def test_no_validation_without_strict(self) -> None:
        expr = Expression.compile("any_marker")

        _validate_marker_names(expr, self._make_config())


@pytest.fixture
def markexpr_module() -> types.ModuleType:
    @ensemble_mark("registered")
    def test_registered():
        pass

    def test_plain():
        pass

    return build_module("test_markexpr", test_registered, test_plain)


@pytest.fixture
def markexpr_spec(tmp_path: Path) -> ConfigSpec:
    return ConfigSpec(
        rootpath=tmp_path,
        inicfg={"markers": ["registered: a registered marker"]},
    )


@pytest.mark.parametrize("option", ["--strict-markers", "--strict"])
def test_strict_prohibits_unregistered_markers_in_markexpr(
    markexpr_module: types.ModuleType, markexpr_spec: ConfigSpec, option: str
) -> None:
    spec = markexpr_spec.replace(args=(option, "-m", "registered or unregisteredmark"))
    with pytest.raises(
        UsageError,
        match=re.escape("Unknown marker(s) in '-m' expression: unregisteredmark"),
    ):
        run_tests(markexpr_module, spec=spec)


def test_strict_allows_registered_markers_in_markexpr(
    markexpr_module: types.ModuleType, markexpr_spec: ConfigSpec
) -> None:
    spec = markexpr_spec.replace(args=("--strict-markers", "-m", "registered"))
    run_tests(markexpr_module, spec=spec).assert_outcomes(passed=1, deselected=1)


def test_unregistered_markers_in_markexpr_allowed_without_strict(
    markexpr_module: types.ModuleType, markexpr_spec: ConfigSpec
) -> None:
    spec = markexpr_spec.replace(args=("-m", "unregisteredmark"))
    run_tests(markexpr_module, spec=spec).assert_outcomes(deselected=2)


@pytest.mark.parametrize(
    ("expr", "expected_passed"),
    [
        ("xyz", ["test_one"]),
        ("(((  xyz))  )", ["test_one"]),
        ("not not xyz", ["test_one"]),
        ("xyz and xyz2", []),
        ("xyz2", ["test_two"]),
        ("xyz or xyz2", ["test_one", "test_two"]),
    ],
)
def test_mark_option(
    expr: str, expected_passed: list[str | None], tmp_path: Path
) -> None:
    @ensemble_mark("xyz")
    def test_one():
        pass

    @ensemble_mark("xyz2")
    def test_two():
        pass

    record = run_tests(
        test_one, test_two, rootpath=tmp_path, spec=ConfigSpec(args=("-m", expr))
    )
    assert passed_names(record) == expected_passed


@pytest.mark.parametrize(
    ("expr", "expected_passed"),
    [
        ("car(color='red')", ["test_one"]),
        ("car(color='red') or car(color='blue')", ["test_one", "test_two"]),
        ("car and not car(temp=5)", ["test_one", "test_three"]),
        ("car(temp=4)", ["test_one"]),
        ("car(temp=4) or car(temp=5)", ["test_one", "test_two"]),
        ("car(temp=4) and car(temp=5)", []),
        ("car(temp=-5)", ["test_three"]),
        ("car(ac=True)", ["test_one"]),
        ("car(ac=False)", ["test_two"]),
        ("car(ac=None)", ["test_three"]),  # test NOT_NONE_SENTINEL
    ],
    ids=str,
)
def test_mark_option_with_kwargs(
    expr: str, expected_passed: list[str | None], tmp_path: Path
) -> None:
    @ensemble_mark("car")
    @ensemble_mark("car", ac=True)
    @ensemble_mark("car", temp=4)
    @ensemble_mark("car", color="red")
    def test_one():
        pass

    @ensemble_mark("car")
    @ensemble_mark("car", ac=False)
    @ensemble_mark("car", temp=5)
    @ensemble_mark("car", color="blue")
    def test_two():
        pass

    @ensemble_mark("car")
    @ensemble_mark("car", ac=None)
    @ensemble_mark("car", temp=-5)
    def test_three():
        pass

    record = run_tests(
        test_one,
        test_two,
        test_three,
        rootpath=tmp_path,
        spec=ConfigSpec(args=("-m", expr)),
    )
    assert passed_names(record) == expected_passed


@pytest.mark.parametrize(
    ("expr", "expected_passed"),
    [("interface", ["test_interface"]), ("not interface", ["test_nointer"])],
)
def test_mark_option_custom(
    expr: str, expected_passed: list[str], tmp_path: Path
) -> None:
    class AddInterfaceMarker:
        def pytest_collection_modifyitems(self, items):
            for item in items:
                if "interface" in item.nodeid:
                    item.add_marker(ensemble_mark("interface"))

    def test_interface():
        pass

    def test_nointer():
        pass

    spec = ConfigSpec(args=("-m", expr), extra_plugins=(AddInterfaceMarker(),))
    record = run_tests(
        test_interface,
        test_nointer,
        rootpath=tmp_path,
        spec=spec,
        name="test_mark_option_custom",
    )
    assert passed_names(record) == expected_passed


@pytest.mark.parametrize(
    ("expr", "expected_passed"),
    [
        ("interface", ["test_interface"]),
        ("not interface", ["test_nointer", "test_pass", "test_1", "test_2"]),
        ("pass", ["test_pass"]),
        ("not pass", ["test_interface", "test_nointer", "test_1", "test_2"]),
        ("not not not (pass)", ["test_interface", "test_nointer", "test_1", "test_2"]),
        ("1 or 2", ["test_1", "test_2"]),
        ("not (1 or 2)", ["test_interface", "test_nointer", "test_pass"]),
    ],
)
def test_keyword_option_custom(
    expr: str, expected_passed: list[str], tmp_path: Path
) -> None:
    def test_interface():
        pass

    def test_nointer():
        pass

    def test_pass():
        pass

    def test_1():
        pass

    def test_2():
        pass

    # the synthesized module name is matched by -k just like a real one, so it
    # is chosen to contain none of the expressions under test.
    record = run_tests(
        test_interface,
        test_nointer,
        test_pass,
        test_1,
        test_2,
        rootpath=tmp_path,
        spec=ConfigSpec(args=("-k", expr)),
        name="test_keyword_custom",
    )
    assert passed_names(record) == expected_passed


def test_keyword_option_considers_mark(tmp_path: Path) -> None:
    @pytest.mark.foo
    def test_mark():
        pass

    def test_unmarked():
        pass

    record = run_tests(
        test_mark,
        test_unmarked,
        rootpath=tmp_path,
        spec=ConfigSpec(args=("-k", "foo")),
        name="test_marks_as_keywords",
    )
    record.assert_outcomes(passed=1, deselected=1)
    assert passed_names(record) == ["test_mark"]


@pytest.mark.parametrize(
    ("expr", "expected_passed"),
    [
        ("None", ["test_func[None]"]),
        ("[1.3]", ["test_func[1.3]"]),
        ("2-3", ["test_func[2-3]"]),
    ],
)
def test_keyword_option_parametrize(
    expr: str, expected_passed: list[str], tmp_path: Path
) -> None:
    @pytest.mark.parametrize("arg", [None, 1.3, "2-3"])
    def test_func(arg):
        pass

    record = run_tests(
        test_func,
        rootpath=tmp_path,
        spec=ConfigSpec(args=("-k", expr)),
        name="test_keyword_parametrize",
    )
    assert passed_names(record) == expected_passed


def test_parametrize_with_module(tmp_path: Path) -> None:
    @pytest.mark.parametrize("arg", [pytest])
    def test_func(arg):
        pass

    record = run_tests(test_func, rootpath=tmp_path)
    record.assert_outcomes(passed=1)
    expected_id = "test_func[" + pytest.__name__ + "]"
    assert passed_names(record) == [expected_id]


@pytest.mark.parametrize(
    ("expr", "expected_error"),
    [
        (
            "foo or",
            "at column 7: expected not OR left parenthesis OR identifier; got end of input",
        ),
        (
            "foo or or",
            "at column 8: expected not OR left parenthesis OR identifier; got or",
        ),
        (
            "(foo",
            "at column 5: expected right parenthesis; got end of input",
        ),
        (
            "foo bar",
            "at column 5: expected end of input; got identifier",
        ),
        (
            "or or",
            "at column 1: expected not OR left parenthesis OR identifier; got or",
        ),
        (
            "not or",
            "at column 5: expected not OR left parenthesis OR identifier; got or",
        ),
        (
            "nonexistent_mark(non_supported='kwarg')",
            "Keyword expressions do not support call parameters",
        ),
    ],
)
def test_keyword_option_wrong_arguments(
    expr: str, expected_error: str, tmp_path: Path
) -> None:
    def test_func(arg):
        pass

    with pytest.raises(UsageError, match=re.escape(expected_error)):
        run_tests(test_func, rootpath=tmp_path, spec=ConfigSpec(args=("-k", expr)))


# ensemble: the subject is selecting a parametrized test by a "file.py::name"
# command line argument; ensemble collection is preset, not argument driven.
def test_parametrized_collected_from_command_line(pytester: Pytester) -> None:
    """Parametrized test not collected if test named specified in command
    line issue#649."""
    py_file = pytester.makepyfile(
        """
        import pytest
        @pytest.mark.parametrize("arg", [None, 1.3, "2-3"])
        def test_func(arg):
            pass
    """
    )
    file_name = os.path.basename(py_file)
    rec = pytester.inline_run(file_name + "::" + "test_func")
    rec.assertoutcome(passed=3)


def test_parametrized_collect_with_wrong_args(tmp_path: Path) -> None:
    """Test collect parametrized func with wrong number of args."""

    @pytest.mark.parametrize("foo, bar", [(1, 2, 3)])
    def test_func(foo, bar):
        pass

    module = build_module("test_parametrized_collect_with_wrong_args", test_func)
    record = run_tests(module, rootpath=tmp_path, capture_output=True)
    record.assert_outcomes(errors=1)
    record.stdout.fnmatch_lines(
        [
            'test_parametrized_collect_with_wrong_args.py::test_func: in "parametrize" the number of names (2):',
            "  ['foo', 'bar']",
            "must be equal to the number of values (3):",
            "  (1, 2, 3)",
        ]
    )


def test_parametrized_with_kwargs(tmp_path: Path) -> None:
    """Test collect parametrized func with wrong number of args."""

    @pytest.fixture(params=[1, 2])
    def a(request):
        return request.param

    @pytest.mark.parametrize(argnames="b", argvalues=[1, 2])
    def test_func(a, b):
        pass

    run_tests(a, test_func, rootpath=tmp_path).assert_outcomes(passed=4)


def test_parametrize_iterator(tmp_path: Path) -> None:
    """`parametrize` should work with generators (#5354)."""

    def gen():
        yield 1
        yield 2
        yield 3

    @pytest.mark.parametrize("a", gen())
    def test(a):
        assert a >= 1

    # should not skip any tests
    run_tests(test, rootpath=tmp_path).assert_outcomes(passed=3)


class TestFunctional:
    def test_merging_markers_deep(self, tmp_path: Path) -> None:
        # issue 199 - propagate markers into nested classes
        class TestA:
            pytestmark = ensemble_mark("a")

            def test_b(self):
                assert True

            class TestC:
                # this one didn't get marked
                def test_d(self):
                    assert True

        items = collect_tests(TestA, rootpath=tmp_path)
        assert len(items) == 2
        for item in items:
            print(item, item.keywords)
            assert [x for x in item.iter_markers() if x.name == "a"]

    def test_mark_decorator_subclass_does_not_propagate_to_base(
        self, tmp_path: Path
    ) -> None:
        @ensemble_mark("a")
        class Base:
            pass

        @ensemble_mark("b")
        class Test1(Base):
            def test_foo(self):
                pass

        class Test2(Base):
            def test_bar(self):
                pass

        items = collect_tests(Test1, Test2, rootpath=tmp_path)
        self.assert_markers(items, test_foo=("a", "b"), test_bar=("a",))

    def test_mark_should_not_pass_to_siebling_class(self, tmp_path: Path) -> None:
        """#568"""

        class TestBase:
            def test_foo(self):
                pass

        @ensemble_mark("b")
        class TestSub(TestBase):
            pass

        class TestOtherSub(TestBase):
            pass

        items = collect_tests(TestBase, TestSub, TestOtherSub, rootpath=tmp_path)
        base_item, sub_item, sub_item_other = items
        print(items, [x.nodeid for x in items])
        # new api segregates
        assert not list(base_item.iter_markers(name="b"))
        assert not list(sub_item_other.iter_markers(name="b"))
        assert list(sub_item.iter_markers(name="b"))

    def test_mark_decorator_baseclasses_merged(self, tmp_path: Path) -> None:
        @ensemble_mark("a")
        class Base:
            pass

        @ensemble_mark("b")
        class Base2(Base):
            pass

        @ensemble_mark("c")
        class Test1(Base2):
            def test_foo(self):
                pass

        class Test2(Base2):
            @ensemble_mark("d")
            def test_bar(self):
                pass

        items = collect_tests(Test1, Test2, rootpath=tmp_path)
        self.assert_markers(items, test_foo=("a", "b", "c"), test_bar=("a", "b", "d"))

    def test_mark_closest(self, tmp_path: Path) -> None:
        @ensemble_mark("c", location="class")
        class Test:
            @ensemble_mark("c", location="function")
            def test_has_own(self):
                pass

            def test_has_inherited(self):
                pass

        has_own, has_inherited = collect_tests(Test, rootpath=tmp_path)
        has_own_marker = has_own.get_closest_marker("c")
        has_inherited_marker = has_inherited.get_closest_marker("c")
        assert has_own_marker is not None
        assert has_inherited_marker is not None
        assert has_own_marker.kwargs == {"location": "function"}
        assert has_inherited_marker.kwargs == {"location": "class"}
        assert has_own.get_closest_marker("missing") is None

    def test_mark_closest_default_mark_decorator(self, tmp_path: Path) -> None:
        def test_without_mark():
            pass

        (item,) = collect_tests(test_without_mark, rootpath=tmp_path)
        default = pytest.mark.foo(location="default")
        assert item.get_closest_marker("foo", default) is default.mark

    def test_mark_with_wrong_marker(self, tmp_path: Path) -> None:
        class pytestmark:
            pass

        def test_func():
            pass

        module = build_module("test_wrong_marker", test_func, pytestmark=pytestmark)
        record = run_tests(module, rootpath=tmp_path)
        (value,) = record.collect_errors
        assert "TypeError" in str(value.longrepr)

    def test_mark_dynamically_in_funcarg(self, tmp_path: Path) -> None:
        class ArgPlugin:
            @pytest.fixture
            def arg(self, request):
                request.applymarker(ensemble_mark("hello"))

        def test_func(arg):
            pass

        spec = ConfigSpec(extra_plugins=(ArgPlugin(),))
        record = run_tests(test_func, rootpath=tmp_path, spec=spec)
        record.assert_outcomes(passed=1)
        # the original scraped this off the terminal summary's report keywords
        call = record["test_func"].call
        assert call is not None
        assert "hello" in call.keywords

    def test_no_marker_match_on_unmarked_names(self, tmp_path: Path) -> None:
        @ensemble_mark("shouldmatch")
        def test_marked():
            assert 1

        def test_unmarked():
            assert 1

        record = run_tests(
            test_marked,
            test_unmarked,
            rootpath=tmp_path,
            spec=ConfigSpec(args=("-m", "test_unmarked")),
        )
        record.assert_outcomes(deselected=2)

    def test_keywords_at_node_level(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="session", autouse=True)
        def some(request):
            request.keywords["hello"] = 42
            assert "world" not in request.keywords

        @pytest.fixture(scope="function", autouse=True)
        def funcsetup(request):
            assert "world" in request.keywords
            assert "hello" in request.keywords

        @ensemble_mark("world")
        def test_function():
            pass

        record = run_tests(some, funcsetup, test_function, rootpath=tmp_path)
        record.assert_outcomes(passed=1)

    def test_keyword_added_for_session(self, tmp_path: Path) -> None:
        class SessionMarkerPlugin:
            def pytest_collection_modifyitems(self, session):
                session.add_marker("mark1")
                session.add_marker(ensemble_mark("mark2"))
                session.add_marker(ensemble_mark("mark3"))
                with pytest.raises(ValueError):
                    session.add_marker(10)

        def test_some(request):
            assert "mark1" in request.keywords
            assert "mark2" in request.keywords
            assert "mark3" in request.keywords
            assert 10 not in request.keywords
            marker = request.node.get_closest_marker("mark1")
            assert marker.name == "mark1"
            assert marker.args == ()
            assert marker.kwargs == {}

        spec = ConfigSpec(args=("-m", "mark1"), extra_plugins=(SessionMarkerPlugin(),))
        record = run_tests(test_some, rootpath=tmp_path, spec=spec)
        record.assert_outcomes(passed=1)

    def assert_markers(self, items, **expected) -> None:
        """Assert that given items have expected marker names applied to them.
        expected should be a dict of (item name -> seq of expected marker names).

        Note: this could be moved to ``pytester`` if proven to be useful
        to other modules.
        """
        items = {x.name: x for x in items}
        for name, expected_markers in expected.items():
            markers = {m.name for m in items[name].iter_markers()}
            assert markers == set(expected_markers)

    @pytest.mark.filterwarnings("ignore")
    def test_mark_from_parameters(self, tmp_path: Path) -> None:
        """#1540"""
        # skipifs inside fixture params
        params = [pytest.mark.skipif(False, reason="dont skip")("parameter")]

        @pytest.fixture(params=params)
        def parameter(request):
            return request.param

        def test_1(parameter):
            assert True

        module = build_module(
            "test_mark_from_parameters",
            parameter,
            test_1,
            pytestmark=pytest.mark.skipif(True, reason="skip all"),
        )
        run_tests(module, rootpath=tmp_path).assert_outcomes(skipped=1)

    # ensemble: string skipif conditions are evaluated in the *host* module's
    # globals (in-memory sources keep this file's __globals__), so the two
    # modules cannot each supply their own ``skip`` name.
    def test_reevaluate_dynamic_expr(self, pytester: Pytester) -> None:
        """#7360"""
        py_file1 = pytester.makepyfile(
            test_reevaluate_dynamic_expr1="""
            import pytest

            skip = True

            @pytest.mark.skipif("skip")
            def test_should_skip():
                assert True
        """
        )
        py_file2 = pytester.makepyfile(
            test_reevaluate_dynamic_expr2="""
            import pytest

            skip = False

            @pytest.mark.skipif("skip")
            def test_should_not_skip():
                assert True
        """
        )

        file_name1 = os.path.basename(py_file1)
        file_name2 = os.path.basename(py_file2)
        reprec = pytester.inline_run(file_name1, file_name2)
        reprec.assertoutcome(passed=1, skipped=1)


class TestKeywordSelection:
    def test_select_simple(self, tmp_path: Path) -> None:
        def test_one():
            assert 0

        class TestClass:
            def test_method_one(self):
                # deliberately false; the source is meant to fail
                assert 42 == 43  # type: ignore[comparison-overlap]

        def check(keyword, name):
            record = run_tests(
                test_one,
                TestClass,
                rootpath=tmp_path,
                spec=ConfigSpec(args=("-k", keyword)),
                name="test_simple_selection",
            )
            record.assert_outcomes(failed=1, deselected=1)
            (nodeid,) = record.by_test
            assert nodeid.split("::")[-1] == name

        for keyword in ["test_one", "est_on"]:
            check(keyword, "test_one")
        check("TestClass and test", "test_method_one")

    @pytest.mark.parametrize(
        "keyword",
        [
            "xxx",
            "xxx and test_2",
            "TestClass",
            "xxx and not test_1",
            "TestClass and test_2",
            "xxx and TestClass and test_2",
        ],
    )
    def test_select_extra_keywords(self, tmp_path: Path, keyword) -> None:
        def test_1():
            pass

        class TestClass:
            def test_2(self):
                pass

        class ExtraKeywordsPlugin:
            @pytest.hookimpl(wrapper=True)
            def pytest_pycollect_makeitem(self, name):
                item = yield
                if name == "TestClass":
                    item.extra_keyword_matches.add("xxx")
                return item

        class DeselectRecorder:
            def __init__(self) -> None:
                self.calls: list[list[object]] = []

            def pytest_deselected(self, items):
                self.calls.append(list(items))

        recorder = DeselectRecorder()
        spec = ConfigSpec(
            args=("-k", keyword),
            extra_plugins=(ExtraKeywordsPlugin(), recorder),
        )
        record = run_tests(
            test_1, TestClass, rootpath=tmp_path, spec=spec, name="test_select"
        )
        print("keyword", repr(keyword))
        record.assert_outcomes(passed=1)
        (nodeid,) = record.by_test
        assert nodeid.endswith("test_2")
        assert len(recorder.calls) == 1
        assert recorder.calls[0][0].name == "test_1"  # type: ignore[attr-defined]

    def test_keyword_extra(self, tmp_path: Path) -> None:
        def test_one():
            assert 0

        setattr(test_one, "mykeyword", True)

        record = run_tests(
            test_one, rootpath=tmp_path, spec=ConfigSpec(args=("-k", "mykeyword"))
        )
        record.assert_outcomes(failed=1)

    @pytest.mark.xfail
    def test_keyword_extra_dash(self, tmp_path: Path) -> None:
        def test_one():
            assert 0

        setattr(test_one, "mykeyword", True)

        # with argparse the argument to an option cannot
        # start with '-'
        record = run_tests(
            test_one, rootpath=tmp_path, spec=ConfigSpec(args=("-k", "-mykeyword"))
        )
        record.assert_outcomes()

    @pytest.mark.parametrize(
        "keyword",
        ["__", "+", ".."],
    )
    def test_no_magic_values(self, tmp_path: Path, keyword: str) -> None:
        """Make sure the tests do not match on magic values,
        no double underscored values, like '__dict__' and '+'.
        """

        def test_one():
            assert 1

        record = run_tests(
            test_one,
            rootpath=tmp_path,
            spec=ConfigSpec(args=("-k", keyword)),
            name="test_no_magic_values",
        )
        record.assert_outcomes(deselected=1)

    # ensemble: `-k` matching against directory names needs a Directory node
    # above the module, which preset in-memory collection has no equivalent for.
    def test_no_match_directories_outside_the_suite(
        self,
        pytester: Pytester,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`-k` should not match against directories containing the test suite (#7040)."""
        pytester.makefile(
            **{
                "suite/pytest": """[pytest]""",
            },
            ext=".ini",
        )
        pytester.makepyfile(
            **{
                "suite/ddd/tests/__init__.py": "",
                "suite/ddd/tests/test_foo.py": """
                def test_aaa(): pass
                def test_ddd(): pass
            """,
            }
        )
        monkeypatch.chdir(pytester.path / "suite")

        def get_collected_names(*args: str) -> list[str]:
            _, rec = pytester.inline_genitems(*args)
            calls = rec.getcalls("pytest_collection_finish")
            assert len(calls) == 1
            return [x.name for x in calls[0].session.items]

        # sanity check: collect both tests in normal runs
        assert get_collected_names() == ["test_aaa", "test_ddd"]

        # do not collect anything based on names outside the collection tree
        assert get_collected_names("-k", pytester._name) == []


class TestMarkDecorator:
    @pytest.mark.parametrize(
        "lhs, rhs, expected",
        [
            (pytest.mark.foo(), pytest.mark.foo(), True),
            (pytest.mark.foo(), pytest.mark.bar(), False),
            (pytest.mark.foo(), "bar", False),
            ("foo", pytest.mark.bar(), False),
        ],
    )
    def test__eq__(self, lhs, rhs, expected) -> None:
        assert (lhs == rhs) == expected

    def test_aliases(self) -> None:
        md = pytest.mark.foo(1, "2", three=3)
        assert md.name == "foo"
        assert md.args == (1, "2")
        assert md.kwargs == {"three": 3}


@pytest.mark.parametrize("mark", [None, "skip", "xfail"])
def test_parameterset_for_parametrize_marks(
    tmp_path: Path, mark: _EmptyParameterSetMark | None
) -> None:
    inicfg: dict[str, object] = {}
    if mark is not None:
        inicfg[EMPTY_PARAMETERSET_OPTION] = mark

    from _pytest.mark import get_empty_parameterset_mark

    # ``configured()`` has already run _pytest.mark's pytest_configure, which
    # is what the pytester version had to do by hand on a merely parsed config.
    with configured(ConfigSpec(rootpath=tmp_path, inicfg=inicfg)) as config:
        result_mark = get_empty_parameterset_mark(config, ["a"], all)
    if mark is None:
        # normalize to the default
        mark = "skip"
    assert result_mark.name == mark
    assert result_mark.kwargs["reason"].startswith("got empty parameter set ")
    if mark == "xfail":
        assert result_mark.kwargs.get("run") is False


def test_parameterset_for_parametrize_marks_invalid(tmp_path: Path) -> None:
    spec = ConfigSpec(rootpath=tmp_path, inicfg={EMPTY_PARAMETERSET_OPTION: "dontcare"})
    expected = (
        f"config option '{EMPTY_PARAMETERSET_OPTION}' expects one of "
        "'skip' | 'xfail' | 'fail_at_collect', got 'dontcare'"
    )
    with pytest.raises(UsageError, match=re.escape(expected)):
        with configured(spec):
            pass


# ensemble: asserts a host-anchored source line ("at line 3") in the rendered
# collection error, plus the INTERRUPTED exit code, neither of which an
# ensemble has.
def test_parameterset_for_fail_at_collect(pytester: Pytester) -> None:
    pytester.makeini(
        f"""
    [pytest]
    {EMPTY_PARAMETERSET_OPTION}=fail_at_collect
    """
    )

    config = pytester.parseconfig()
    from _pytest.mark import get_empty_parameterset_mark
    from _pytest.mark import pytest_configure

    pytest_configure(config)

    with pytest.raises(
        Collector.CollectError,
        match=r"Empty parameter set in 'pytest_configure' at line \d\d+",
    ):
        get_empty_parameterset_mark(config, ["a"], pytest_configure)

    p1 = pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parametrize("empty", [])
        def test():
            pass
        """
    )
    result = pytester.runpytest(str(p1))
    result.stdout.fnmatch_lines(
        [
            "collected 0 items / 1 error",
            "* ERROR collecting test_parameterset_for_fail_at_collect.py *",
            "Empty parameter set in 'test' at line 3",
            "*= 1 error in *",
        ]
    )
    assert result.ret == ExitCode.INTERRUPTED


def test_paramset_empty_no_idfunc(tmp_path: Path) -> None:
    """An empty parameter set should not call the user provided id function (#13031)."""

    def idfunc(value):
        raise ValueError()

    @pytest.mark.parametrize("param", [], ids=idfunc)
    def test(param):
        pass

    # collecting at all proves idfunc was never called: it would blow up.
    assert len(collect_tests(test, rootpath=tmp_path)) == 1
    record = run_tests(test, rootpath=tmp_path)
    record.assert_outcomes(skipped=1)
    (item_record,) = record.by_test.values()
    assert item_record.setup is not None
    assert "got empty parameter set for (param)" in item_record.setup.longreprtext


def test_mark_expressions_no_smear(tmp_path: Path) -> None:
    class BaseTests:
        def test_something(self):
            pass

    @ensemble_mark("FOO")
    class TestFooClass(BaseTests):
        pass

    @ensemble_mark("BAR")
    class TestBarClass(BaseTests):
        pass

    record = run_tests(
        TestFooClass,
        TestBarClass,
        rootpath=tmp_path,
        spec=ConfigSpec(args=("-m", "FOO")),
    )
    record.assert_outcomes(passed=1, deselected=1)

    # todo: fixed
    # keywords smear - expected behaviour
    # reprec_keywords = pytester.inline_run("-k", "FOO")
    # passed_k, skipped_k, failed_k = reprec_keywords.countoutcomes()
    # assert passed_k == 2
    # assert skipped_k == failed_k == 0


def test_addmarker_order(tmp_path: Path) -> None:
    session = mock.Mock()
    session.own_markers = []
    session.parent = None
    session.nodeid = ""
    session.path = tmp_path
    node = Node.from_parent(session, name="Test")
    node.add_marker("foo")
    node.add_marker("bar")
    node.add_marker("baz", append=False)
    extracted = [x.name for x in node.iter_markers()]
    assert extracted == ["baz", "foo", "bar"]


@pytest.mark.filterwarnings("ignore")
def test_markers_from_parametrize(tmp_path: Path) -> None:
    """#3605"""
    first_custom_mark = ensemble_mark("custom_marker")
    custom_mark = ensemble_mark("custom_mark")

    @pytest.fixture(autouse=True)
    def trigger(request):
        seen = list(request.node.iter_markers("custom_mark"))
        print(f"Custom mark {seen}")

    @custom_mark("custom mark non parametrized")
    def test_custom_mark_non_parametrized():
        print("Hey from test")

    @pytest.mark.parametrize(
        "obj_type",
        [
            first_custom_mark("first custom mark")("template"),
            pytest.param(  # Think this should be recommended way?
                "disk", marks=custom_mark("custom mark1")
            ),
            custom_mark("custom mark2")("vm"),  # Tried also this
        ],
    )
    def test_custom_mark_parametrized(obj_type):
        print("obj_type is:", obj_type)

    record = run_tests(
        trigger,
        test_custom_mark_non_parametrized,
        test_custom_mark_parametrized,
        rootpath=tmp_path,
    )
    record.assert_outcomes(passed=4)


def test_pytest_param_id_requires_string() -> None:
    with pytest.raises(TypeError) as excinfo:
        pytest.param(id=True)  # type: ignore[arg-type]
    (msg,) = excinfo.value.args
    expected = (
        "Expected id to be a string or a `pytest.HIDDEN_PARAM` sentinel, "
        "got <class 'bool'>: True"
    )
    assert msg == expected


@pytest.mark.parametrize("s", (None, "hello world"))
def test_pytest_param_id_allows_none_or_string(s) -> None:
    assert pytest.param(id=s)


@pytest.mark.parametrize("expr", ("NOT internal_err", "NOT (internal_err)", "bogus="))
def test_marker_expr_eval_failure_handling(tmp_path: Path, expr) -> None:
    @ensemble_mark("internal_err")
    def test_foo():
        pass

    expected = f"Wrong expression passed to '-m': {expr}: "
    with pytest.raises(UsageError, match=re.escape(expected)):
        run_tests(test_foo, rootpath=tmp_path, spec=ConfigSpec(args=("-m", expr)))


def test_mark_mro() -> None:
    xfail = pytest.mark.xfail

    @xfail("a")
    class A:
        pass

    @xfail("b")
    class B:
        pass

    @xfail("c")
    class C(A, B):
        pass

    from _pytest.mark.structures import get_unpacked_marks

    all_marks = get_unpacked_marks(C)

    assert all_marks == [xfail("b").mark, xfail("a").mark, xfail("c").mark]

    assert get_unpacked_marks(C, consider_mro=False) == [xfail("c").mark]


# @pytest.mark.issue("https://github.com/pytest-dev/pytest/issues/10447")
def test_mark_fixture_order_mro(tmp_path: Path):
    """This ensures we walk marks of the mro starting with the base classes
    the action at a distance fixtures are taken as minimal example from a real project

    """

    @pytest.fixture
    def add_attr1(request):
        request.instance.attr1 = object()

    @pytest.fixture
    def add_attr2(request):
        request.instance.attr2 = request.instance.attr1

    @pytest.mark.usefixtures("add_attr1")
    class Parent:
        pass

    @pytest.mark.usefixtures("add_attr2")
    class TestThings(Parent):
        def test_attrs(self):
            # both attributes are injected by the usefixtures fixtures above
            assert self.attr1 == self.attr2  # type: ignore[attr-defined]

    record = run_tests(add_attr1, add_attr2, TestThings, rootpath=tmp_path)
    record.assert_outcomes(passed=1)


def test_mark_parametrize_over_staticmethod(tmp_path: Path) -> None:
    """Check that applying marks works as intended on classmethods and staticmethods.

    Regression test for #12863.
    """

    class TestClass:
        @pytest.mark.parametrize("value", [1, 2])
        @classmethod
        def test_classmethod_wrapper(cls, value: int):
            assert value in [1, 2]

        @classmethod
        @pytest.mark.parametrize("value", [1, 2])
        def test_classmethod_wrapper_on_top(cls, value: int):
            assert value in [1, 2]

        @pytest.mark.parametrize("value", [1, 2])
        @staticmethod
        def test_staticmethod_wrapper(value: int):
            assert value in [1, 2]

        @staticmethod
        @pytest.mark.parametrize("value", [1, 2])
        def test_staticmethod_wrapper_on_top(value: int):
            assert value in [1, 2]

    record = run_tests(TestClass, rootpath=tmp_path)
    record.assert_outcomes(passed=8)


def test_fixture_disallow_on_marked_functions() -> None:
    """Test that applying @pytest.fixture to a marked function errors (#3364)."""
    with pytest.raises(
        pytest.fail.Exception,
        match=r"Marks cannot be applied to fixtures",
    ):

        @pytest.fixture
        @pytest.mark.parametrize("example", ["hello"])
        @pytest.mark.usefixtures("tmp_path")
        def foo():
            raise NotImplementedError()


def test_fixture_disallow_marks_on_fixtures() -> None:
    """Test that applying a mark to a fixture errors (#3364)."""
    with pytest.raises(
        pytest.fail.Exception,
        match=r"Marks cannot be applied to fixtures",
    ):

        @pytest.mark.parametrize("example", ["hello"])
        @pytest.mark.usefixtures("tmp_path")
        @pytest.fixture
        def foo():
            raise NotImplementedError()


def test_fixture_disallowed_between_marks() -> None:
    """Test that applying a mark to a fixture errors (#3364)."""
    with pytest.raises(
        pytest.fail.Exception,
        match=r"Marks cannot be applied to fixtures",
    ):

        @pytest.mark.parametrize("example", ["hello"])
        @pytest.fixture
        @pytest.mark.usefixtures("tmp_path")
        def foo():
            raise NotImplementedError()


def test_module_getattr_without_attributeerror(tmp_path: Path) -> None:
    """
    Test that a helpful warning is emitted when a module-level
    __getattr__ returns None instead of raising AttributeError.

    Regression test for https://github.com/pytest-dev/pytest/issues/8265
    """

    def __getattr__(key):
        # Bug: should raise AttributeError, but returns None
        return None

    def test_something():
        assert True

    module = build_module(
        "test_module_getattr", test_something, __getattr__=__getattr__
    )
    spec = ConfigSpec(
        rootpath=tmp_path, args=("-W", "always::pytest.PytestCollectionWarning")
    )
    # The module is buggy (__getattr__ returns None for all attributes),
    # so no tests are collected, but pytest should NOT crash with a TypeError -
    # which in an ensemble would surface as the exception escaping run_tests.
    record = run_tests(module, spec=spec)
    record.assert_outcomes()
    (warning,) = [
        w for w in record.warnings if w.category is pytest.PytestCollectionWarning
    ]
    assert "__getattr__" in str(warning.message)
    assert "returns None" in str(warning.message)
    assert "AttributeError" in str(warning.message)

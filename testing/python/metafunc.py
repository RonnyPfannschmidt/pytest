# mypy: allow-untyped-defs
from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Sequence
import dataclasses
import itertools
from pathlib import Path
import re
import sys
import textwrap
from typing import Any
from typing import cast
from typing import ClassVar

import hypothesis
from hypothesis import strategies

from _pytest import fixtures
from _pytest import python
from _pytest.compat import getfuncargnames
from _pytest.compat import NOTSET
from _pytest.ensemble import build_module
from _pytest.ensemble import collect_tests
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import run_tests
from _pytest.outcomes import fail
from _pytest.outcomes import Failed
from _pytest.pytester import Pytester
from _pytest.python import Function
from _pytest.python import IdMaker
from _pytest.scope import Scope
import pytest


_IDMAKER_INI_DEFAULTS: dict[str, object] = {
    "disable_test_id_escaping_and_forfeit_all_rights_to_community_support": False,
    "parametrize_long_str_id_strategy": "short",
    "strict_parametrization_ids": None,
    "strict": False,
}


class _IdMakerConfig:
    """Mock config for IdMaker tests.

    Only serves known ini keys — raises KeyError on unknown ones
    to catch accidental lookups for keys without proper defaults.
    """

    def __init__(self, overrides: dict[str, object] | None = None) -> None:
        self._ini: dict[str, object] = {**_IDMAKER_INI_DEFAULTS, **(overrides or {})}

    @property
    def hook(self) -> _IdMakerConfig:
        return self

    def pytest_make_parametrize_id(self, **kw: object) -> None:
        return None

    def getini(self, name: str) -> object:
        return self._ini[name]


class TestMetafunc:
    def Metafunc(self, func, config=None) -> python.Metafunc:
        # The unit tests of this class check if things work correctly
        # on the funcarg level, so we don't need a full blown
        # initialization.
        class FuncFixtureInfoMock:
            name2fixturedefs: dict[str, list[fixtures.FixtureDef[object]]] = {}

            def __init__(self, names):
                self.names_closure = names

        @dataclasses.dataclass
        class FixtureManagerMock:
            config: Any

        @dataclasses.dataclass
        class SessionMock:
            config: Any
            _fixturemanager: FixtureManagerMock
            nodeid: ClassVar = ""

        @dataclasses.dataclass
        class DefinitionMock(python.FunctionDefinition):
            _nodeid: str
            obj: object

        names = getfuncargnames(func)
        fixtureinfo: Any = FuncFixtureInfoMock(names)
        definition: Any = DefinitionMock._create(obj=func, _nodeid="mock::nodeid")
        definition._fixtureinfo = fixtureinfo
        definition.session = SessionMock(config, FixtureManagerMock({}))
        return python.Metafunc(definition, fixtureinfo, config, _ispytest=True)

    def test_no_funcargs(self) -> None:
        def function():
            pass

        metafunc = self.Metafunc(function)
        assert not metafunc.fixturenames
        repr(metafunc._calls)

    def test_function_basic(self) -> None:
        def func(arg1, arg2="qwe"):
            pass

        metafunc = self.Metafunc(func)
        assert len(metafunc.fixturenames) == 1
        assert "arg1" in metafunc.fixturenames
        assert metafunc.function is func
        assert metafunc.cls is None

    def test_parametrize_single_arg_trailing_comma(self) -> None:
        """Test that trailing comma in string argnames behaves like tuple argnames.

        Regression test for https://github.com/pytest-dev/pytest/issues/719

        When using a single argument with:
        - "arg" (string, no comma): argvalues is a list of values
        - "arg," (string, trailing comma): argvalues is a list of tuples (like tuple form)
        - ("arg",) (tuple): argvalues is a list of tuples
        """

        def func(arg):
            pass  # pragma: no cover

        scenarios = [("a",), ("b",)]

        # Tuple form: argvalues are tuples, unpacked to get the value
        metafunc = self.Metafunc(func)
        metafunc.parametrize(("arg",), scenarios)
        assert metafunc._calls[0].params == {"arg": "a"}
        assert metafunc._calls[1].params == {"arg": "b"}

        # String with trailing comma: should behave like tuple form
        metafunc = self.Metafunc(func)
        metafunc.parametrize("arg,", scenarios)
        assert metafunc._calls[0].params == {"arg": "a"}
        assert metafunc._calls[1].params == {"arg": "b"}

        # String without comma: argvalues are values directly (tuples are passed as-is)
        metafunc = self.Metafunc(func)
        metafunc.parametrize("arg", scenarios)
        assert metafunc._calls[0].params == {"arg": ("a",)}
        assert metafunc._calls[1].params == {"arg": ("b",)}

        # String without comma with plain values: values are used directly
        metafunc = self.Metafunc(func)
        metafunc.parametrize("arg", ["a", "b"])
        assert metafunc._calls[0].params == {"arg": "a"}
        assert metafunc._calls[1].params == {"arg": "b"}

    def test_parametrize_error(self) -> None:
        def func(x, y):
            pass

        metafunc = self.Metafunc(func)
        metafunc.parametrize("x", [1, 2])
        with pytest.raises(pytest.Collector.CollectError):
            metafunc.parametrize("x", [5, 6])
        with pytest.raises(pytest.Collector.CollectError):
            metafunc.parametrize("x", [5, 6])
        metafunc.parametrize("y", [1, 2])
        with pytest.raises(pytest.Collector.CollectError):
            metafunc.parametrize("y", [5, 6])
        with pytest.raises(pytest.Collector.CollectError):
            metafunc.parametrize("y", [5, 6])

        with pytest.raises(TypeError, match=r"^ids must be a callable or an iterable$"):
            metafunc.parametrize("y", [5, 6], ids=42)  # type: ignore[arg-type]

    def test_parametrize_error_iterator(self) -> None:
        def func(x):
            raise NotImplementedError()

        class Exc(Exception):
            def __repr__(self):
                return "Exc(from_gen)"

        def gen() -> Iterator[int | Exc | None]:
            yield 0
            yield None
            yield Exc()

        metafunc = self.Metafunc(func)
        # When the input is an iterator, only len(args) are taken,
        # so the bad Exc isn't reached.
        metafunc.parametrize("x", [1, 2], ids=gen())
        assert [(x.params, x.id) for x in metafunc._calls] == [
            ({"x": 1}, "0"),
            ({"x": 2}, "2"),
        ]
        with pytest.raises(
            fail.Exception,
            match=(
                r"In mock::nodeid: ids contains unsupported value Exc\(from_gen\) \(type: <class .*Exc'>\) at index 2. "
                r"Supported types are: .*"
            ),
        ):
            metafunc.parametrize("x", [1, 2, 3], ids=gen())

    def test_parametrize_bad_scope(self) -> None:
        def func(x):
            pass

        metafunc = self.Metafunc(func)
        with pytest.raises(
            fail.Exception,
            match=r"parametrize\(\) call in func got an unexpected scope value 'doggy'",
        ):
            metafunc.parametrize("x", [1], scope="doggy")  # type: ignore[arg-type]

    def test_parametrize_request_name(self) -> None:
        """Show proper error  when 'request' is used as a parameter name in parametrize (#6183)"""

        def func(request):
            raise NotImplementedError()

        metafunc = self.Metafunc(func)
        with pytest.raises(
            fail.Exception,
            match=r"'request' is a reserved name and cannot be used in @pytest.mark.parametrize",
        ):
            metafunc.parametrize("request", [1])

    def test_infer_parametrize_scope(self) -> None:
        """Unit test for _infer_parameterize_scope (#3941)."""
        from _pytest.python import _infer_parametrize_scope

        @dataclasses.dataclass
        class DummyFixtureDef:
            _scope: Scope

        fixtures_defs = cast(
            dict[str, Sequence[fixtures.FixtureDef[object]]],
            dict(
                session_fix=[DummyFixtureDef(Scope.Session)],
                package_fix=[DummyFixtureDef(Scope.Package)],
                module_fix=[DummyFixtureDef(Scope.Module)],
                class_fix=[DummyFixtureDef(Scope.Class)],
                func_fix=[DummyFixtureDef(Scope.Function)],
                mixed_fix=[DummyFixtureDef(Scope.Module), DummyFixtureDef(Scope.Class)],
            ),
        )

        # use arguments to determine narrow scope; the cause of the bug is that it would look on all
        # fixture defs given to the method
        def find_scope(argnames, indirect):
            return _infer_parametrize_scope(argnames, fixtures_defs, indirect=indirect)

        assert find_scope(["func_fix"], indirect=True) == Scope.Function
        assert find_scope(["class_fix"], indirect=True) == Scope.Class
        assert find_scope(["module_fix"], indirect=True) == Scope.Module
        assert find_scope(["package_fix"], indirect=True) == Scope.Package
        assert find_scope(["session_fix"], indirect=True) == Scope.Session

        assert find_scope(["class_fix", "func_fix"], indirect=True) == Scope.Function
        assert find_scope(["func_fix", "session_fix"], indirect=True) == Scope.Function
        assert find_scope(["session_fix", "class_fix"], indirect=True) == Scope.Class
        assert (
            find_scope(["package_fix", "session_fix"], indirect=True) == Scope.Package
        )
        assert find_scope(["module_fix", "session_fix"], indirect=True) == Scope.Module

        # when indirect is False or is not for all scopes, always use function
        assert (
            find_scope(["session_fix", "module_fix"], indirect=False) == Scope.Function
        )
        assert (
            find_scope(["session_fix", "module_fix"], indirect=["module_fix"])
            == Scope.Function
        )
        assert (
            find_scope(
                ["session_fix", "module_fix"], indirect=["session_fix", "module_fix"]
            )
            == Scope.Module
        )
        assert find_scope(["mixed_fix"], indirect=True) == Scope.Class

    def test_parametrize_and_id(self) -> None:
        def func(x, y):
            pass

        metafunc = self.Metafunc(func)

        metafunc.parametrize("x", [1, 2], ids=["basic", "advanced"])
        metafunc.parametrize("y", ["abc", "def"])
        ids = [x.id for x in metafunc._calls]
        assert ids == ["basic-abc", "basic-def", "advanced-abc", "advanced-def"]

    def test_parametrize_and_id_unicode(self) -> None:
        """Allow unicode strings for "ids" parameter in Python 2 (##1905)"""

        def func(x):
            pass

        metafunc = self.Metafunc(func)
        metafunc.parametrize("x", [1, 2], ids=["basic", "advanced"])
        ids = [x.id for x in metafunc._calls]
        assert ids == ["basic", "advanced"]

    def test_parametrize_with_wrong_number_of_ids(self) -> None:
        def func(x, y):
            pass

        metafunc = self.Metafunc(func)

        with pytest.raises(fail.Exception):
            metafunc.parametrize("x", [1, 2], ids=["basic"])

        with pytest.raises(fail.Exception):
            metafunc.parametrize(
                ("x", "y"), [("abc", "def"), ("ghi", "jkl")], ids=["one"]
            )

    def test_parametrize_ids_iterator_without_mark(self) -> None:
        def func(x, y):
            pass

        it = itertools.count()

        metafunc = self.Metafunc(func)
        metafunc.parametrize("x", [1, 2], ids=it)
        metafunc.parametrize("y", [3, 4], ids=it)
        ids = [x.id for x in metafunc._calls]
        assert ids == ["0-2", "0-3", "1-2", "1-3"]

        metafunc = self.Metafunc(func)
        metafunc.parametrize("x", [1, 2], ids=it)
        metafunc.parametrize("y", [3, 4], ids=it)
        ids = [x.id for x in metafunc._calls]
        assert ids == ["4-6", "4-7", "5-6", "5-7"]

    def test_parametrize_empty_list(self) -> None:
        """#510"""

        def func(y):
            pass

        class MockConfig:
            def getini(self, name):
                return "skip"

            @property
            def hook(self):
                return self

            def pytest_make_parametrize_id(self, **kw):
                pass

        metafunc = self.Metafunc(func, MockConfig())
        metafunc.parametrize("y", [])
        assert "skip" == metafunc._calls[0].marks[0].name

    def test_parametrize_with_userobjects(self) -> None:
        def func(x, y):
            pass

        metafunc = self.Metafunc(func)

        class A:
            pass

        metafunc.parametrize("x", [A(), A()])
        metafunc.parametrize("y", list("ab"))
        assert metafunc._calls[0].id == "x0-a"
        assert metafunc._calls[1].id == "x0-b"
        assert metafunc._calls[2].id == "x1-a"
        assert metafunc._calls[3].id == "x1-b"

    @hypothesis.given(strategies.text() | strategies.binary())
    @hypothesis.settings(
        deadline=400.0
    )  # very close to std deadline and CI boxes are not reliable in CPU power
    def test_idval_hypothesis(self, value) -> None:
        escaped = IdMaker([], [], None, None, None, None)._idval(value, "a", 6)
        assert isinstance(escaped, str)
        escaped.encode("ascii")

    def test_unicode_idval(self) -> None:
        """Test that Unicode strings outside the ASCII character set get
        escaped, using byte escapes if they're in that range or unicode
        escapes if they're not.

        """
        values = [
            ("", r""),
            ("ascii", r"ascii"),
            ("ação", r"a\xe7\xe3o"),
            ("josé@blah.com", r"jos\xe9@blah.com"),
            (
                r"δοκ.ιμή@παράδειγμα.δοκιμή",
                r"\u03b4\u03bf\u03ba.\u03b9\u03bc\u03ae@\u03c0\u03b1\u03c1\u03ac\u03b4\u03b5\u03b9\u03b3"
                r"\u03bc\u03b1.\u03b4\u03bf\u03ba\u03b9\u03bc\u03ae",
            ),
        ]
        for val, expected in values:
            assert (
                IdMaker([], [], None, None, None, None)._idval(val, "a", 6) == expected
            )

    def test_unicode_idval_with_config(self) -> None:
        """Unit test for expected behavior to obtain ids with
        disable_test_id_escaping_and_forfeit_all_rights_to_community_support
        option (#5294)."""
        option = "disable_test_id_escaping_and_forfeit_all_rights_to_community_support"

        values: list[tuple[str, Any, str]] = [
            ("ação", _IdMakerConfig({option: True}), "ação"),
            ("ação", _IdMakerConfig({option: False}), "a\\xe7\\xe3o"),
        ]
        for val, config, expected in values:
            actual = IdMaker([], [], None, None, config, None)._idval(val, "a", 6)
            assert actual == expected

    def test_bytes_idval(self) -> None:
        """Unit test for the expected behavior to obtain ids for parametrized
        bytes values: bytes objects are always escaped using "binary escape"."""
        values = [
            (b"", r""),
            (b"\xc3\xb4\xff\xe4", r"\xc3\xb4\xff\xe4"),
            (b"ascii", r"ascii"),
            ("αρά".encode(), r"\xce\xb1\xcf\x81\xce\xac"),
        ]
        for val, expected in values:
            assert (
                IdMaker([], [], None, None, None, None)._idval(val, "a", 6) == expected
            )

    def test_class_or_function_idval(self) -> None:
        """Unit test for the expected behavior to obtain ids for parametrized
        values that are classes or functions: their __name__."""

        class TestClass:
            pass

        def test_function():
            pass

        values = [(TestClass, "TestClass"), (test_function, "test_function")]
        for val, expected in values:
            assert (
                IdMaker([], [], None, None, None, None)._idval(val, "a", 6) == expected
            )

    def test_notset_idval(self) -> None:
        """Test that a NOTSET value (used by an empty parameterset) generates
        a proper ID.

        Regression test for #7686.
        """
        assert IdMaker([], [], None, None, None, None)._idval(NOTSET, "a", 0) == "a0"

    def test_idmaker_autoname(self) -> None:
        """#250"""
        result = IdMaker(
            ("a", "b"),
            [pytest.param("string", 1.0), pytest.param("st-ring", 2.0)],
            None,
            None,
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == ["string-1.0", "st-ring-2.0"]

        result = IdMaker(
            ("a", "b"),
            [pytest.param(object(), 1.0), pytest.param(object(), object())],
            None,
            None,
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == ["a0-1.0", "a1-b1"]
        # unicode mixing, issue250
        result = IdMaker(
            ("a", "b"), [pytest.param({}, b"\xc3\xb4")], None, None, None, None
        ).make_unique_parameterset_ids()
        assert result == ["a0-\\xc3\\xb4"]

    def test_idmaker_with_bytes_regex(self) -> None:
        result = IdMaker(
            ("a"), [pytest.param(re.compile(b"foo"))], None, None, None, None
        ).make_unique_parameterset_ids()
        assert result == ["foo"]

    def test_idmaker_native_strings(self) -> None:
        result = IdMaker(
            ("a", "b"),
            [
                pytest.param(1.0, -1.1),
                pytest.param(2, -202),
                pytest.param("three", "three hundred"),
                pytest.param(True, False),
                pytest.param(None, None),
                pytest.param(re.compile("foo"), re.compile("bar")),
                pytest.param(str, int),
                pytest.param(list("six"), [66, 66]),
                pytest.param({7}, set("seven")),
                pytest.param(tuple("eight"), (8, -8, 8)),
                pytest.param(b"\xc3\xb4", b"name"),
                pytest.param(b"\xc3\xb4", "other"),
                pytest.param(1.0j, -2.0j),
            ],
            None,
            None,
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == [
            "1.0--1.1",
            "2--202",
            "three-three hundred",
            "True-False",
            "None-None",
            "foo-bar",
            "str-int",
            "a7-b7",
            "a8-b8",
            "a9-b9",
            "\\xc3\\xb4-name",
            "\\xc3\\xb4-other",
            "1j-(-0-2j)",
        ]

    def test_idmaker_non_printable_characters(self) -> None:
        result = IdMaker(
            ("s", "n"),
            [
                pytest.param("\x00", 1),
                pytest.param("\x05", 2),
                pytest.param(b"\x00", 3),
                pytest.param(b"\x05", 4),
                pytest.param("\t", 5),
                pytest.param(b"\t", 6),
            ],
            None,
            None,
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == ["\\x00-1", "\\x05-2", "\\x00-3", "\\x05-4", "\\t-5", "\\t-6"]

    def test_idmaker_manual_ids_must_be_printable(self) -> None:
        result = IdMaker(
            ("s",),
            [
                pytest.param("x00", id="hello \x00"),
                pytest.param("x05", id="hello \x05"),
            ],
            None,
            None,
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == ["hello \\x00", "hello \\x05"]

    def test_idmaker_enum(self) -> None:
        enum = pytest.importorskip("enum")
        e = enum.Enum("Foo", "one, two")
        result = IdMaker(
            ("a", "b"), [pytest.param(e.one, e.two)], None, None, None, None
        ).make_unique_parameterset_ids()
        assert result == ["Foo.one-Foo.two"]

    def test_idmaker_idfn(self) -> None:
        """#351"""

        def ids(val: object) -> str | None:
            if isinstance(val, Exception):
                return repr(val)
            return None

        result = IdMaker(
            ("a", "b"),
            [
                pytest.param(10.0, IndexError()),
                pytest.param(20, KeyError()),
                pytest.param("three", [1, 2, 3]),
            ],
            ids,
            None,
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == ["10.0-IndexError()", "20-KeyError()", "three-b2"]

    def test_idmaker_idfn_unique_names(self) -> None:
        """#351"""

        def ids(val: object) -> str:
            return "a"

        result = IdMaker(
            ("a", "b"),
            [
                pytest.param(10.0, IndexError()),
                pytest.param(20, KeyError()),
                pytest.param("three", [1, 2, 3]),
            ],
            ids,
            None,
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == ["a-a0", "a-a1", "a-a2"]

    def test_idmaker_with_idfn_and_config(self) -> None:
        """Unit test for expected behavior to create ids with idfn and
        disable_test_id_escaping_and_forfeit_all_rights_to_community_support
        option (#5294).
        """
        option = "disable_test_id_escaping_and_forfeit_all_rights_to_community_support"

        values: list[tuple[Any, str]] = [
            (_IdMakerConfig({option: True}), "ação"),
            (_IdMakerConfig({option: False}), "a\\xe7\\xe3o"),
        ]
        for config, expected in values:
            result = IdMaker(
                ("a",),
                [pytest.param("string")],
                lambda _: "ação",
                None,
                config,
                None,
            ).make_unique_parameterset_ids()
            assert result == [expected]

    def test_idmaker_with_ids_and_config(self) -> None:
        """Unit test for expected behavior to create ids with ids and
        disable_test_id_escaping_and_forfeit_all_rights_to_community_support
        option (#5294).
        """
        option = "disable_test_id_escaping_and_forfeit_all_rights_to_community_support"

        values: list[tuple[Any, str]] = [
            (_IdMakerConfig({option: True}), "ação"),
            (_IdMakerConfig({option: False}), "a\\xe7\\xe3o"),
        ]
        for config, expected in values:
            result = IdMaker(
                ("a",), [pytest.param("string")], None, ["ação"], config, None
            ).make_unique_parameterset_ids()
            assert result == [expected]

    @pytest.mark.parametrize("id_style", ["short", "sha256", "legacy"])
    @pytest.mark.parametrize(
        "kind",
        [
            pytest.param("a", id="str"),
            pytest.param(b"a", id="bytes"),
        ],
    )
    def test_idmaker_long_string(self, id_style: str, kind: str | bytes) -> None:
        maker = IdMaker(
            "a",
            [pytest.param(kind * 1000)],
            None,
            None,
            cast(
                pytest.Config,
                _IdMakerConfig({"parametrize_long_str_id_strategy": id_style}),
            ),
            None,
        )

        res = maker.make_unique_parameterset_ids()
        expected = {
            "legacy": "a" * 1000,
            "short": "a0",
            "sha256": "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3",
        }
        assert res == [expected[id_style]]

    @pytest.mark.parametrize(
        "kind",
        [
            pytest.param("a", id="str"),
            pytest.param(b"a", id="bytes"),
        ],
    )
    def test_idmaker_long_string_disallow(self, kind: str | bytes) -> None:
        maker = IdMaker(
            "a",
            [pytest.param(kind * 1000)],
            None,
            None,
            cast(
                pytest.Config,
                _IdMakerConfig({"parametrize_long_str_id_strategy": "disallow"}),
            ),
            None,
        )

        with pytest.raises(Failed, match="too long for an auto-generated ID"):
            maker.make_unique_parameterset_ids()

    def test_idmaker_long_string_disallow_short_value(self) -> None:
        """The disallow strategy should pass through short values normally."""
        maker = IdMaker(
            "a",
            [pytest.param("short")],
            None,
            None,
            cast(
                pytest.Config,
                _IdMakerConfig({"parametrize_long_str_id_strategy": "disallow"}),
            ),
            None,
        )
        assert maker.make_unique_parameterset_ids() == ["short"]

    def test_idmaker_long_string_no_config(self) -> None:
        """Without config, defaults to 'short' strategy."""
        maker = IdMaker(
            "a",
            [pytest.param("x" * 200)],
            None,
            None,
            None,
            None,
        )
        assert maker.make_unique_parameterset_ids() == ["a0"]

    def test_idmaker_long_string_unknown_strategy(self) -> None:
        """An unknown strategy value should raise UsageError."""
        maker = IdMaker(
            "a",
            [pytest.param("x" * 200)],
            None,
            None,
            cast(
                pytest.Config,
                _IdMakerConfig({"parametrize_long_str_id_strategy": "bogus"}),
            ),
            None,
        )
        with pytest.raises(pytest.UsageError, match=r"Unknown.*bogus"):
            maker.make_unique_parameterset_ids()

    @pytest.mark.parametrize("id_style", ["short", "sha256", "disallow"])
    def test_idmaker_long_string_explicit_ids_unaffected(self, id_style: str) -> None:
        """Explicit ids=[...] with long strings should work regardless of strategy."""
        long_id = "x" * 200
        maker = IdMaker(
            "a",
            [pytest.param("value")],
            None,
            [long_id],
            cast(
                pytest.Config,
                _IdMakerConfig({"parametrize_long_str_id_strategy": id_style}),
            ),
            None,
        )
        result = maker.make_unique_parameterset_ids()
        assert result == [long_id]

    def test_idmaker_with_param_id_and_config(self) -> None:
        """Unit test for expected behavior to create ids with pytest.param(id=...) and
        disable_test_id_escaping_and_forfeit_all_rights_to_community_support
        option (#9037).
        """
        option = "disable_test_id_escaping_and_forfeit_all_rights_to_community_support"

        values: list[tuple[Any, str]] = [
            (_IdMakerConfig({option: True}), "ação"),
            (_IdMakerConfig({option: False}), "a\\xe7\\xe3o"),
        ]
        for config, expected in values:
            result = IdMaker(
                ("a",),
                [pytest.param("string", id="ação")],
                None,
                None,
                config,
                None,
            ).make_unique_parameterset_ids()
            assert result == [expected]

    def test_idmaker_duplicated_empty_str(self) -> None:
        """Regression test for empty strings parametrized more than once (#11563)."""
        result = IdMaker(
            ("a",), [pytest.param(""), pytest.param("")], None, None, None, None
        ).make_unique_parameterset_ids()
        assert result == ["0", "1"]

    def test_parametrize_ids_exception(self, tmp_path: Path) -> None:
        """An ids callable that raises reports which parameter it choked on."""

        def ids(arg):
            raise Exception("bad ids")

        @pytest.mark.parametrize("arg", ["a", "b"], ids=ids)
        def test_foo(arg):
            pass

        # ensemble: the module name is part of the reported nodeid.
        module = build_module("test_parametrize_ids_exception", test_foo)
        record = run_tests(module, rootpath=tmp_path, capture_output=True)
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            [
                "*Exception: bad ids",
                "*test_foo: error raised while trying to determine id of parameter 'arg' at position 0",
            ]
        )

    def test_parametrize_ids_returns_non_string(self, tmp_path: Path) -> None:
        def ids(d):
            return d

        @pytest.mark.parametrize("arg", ({1: 2}, {3, 4}), ids=ids)
        def test(arg):
            assert arg

        @pytest.mark.parametrize("arg", (1, 2.0, True), ids=ids)
        def test_int(arg):
            assert arg

        module = build_module("test_parametrize_ids_returns_non_string", test, test_int)
        record = run_tests(module, rootpath=tmp_path)
        assert list(record.by_test) == [
            "test_parametrize_ids_returns_non_string.py::test[arg0]",
            "test_parametrize_ids_returns_non_string.py::test[arg1]",
            "test_parametrize_ids_returns_non_string.py::test_int[1]",
            "test_parametrize_ids_returns_non_string.py::test_int[2.0]",
            "test_parametrize_ids_returns_non_string.py::test_int[True]",
        ]
        record.assert_outcomes(passed=5)

    def test_idmaker_with_ids(self) -> None:
        result = IdMaker(
            ("a", "b"),
            [pytest.param(1, 2), pytest.param(3, 4)],
            None,
            ["a", None],
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == ["a", "3-4"]

    def test_idmaker_with_paramset_id(self) -> None:
        result = IdMaker(
            ("a", "b"),
            [pytest.param(1, 2, id="me"), pytest.param(3, 4, id="you")],
            None,
            ["a", None],
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == ["me", "you"]

    def test_idmaker_with_ids_unique_names(self) -> None:
        result = IdMaker(
            ("a"),
            list(map(pytest.param, [1, 2, 3, 4, 5])),
            None,
            ["a", "a", "b", "c", "b"],
            None,
            None,
        ).make_unique_parameterset_ids()
        assert result == ["a0", "a1", "b0", "c", "b1"]

    def test_parametrize_indirect(self) -> None:
        """#714"""

        def func(x, y):
            pass

        metafunc = self.Metafunc(func)
        metafunc.parametrize("x", [1], indirect=True)
        metafunc.parametrize("y", [2, 3], indirect=True)
        assert len(metafunc._calls) == 2
        assert metafunc._calls[0].params == dict(x=1, y=2)
        assert metafunc._calls[1].params == dict(x=1, y=3)

    def test_parametrize_indirect_list(self) -> None:
        """#714"""

        def func(x, y):
            pass

        metafunc = self.Metafunc(func)
        metafunc.parametrize("x, y", [("a", "b")], indirect=["x"])
        assert metafunc._calls[0].params == dict(x="a", y="b")
        # Since `y` is a direct parameter, its DirectParamFixtureDef would
        # be registered.
        assert list(metafunc._arg2fixturedefs.keys()) == ["y"]

    def test_parametrize_indirect_list_all(self) -> None:
        """#714"""

        def func(x, y):
            pass

        metafunc = self.Metafunc(func)
        metafunc.parametrize("x, y", [("a", "b")], indirect=["x", "y"])
        assert metafunc._calls[0].params == dict(x="a", y="b")
        assert list(metafunc._arg2fixturedefs.keys()) == []

    def test_parametrize_indirect_list_empty(self) -> None:
        """#714"""

        def func(x, y):
            pass

        metafunc = self.Metafunc(func)
        metafunc.parametrize("x, y", [("a", "b")], indirect=[])
        assert metafunc._calls[0].params == dict(x="a", y="b")
        assert list(metafunc._arg2fixturedefs.keys()) == ["x", "y"]

    def test_parametrize_indirect_wrong_type(self) -> None:
        def func(x, y):
            pass

        metafunc = self.Metafunc(func)
        with pytest.raises(
            fail.Exception,
            match="In mock::nodeid: expected Sequence or boolean for indirect, got dict",
        ):
            metafunc.parametrize("x, y", [("a", "b")], indirect={})  # type: ignore[arg-type]

    def test_parametrize_positional_indirect_error(self) -> None:
        """`indirect` and later arguments are keyword-only, so that extra
        positional arguments (e.g. several argname strings, #8593) fail with
        an understandable TypeError rather than being silently interpreted."""

        def func(x, y):
            raise NotImplementedError()

        metafunc = self.Metafunc(func)
        with pytest.raises(TypeError, match="positional arguments"):
            metafunc.parametrize("x, y", [("a", "b")], ["x"])  # type: ignore[call-arg]

    def test_parametrize_indirect_list_functional(self, tmp_path: Path) -> None:
        """
        #714
        Test parametrization with 'indirect' parameter applied on
        particular arguments. As y is direct, its value should
        be used directly rather than being passed to the fixture y.
        """

        @pytest.fixture(scope="function")
        def x(request):
            return request.param * 3

        @pytest.fixture(scope="function")
        def y(request):
            return request.param * 2

        @pytest.mark.parametrize("x, y", [("a", "b")], indirect=["x"])
        def test_simple(x, y):
            assert len(x) == 3
            assert len(y) == 1

        record = run_tests(
            build_module(
                "test_parametrize_indirect_list_functional", x, y, test_simple
            ),
            rootpath=tmp_path,
        )
        assert list(record.by_test) == [
            "test_parametrize_indirect_list_functional.py::test_simple[a-b]"
        ]
        record.assert_outcomes(passed=1)

    def test_parametrize_indirect_list_error(self) -> None:
        """#714"""

        def func(x, y):
            pass

        metafunc = self.Metafunc(func)
        with pytest.raises(fail.Exception):
            metafunc.parametrize("x, y", [("a", "b")], indirect=["x", "z"])

    def test_parametrize_uses_no_fixture_error_indirect_false(
        self, tmp_path: Path
    ) -> None:
        """The 'uses no fixture' error tells the user at collection time
        that the parametrize data they've set up doesn't correspond to the
        fixtures in their test function, rather than silently ignoring this
        and letting the test potentially pass.

        #714
        """

        @pytest.mark.parametrize("x, y", [("a", "b")], indirect=False)
        def test_simple(x):
            assert len(x) == 3

        with pytest.raises(pytest.Collector.CollectError, match="uses no argument 'y'"):
            collect_tests(test_simple, rootpath=tmp_path)

    def test_parametrize_uses_no_fixture_error_indirect_true(
        self, tmp_path: Path
    ) -> None:
        """#714"""

        @pytest.fixture(scope="function")
        def x(request):
            return request.param * 3

        @pytest.fixture(scope="function")
        def y(request):
            return request.param * 2

        @pytest.mark.parametrize("x, y", [("a", "b")], indirect=True)
        def test_simple(x):
            assert len(x) == 3

        with pytest.raises(pytest.Collector.CollectError, match="uses no fixture 'y'"):
            collect_tests(x, y, test_simple, rootpath=tmp_path)

    def test_parametrize_indirect_uses_no_fixture_error_indirect_string(
        self, tmp_path: Path
    ) -> None:
        """#714"""

        @pytest.fixture(scope="function")
        def x(request):
            return request.param * 3

        @pytest.mark.parametrize("x, y", [("a", "b")], indirect="y")
        def test_simple(x):
            assert len(x) == 3

        with pytest.raises(pytest.Collector.CollectError, match="uses no fixture 'y'"):
            collect_tests(x, test_simple, rootpath=tmp_path)

    def test_parametrize_indirect_uses_no_fixture_error_indirect_list(
        self, tmp_path: Path
    ) -> None:
        """#714"""

        @pytest.fixture(scope="function")
        def x(request):
            return request.param * 3

        @pytest.mark.parametrize("x, y", [("a", "b")], indirect=["y"])
        def test_simple(x):
            assert len(x) == 3

        with pytest.raises(pytest.Collector.CollectError, match="uses no fixture 'y'"):
            collect_tests(x, test_simple, rootpath=tmp_path)

    def test_parametrize_argument_not_in_indirect_list(self, tmp_path: Path) -> None:
        """#714"""

        @pytest.fixture(scope="function")
        def x(request):
            return request.param * 3

        @pytest.mark.parametrize("x, y", [("a", "b")], indirect=["x"])
        def test_simple(x):
            assert len(x) == 3

        with pytest.raises(pytest.Collector.CollectError, match="uses no argument 'y'"):
            collect_tests(x, test_simple, rootpath=tmp_path)

    def test_parametrize_gives_indicative_error_on_function_with_default_argument(
        self, tmp_path: Path
    ) -> None:
        @pytest.mark.parametrize("x, y", [("a", "b")])
        def test_simple(x, y=1):
            assert len(x) == 1

        with pytest.raises(
            pytest.Collector.CollectError,
            match="already takes an argument 'y' with a default value",
        ):
            collect_tests(test_simple, rootpath=tmp_path)

    def test_parametrize_functional(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            metafunc.parametrize("x", [1, 2], indirect=True)
            metafunc.parametrize("y", [2])

        @pytest.fixture
        def x(request):
            return request.param * 10

        def test_simple(x, y):
            assert x in (10, 20)
            assert y == 2

        record = run_tests(
            build_module(
                "test_parametrize_functional", pytest_generate_tests, x, test_simple
            ),
            rootpath=tmp_path,
        )
        assert list(record.by_test) == [
            "test_parametrize_functional.py::test_simple[1-2]",
            "test_parametrize_functional.py::test_simple[2-2]",
        ]
        record.assert_outcomes(passed=2)

    def test_parametrize_onearg(self) -> None:
        metafunc = self.Metafunc(lambda x: None)
        metafunc.parametrize("x", [1, 2])
        assert len(metafunc._calls) == 2
        assert metafunc._calls[0].params == dict(x=1)
        assert metafunc._calls[0].id == "1"
        assert metafunc._calls[1].params == dict(x=2)
        assert metafunc._calls[1].id == "2"

    def test_parametrize_onearg_indirect(self) -> None:
        metafunc = self.Metafunc(lambda x: None)
        metafunc.parametrize("x", [1, 2], indirect=True)
        assert metafunc._calls[0].params == dict(x=1)
        assert metafunc._calls[0].id == "1"
        assert metafunc._calls[1].params == dict(x=2)
        assert metafunc._calls[1].id == "2"

    def test_parametrize_twoargs(self) -> None:
        metafunc = self.Metafunc(lambda x, y: None)
        metafunc.parametrize(("x", "y"), [(1, 2), (3, 4)])
        assert len(metafunc._calls) == 2
        assert metafunc._calls[0].params == dict(x=1, y=2)
        assert metafunc._calls[0].id == "1-2"
        assert metafunc._calls[1].params == dict(x=3, y=4)
        assert metafunc._calls[1].id == "3-4"

    def test_high_scoped_parametrize_reordering(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize("arg2", [3, 4])
        @pytest.mark.parametrize("arg1", [0, 1, 2], scope="module")
        def test1(arg1, arg2):
            pass

        def test2():
            pass

        @pytest.mark.parametrize("arg1", [0, 1, 2], scope="module")
        def test3(arg1):
            pass

        # ensemble: collection order is the order the members are listed in,
        # so this mirrors the original file's definition order.
        items = collect_tests(
            build_module(
                "test_high_scoped_parametrize_reordering", test1, test2, test3
            ),
            rootpath=tmp_path,
        )
        # Items are grouped by the *value* of the module-scoped arg1 (#8914),
        # so arg1 is set up only once per distinct value: 0, 1, 2.
        assert [item.name for item in items] == [
            "test1[0-3]",
            "test1[0-4]",
            "test3[0]",
            "test1[1-3]",
            "test1[1-4]",
            "test3[1]",
            "test1[2-3]",
            "test1[2-4]",
            "test3[2]",
            "test2",
        ]

    def test_parametrize_multiple_times(self, tmp_path: Path) -> None:
        def test_func(x):
            assert 0, x

        class TestClass:
            pytestmark = pytest.mark.parametrize("y", [3, 4])

            def test_meth(self, x, y):
                assert 0, x

        record = run_tests(
            build_module(
                "test_parametrize_multiple_times",
                test_func,
                TestClass,
                pytestmark=pytest.mark.parametrize("x", [1, 2]),
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(failed=6)

    def test_parametrize_CSV(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize("x, y,", [(1, 2), (2, 3)])
        def test_func(x, y):
            assert x + 1 == y

        run_tests(test_func, rootpath=tmp_path).assert_outcomes(passed=2)

    def test_parametrize_class_scenarios(self, tmp_path: Path) -> None:
        # same as doc/en/example/parametrize scenario example
        def pytest_generate_tests(metafunc):
            idlist = []
            argvalues = []
            for scenario in metafunc.cls.scenarios:
                idlist.append(scenario[0])
                items = scenario[1].items()
                argnames = [x[0] for x in items]
                argvalues.append([x[1] for x in items])
            metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")

        class Test:
            scenarios = [
                ["1", {"arg": {1: 2}, "arg2": "value2"}],
                ["2", {"arg": "value2", "arg2": "value2"}],
            ]

            def test_1(self, arg, arg2):
                pass

            def test_2(self, arg2, arg):
                pass

            def test_3(self, arg, arg2):
                pass

        # ensemble: the collected order of the methods is their definition
        # order in the class body, as in the original file.
        record = run_tests(
            build_module(
                "test_parametrize_class_scenarios", pytest_generate_tests, Test
            ),
            rootpath=tmp_path,
        )
        assert [nodeid.rpartition("::")[2] for nodeid in record.by_test] == [
            "test_1[1]",
            "test_2[1]",
            "test_3[1]",
            "test_1[2]",
            "test_2[2]",
            "test_3[2]",
        ]
        record.assert_outcomes(passed=6)

    def test_parametrize_iterator_deprecation(self) -> None:
        """Test that using iterators for argvalues raises a deprecation warning."""

        def func(x: int) -> None:
            raise NotImplementedError()

        def data_generator() -> Iterator[int]:
            yield 1
            yield 2

        metafunc = self.Metafunc(func)

        with pytest.warns(
            pytest.PytestRemovedIn10Warning,
            match=r"Passing a non-Collection iterable to parametrize is deprecated",
        ):
            metafunc.parametrize("x", data_generator())


class TestMetafuncFunctional:
    def test_attributes(self, tmp_path: Path) -> None:
        # ensemble: the sources live in *this* file, so ``__name__`` inside
        # them is this module's; the collected module's synthetic name is
        # asserted against explicitly instead.
        module_name = "test_attributes"

        def pytest_generate_tests(metafunc):
            metafunc.parametrize("metafunc", [metafunc])

        @pytest.fixture
        def metafunc(request):
            return request.param

        def test_function(metafunc, pytestconfig):
            assert metafunc.config == pytestconfig
            assert metafunc.module.__name__ == module_name
            assert metafunc.function == test_function
            assert metafunc.cls is None

        class TestClass:
            def test_method(self, metafunc, pytestconfig):
                assert metafunc.config == pytestconfig
                assert metafunc.module.__name__ == module_name
                unbound = TestClass.test_method
                assert metafunc.function == unbound
                assert metafunc.cls == TestClass

        record = run_tests(
            build_module(
                module_name, pytest_generate_tests, metafunc, test_function, TestClass
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=2)

    def test_two_functions(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            metafunc.parametrize("arg1", [10, 20], ids=["0", "1"])

        def test_func1(arg1):
            assert arg1 == 10

        def test_func2(arg1):
            assert arg1 in (10, 20)

        module = build_module(
            "test_two_functions", pytest_generate_tests, test_func1, test_func2
        )
        record = run_tests(module, rootpath=tmp_path)
        record.assert_outcomes(passed=3, failed=1)
        assert record["test_two_functions.py::test_func1[0]"].passed
        assert record["test_two_functions.py::test_func1[1]"].failed

    def test_noself_in_method(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            assert "xyz" not in metafunc.fixturenames

        class TestHello:
            def test_hello(xyz):
                pass

        record = run_tests(
            build_module("test_noself_in_method", pytest_generate_tests, TestHello),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=1)

    def test_generate_tests_in_class(self, tmp_path: Path) -> None:
        class TestClass:
            def pytest_generate_tests(self, metafunc):
                metafunc.parametrize("hello", ["world"], ids=["hellow"])

            def test_myfunc(self, hello):
                assert hello == "world"

        record = run_tests(
            build_module("test_generate_tests_in_class", TestClass), rootpath=tmp_path
        )
        assert list(record.by_test) == [
            "test_generate_tests_in_class.py::TestClass::test_myfunc[hellow]"
        ]
        record.assert_outcomes(passed=1)

    def test_two_functions_not_same_instance(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            metafunc.parametrize("arg1", [10, 20], ids=["0", "1"])

        class TestClass:
            def test_func(self, arg1):
                assert not hasattr(self, "x")
                self.x = 1

        record = run_tests(
            build_module(
                "test_two_functions_not_same_instance",
                pytest_generate_tests,
                TestClass,
            ),
            rootpath=tmp_path,
        )
        assert list(record.by_test) == [
            "test_two_functions_not_same_instance.py::TestClass::test_func[0]",
            "test_two_functions_not_same_instance.py::TestClass::test_func[1]",
        ]
        record.assert_outcomes(passed=2)

    def test_issue28_setup_method_in_generate_tests(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            metafunc.parametrize("arg1", [1])

        class TestClass:
            def test_method(self, arg1):
                assert arg1 == self.val

            def setup_method(self, func):
                self.val = 1

        record = run_tests(
            build_module(
                "test_issue28_setup_method_in_generate_tests",
                pytest_generate_tests,
                TestClass,
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=1)

    def test_parametrize_functional2(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            metafunc.parametrize("arg1", [1, 2])
            metafunc.parametrize("arg2", [4, 5])

        def test_hello(arg1, arg2):
            assert 0, (arg1, arg2)

        record = run_tests(
            build_module(
                "test_parametrize_functional2", pytest_generate_tests, test_hello
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(failed=4)
        for args in [(1, 4), (1, 5), (2, 4), (2, 5)]:
            item = record[f"test_hello[{args[0]}-{args[1]}]"]
            assert item.call is not None
            assert str(args) in str(item.call.longrepr)

    def test_parametrize_single_arg_trailing_comma_functional(
        self, tmp_path: Path
    ) -> None:
        """Test that trailing comma in string argnames behaves like tuple argnames.

        Regression test for https://github.com/pytest-dev/pytest/issues/719
        """
        scenarios = [("a",), ("b",)]

        @pytest.mark.parametrize(("arg",), scenarios)
        def test_tuple_form(arg):
            # Tuple argnames: values are unpacked from tuples
            assert arg in ("a", "b")
            assert isinstance(arg, str)

        @pytest.mark.parametrize("arg,", scenarios)
        def test_string_trailing_comma(arg):
            # String with trailing comma: should behave like tuple form
            assert arg in ("a", "b")
            assert isinstance(arg, str)

        @pytest.mark.parametrize("arg", scenarios)
        def test_string_no_comma(arg):
            # String without comma: tuples are passed as-is
            assert arg in (("a",), ("b",))
            assert isinstance(arg, tuple)

        record = run_tests(
            test_tuple_form,
            test_string_trailing_comma,
            test_string_no_comma,
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=6)

    def test_parametrize_and_inner_getfixturevalue(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            metafunc.parametrize("arg1", [1], indirect=True)
            metafunc.parametrize("arg2", [10], indirect=True)

        @pytest.fixture
        def arg1(request):
            x = request.getfixturevalue("arg2")
            return x + request.param

        @pytest.fixture
        def arg2(request):
            return request.param

        def test_func1(arg1, arg2):
            assert arg1 == 11

        record = run_tests(
            build_module(
                "test_parametrize_and_inner_getfixturevalue",
                pytest_generate_tests,
                arg1,
                arg2,
                test_func1,
            ),
            rootpath=tmp_path,
        )
        assert list(record.by_test) == [
            "test_parametrize_and_inner_getfixturevalue.py::test_func1[1-10]"
        ]
        record.assert_outcomes(passed=1)

    def test_parametrize_on_setup_arg(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            assert "arg1" in metafunc.fixturenames
            metafunc.parametrize("arg1", [1], indirect=True)

        @pytest.fixture
        def arg1(request):
            return request.param

        @pytest.fixture
        def arg2(request, arg1):
            return 10 * arg1

        def test_func(arg2):
            assert arg2 == 10

        record = run_tests(
            build_module(
                "test_parametrize_on_setup_arg",
                pytest_generate_tests,
                arg1,
                arg2,
                test_func,
            ),
            rootpath=tmp_path,
        )
        assert list(record.by_test) == [
            "test_parametrize_on_setup_arg.py::test_func[1]"
        ]
        record.assert_outcomes(passed=1)

    def test_parametrize_with_ids(self, tmp_path: Path) -> None:
        # ensemble: the original set console_output_style=classic purely to
        # make the -v output greppable; nothing is read off the output now.
        def pytest_generate_tests(metafunc):
            metafunc.parametrize(
                ("a", "b"), [(1, 1), (1, 2)], ids=["basic", "advanced"]
            )

        def test_function(a, b):
            assert a == b

        record = run_tests(
            build_module(
                "test_parametrize_with_ids", pytest_generate_tests, test_function
            ),
            rootpath=tmp_path,
        )
        assert record["test_function[basic]"].passed
        assert record["test_function[advanced]"].failed
        record.assert_outcomes(passed=1, failed=1)

    def test_parametrize_without_ids(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            metafunc.parametrize(("a", "b"), [(1, object()), (1.3, object())])

        def test_function(a, b):
            assert 1

        items = collect_tests(
            build_module(
                "test_parametrize_without_ids", pytest_generate_tests, test_function
            ),
            rootpath=tmp_path,
        )
        assert [item.name for item in items] == [
            "test_function[1-b0]",
            "test_function[1.3-b1]",
        ]

    def test_parametrize_with_None_in_ids(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            metafunc.parametrize(
                ("a", "b"), [(1, 1), (1, 1), (1, 2)], ids=["basic", None, "advanced"]
            )

        def test_function(a, b):
            assert a == b

        record = run_tests(
            build_module(
                "test_parametrize_with_None_in_ids",
                pytest_generate_tests,
                test_function,
            ),
            rootpath=tmp_path,
        )
        assert record["test_function[basic]"].passed
        assert record["test_function[1-1]"].passed
        assert record["test_function[advanced]"].failed
        record.assert_outcomes(passed=2, failed=1)

    def test_fixture_parametrized_empty_ids(self, tmp_path: Path) -> None:
        """Fixtures parametrized with empty ids cause an internal error (#1849)."""

        @pytest.fixture(scope="module", ids=[], params=[])
        def temp(request):
            return request.param

        def test_temp(temp):
            pass

        record = run_tests(
            build_module("test_fixture_parametrized_empty_ids", temp, test_temp),
            rootpath=tmp_path,
        )
        record.assert_outcomes(skipped=1)

    def test_parametrized_empty_ids(self, tmp_path: Path) -> None:
        """Tests parametrized with empty ids cause an internal error (#1849)."""

        @pytest.mark.parametrize("temp", [], ids=list())
        def test_temp(temp):
            pass

        run_tests(test_temp, rootpath=tmp_path).assert_outcomes(skipped=1)

    def test_parametrized_ids_invalid_type(self, tmp_path: Path) -> None:
        """Test error with non-strings/non-ints, without generator (#1857)."""

        @pytest.mark.parametrize(
            "x, expected",
            [(1, 2), (3, 4), (5, 6)],
            ids=(None, 2, OSError()),  # type: ignore[arg-type]
        )
        def test_ids_numbers(x, expected):
            assert x * 2 == expected

        # ensemble: the module name is part of the reported nodeid.
        module = build_module("test_parametrized_ids_invalid_type", test_ids_numbers)
        record = run_tests(module, rootpath=tmp_path, capture_output=True)
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            [
                "In test_parametrized_ids_invalid_type.py::test_ids_numbers: ids contains unsupported value "
                "OSError() (type: <class 'OSError'>) at index 2. "
                "Supported types are: str, bytes, int, float, complex, bool, enum, regex or anything with a __name__."
            ]
        )

    def test_parametrize_with_identical_ids_get_unique_names(
        self, tmp_path: Path
    ) -> None:
        def pytest_generate_tests(metafunc):
            metafunc.parametrize(("a", "b"), [(1, 1), (1, 2)], ids=["a", "a"])

        def test_function(a, b):
            assert a == b

        record = run_tests(
            build_module(
                "test_parametrize_with_identical_ids_get_unique_names",
                pytest_generate_tests,
                test_function,
            ),
            rootpath=tmp_path,
        )
        assert record["test_function[a0]"].passed
        assert record["test_function[a1]"].failed
        record.assert_outcomes(passed=1, failed=1)

    @pytest.mark.parametrize(("scope", "length"), [("module", 2), ("function", 4)])
    def test_parametrize_scope_overrides(
        self, tmp_path: Path, scope: str, length: int
    ) -> None:
        values: list[object] = []

        def pytest_generate_tests(metafunc):
            if "arg" in metafunc.fixturenames:
                metafunc.parametrize("arg", [1, 2], indirect=True, scope=scope)

        @pytest.fixture
        def arg(request):
            values.append(request.param)
            return request.param

        def test_hello(arg):
            assert arg in (1, 2)

        def test_world(arg):
            assert arg in (1, 2)

        def test_checklength():
            assert len(values) == length

        record = run_tests(
            build_module(
                "test_parametrize_scope_overrides",
                pytest_generate_tests,
                arg,
                test_hello,
                test_world,
                test_checklength,
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=5)

    def test_parametrize_issue323(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="module", params=range(966))
        def foo(request):
            return request.param

        def test_it(foo):
            pass

        def test_it2(foo):
            pass

        # ensemble: collect_tests raises on a failed collection, so this can
        # no longer collect nothing and pass by accident.
        items = collect_tests(
            build_module("test_parametrize_issue323", foo, test_it, test_it2),
            rootpath=tmp_path,
        )
        assert len(items) == 2 * 966

    def test_usefixtures_seen_in_generate_tests(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            assert "abc" in metafunc.fixturenames
            metafunc.parametrize("abc", [1])

        @pytest.mark.usefixtures("abc")
        def test_function():
            pass

        record = run_tests(
            build_module(
                "test_usefixtures_seen_in_generate_tests",
                pytest_generate_tests,
                test_function,
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=1)

    # ensemble: conftest hooks become globally registered plugins, so there is
    # no way to express a hook that only applies to one directory - which is
    # exactly what this test is about.
    def test_generate_tests_only_done_in_subdir(self, pytester: Pytester) -> None:
        sub1 = pytester.mkpydir("sub1")
        sub2 = pytester.mkpydir("sub2")
        sub1.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                def pytest_generate_tests(metafunc):
                    assert metafunc.function.__name__ == "test_1"
                """
            ),
            encoding="utf-8",
        )
        sub2.joinpath("conftest.py").write_text(
            textwrap.dedent(
                """\
                def pytest_generate_tests(metafunc):
                    assert metafunc.function.__name__ == "test_2"
                """
            ),
            encoding="utf-8",
        )
        sub1.joinpath("test_in_sub1.py").write_text(
            "def test_1(): pass", encoding="utf-8"
        )
        sub2.joinpath("test_in_sub2.py").write_text(
            "def test_2(): pass", encoding="utf-8"
        )
        result = pytester.runpytest("--keep-duplicates", "-v", "-s", sub1, sub2, sub1)
        result.assert_outcomes(passed=3)

    def test_generate_same_function_names_issue403(self, tmp_path: Path) -> None:
        def make_tests():
            @pytest.mark.parametrize("x", range(2))
            def test_foo(x):
                pass

            return test_foo

        # ensemble: both functions are named ``test_foo``, so they have to be
        # placed under explicit module attribute names.
        record = run_tests(
            build_module(
                "test_generate_same_function_names_issue403",
                test_x=make_tests(),
                test_y=make_tests(),
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=4)

    # ensemble: ``@pytest.mark.parametrise`` raises Failed against the *host*
    # config while the decorator is applied, so the source cannot be built.
    def test_parametrize_misspelling(self, pytester: Pytester) -> None:
        """#463"""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.parametrise("x", range(2))
            def test_foo(x):
                pass
        """
        )
        result = pytester.runpytest("--collect-only")
        result.stdout.fnmatch_lines(
            [
                "collected 0 items / 1 error",
                "",
                "*= ERRORS =*",
                "*_ ERROR collecting test_parametrize_misspelling.py _*",
                "test_parametrize_misspelling.py:3: in <module>",
                '    @pytest.mark.parametrise("x", range(2))',
                "E   Failed: Unknown 'parametrise' mark, did you mean 'parametrize'?",
                "*! Interrupted: 1 error during collection !*",
                "*= no tests collected, 1 error in *",
            ]
        )

    @pytest.mark.parametrize("scope", ["class", "package"])
    def test_parametrize_missing_scope_doesnt_crash(
        self, tmp_path: Path, scope: str
    ) -> None:
        """Doesn't crash when parametrize(scope=<scope>) is used without a
        corresponding <scope> node."""

        @pytest.mark.parametrize("x", [0], scope=scope)  # type: ignore[arg-type]
        def test_it(x):
            pass

        run_tests(test_it, rootpath=tmp_path).assert_outcomes(passed=1)

    def test_parametrize_module_level_test_with_class_scope(
        self, tmp_path: Path
    ) -> None:
        """
        Test that a class-scoped parametrization without a corresponding `Class`
        gets module scope, i.e. we only create a single FixtureDef for it per module.
        """

        @pytest.mark.parametrize("x", [0, 1], scope="class")
        def test_1(x):
            pass

        @pytest.mark.parametrize("x", [1, 2], scope="module")
        def test_2(x):
            pass

        items = collect_tests(
            build_module(
                "test_parametrize_module_level_test_with_class_scope", test_1, test_2
            ),
            rootpath=tmp_path,
        )
        # ensemble: unlike Pytester.genitems() this goes through the full
        # collection protocol, which reorders high-scoped parametrizations, so
        # the items are looked up by name rather than by position.
        by_name = {item.name: item for item in items}
        assert sorted(by_name) == ["test_1[0]", "test_1[1]", "test_2[1]", "test_2[2]"]

        test_1_0 = by_name["test_1[0]"]
        assert isinstance(test_1_0, Function)
        test_1_fixture_x = test_1_0._fixtureinfo.name2fixturedefs["x"][-1]

        test_2_0 = by_name["test_2[1]"]
        assert isinstance(test_2_0, Function)
        test_2_fixture_x = test_2_0._fixtureinfo.name2fixturedefs["x"][-1]

        assert test_1_fixture_x is test_2_fixture_x

    # ensemble: this goes green under an ensemble, but for the wrong reason -
    # the package-scoped conftest fixture degrades to a plugin fixture with no
    # ``Package`` node, so a different reorder path is exercised.
    def test_reordering_with_scopeless_and_just_indirect_parametrization(
        self, pytester: Pytester
    ) -> None:
        pytester.makeconftest(
            """
            import pytest

            @pytest.fixture(scope="package")
            def fixture1():
                pass
            """
        )
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture(scope="module")
            def fixture0():
                pass

            @pytest.fixture(scope="module")
            def fixture1(fixture0):
                pass

            @pytest.mark.parametrize("fixture1", [0], indirect=True)
            def test_0(fixture1):
                pass

            @pytest.fixture(scope="module")
            def fixture():
                pass

            @pytest.mark.parametrize("fixture", [0], indirect=True)
            def test_1(fixture):
                pass

            def test_2():
                pass

            class Test:
                @pytest.fixture(scope="class")
                @classmethod
                def fixture(cls, fixture):
                    pass

                @pytest.mark.parametrize("fixture", [0], indirect=True)
                def test_3(self, fixture):
                    pass
            """
        )
        result = pytester.runpytest("-v")
        assert result.ret == 0
        result.stdout.fnmatch_lines(
            [
                "*test_0*",
                "*test_1*",
                "*test_2*",
                "*test_3*",
            ]
        )

    # ensemble: needs a real subprocess running pytest.main() twice.
    def test_parametrize_generator_multiple_runs(self, pytester: Pytester) -> None:
        """Test that generators in parametrize work with multiple pytest.main() (deprecated)."""
        testfile = pytester.makepyfile(
            """
            import pytest

            def data_generator():
                yield 1
                yield 2

            @pytest.mark.parametrize("bar", data_generator())
            def test_foo(bar):
                pass

            if __name__ == '__main__':
                args = ["-q", "--collect-only"]
                pytest.main(args)  # First run - should work with warning
                pytest.main(args)  # Second run - should also work with warning
            """
        )
        result = pytester.run(sys.executable, "-Wdefault", testfile)
        # Should see the deprecation warnings.
        result.stdout.fnmatch_lines(
            [
                "*PytestRemovedIn10Warning: Passing a non-Collection iterable*",
                "*PytestRemovedIn10Warning: Passing a non-Collection iterable*",
            ]
        )

    def test_parametrize_iterator_class_multiple_tests(self, tmp_path: Path) -> None:
        """Test that iterators in parametrize on a class get exhausted (deprecated)."""

        @pytest.mark.parametrize("n", iter(range(2)))
        class Test:
            def test_1(self, n):
                pass

            def test_2(self, n):
                pass

        # ensemble: the host suite's ``filterwarnings = error`` is inherited,
        # which would turn the deprecation into a collection error.
        spec = ConfigSpec(rootpath=tmp_path, inicfg={"filterwarnings": ["always"]})
        record = run_tests(
            build_module("test_parametrize_iterator_class_multiple_tests", Test),
            spec=spec,
        )
        # Iterator gets exhausted after first test, second test gets no parameters.
        # This is deprecated.
        assert list(record.by_test) == [
            "test_parametrize_iterator_class_multiple_tests.py::Test::test_1[0]",
            "test_parametrize_iterator_class_multiple_tests.py::Test::test_1[1]",
            "test_parametrize_iterator_class_multiple_tests.py::Test::test_2[NOTSET]",
        ]
        record.assert_outcomes(passed=2, skipped=1)
        assert any(
            "Passing a non-Collection iterable" in str(warning.message)
            for warning in record.warnings
        )


class TestMetafuncFunctionalAuto:
    """Tests related to automatically find out the correct scope for
    parametrized tests (#1832)."""

    def test_parametrize_auto_scope(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="session", autouse=True)
        def fixture():
            return 1

        @pytest.mark.parametrize("animal", ["dog", "cat"])
        def test_1(animal):
            assert animal in ("dog", "cat")

        @pytest.mark.parametrize("animal", ["fish"])
        def test_2(animal):
            assert animal == "fish"

        record = run_tests(
            build_module("test_parametrize_auto_scope", fixture, test_1, test_2),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=3)

    def test_parametrize_auto_scope_indirect(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="session")
        def echo(request):
            return request.param

        @pytest.mark.parametrize(
            "animal, echo", [("dog", 1), ("cat", 2)], indirect=["echo"]
        )
        def test_1(animal, echo):
            assert animal in ("dog", "cat")
            assert echo in (1, 2, 3)

        @pytest.mark.parametrize("animal, echo", [("fish", 3)], indirect=["echo"])
        def test_2(animal, echo):
            assert animal == "fish"
            assert echo in (1, 2, 3)

        record = run_tests(
            build_module("test_parametrize_auto_scope_indirect", echo, test_1, test_2),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=3)

    def test_parametrize_auto_scope_override_fixture(self, tmp_path: Path) -> None:
        @pytest.fixture(scope="session", autouse=True)
        def animal():
            return "fox"

        @pytest.mark.parametrize("animal", ["dog", "cat"])
        def test_1(animal):
            assert animal in ("dog", "cat")

        record = run_tests(
            build_module(
                "test_parametrize_auto_scope_override_fixture", animal, test_1
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=2)

    def test_parametrize_all_indirects(self, tmp_path: Path) -> None:
        @pytest.fixture
        def animal(request):
            return request.param

        @pytest.fixture(scope="session")
        def echo(request):
            return request.param

        @pytest.mark.parametrize(
            "animal, echo", [("dog", 1), ("cat", 2)], indirect=True
        )
        def test_1(animal, echo):
            assert animal in ("dog", "cat")
            assert echo in (1, 2, 3)

        @pytest.mark.parametrize("animal, echo", [("fish", 3)], indirect=True)
        def test_2(animal, echo):
            assert animal == "fish"
            assert echo in (1, 2, 3)

        record = run_tests(
            build_module(
                "test_parametrize_all_indirects", animal, echo, test_1, test_2
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=3)

    def test_parametrize_some_arguments_auto_scope(self, tmp_path: Path) -> None:
        """Integration test for (#3941)"""
        # ensemble: the sources are real objects, so the setup log is a plain
        # closed-over list instead of an attribute smuggled onto ``sys``.
        class_fix_setup: list[object] = []
        func_fix_setup: list[object] = []

        @pytest.fixture(scope="class", autouse=True)
        def class_fix(request):
            class_fix_setup.append(request.param)

        @pytest.fixture(autouse=True)
        def func_fix():
            func_fix_setup.append(True)

        @pytest.mark.parametrize("class_fix", [10, 20], indirect=True)
        class Test:
            def test_foo(self):
                pass

            def test_bar(self):
                pass

        record = run_tests(
            build_module(
                "test_parametrize_some_arguments_auto_scope",
                class_fix,
                func_fix,
                Test,
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=4)
        assert func_fix_setup == [True] * 4
        assert class_fix_setup == [10, 20]

    def test_parametrize_issue634(self, tmp_path: Path) -> None:
        # ensemble: what the original grepped out of the captured stdout is
        # recorded directly instead.
        prepared: list[int] = []

        @pytest.fixture(scope="module")
        def foo(request):
            prepared.append(request.param)
            return f"foo-{request.param}"

        def test_one(foo):
            pass

        def test_two(foo):
            pass

        test_two.test_with = (2, 3)  # type: ignore[attr-defined]

        def pytest_generate_tests(metafunc):
            params = (1, 2, 3, 4)
            if "foo" not in metafunc.fixturenames:
                return

            test_with = getattr(metafunc.function, "test_with", None)
            if test_with:
                params = test_with
            metafunc.parametrize("foo", params, indirect=True)

        record = run_tests(
            build_module(
                "test_parametrize_issue634",
                foo,
                test_one,
                test_two,
                pytest_generate_tests,
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=6)
        assert prepared.count(2) == 1
        assert prepared.count(3) == 1


class TestMarkersWithParametrization:
    """#308"""

    def test_simple_mark(self, tmp_path: Path) -> None:
        @pytest.mark.foo
        @pytest.mark.parametrize(
            ("n", "expected"),
            [
                (1, 2),
                pytest.param(1, 3, marks=pytest.mark.bar),
                (2, 3),
            ],
        )
        def test_increment(n, expected):
            assert n + 1 == expected

        items = collect_tests(test_increment, rootpath=tmp_path)
        assert len(items) == 3
        for item in items:
            assert "foo" in item.keywords
        assert "bar" not in items[0].keywords
        assert "bar" in items[1].keywords
        assert "bar" not in items[2].keywords

    def test_select_based_on_mark(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            ("n", "expected"),
            [
                (1, 2),
                pytest.param(2, 3, marks=pytest.mark.foo),
                (3, 4),
            ],
        )
        def test_increment(n, expected):
            assert n + 1 == expected

        spec = ConfigSpec(rootpath=tmp_path, args=("-m", "foo"))
        record = run_tests(test_increment, spec=spec)
        record.assert_outcomes(passed=1, deselected=2)

    def test_simple_xfail(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            ("n", "expected"),
            [
                (1, 2),
                pytest.param(1, 3, marks=pytest.mark.xfail),
                (2, 3),
            ],
        )
        def test_increment(n, expected):
            assert n + 1 == expected

        # ensemble: HookRecorder.assertoutcome() lumped xfails in with the
        # skips (hence the old "xfail is skip??"); RunRecord reports the real
        # category, so this now asserts xfailed=1.
        run_tests(test_increment, rootpath=tmp_path).assert_outcomes(
            passed=2, xfailed=1
        )

    def test_simple_xfail_single_argname(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            "n",
            [
                2,
                pytest.param(3, marks=pytest.mark.xfail),
                4,
            ],
        )
        def test_isEven(n):
            assert n % 2 == 0

        # ensemble: xfailed, not skipped - see test_simple_xfail.
        run_tests(test_isEven, rootpath=tmp_path).assert_outcomes(passed=2, xfailed=1)

    def test_xfail_with_arg(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            ("n", "expected"),
            [
                (1, 2),
                pytest.param(1, 3, marks=pytest.mark.xfail("True")),
                (2, 3),
            ],
        )
        def test_increment(n, expected):
            assert n + 1 == expected

        # ensemble: xfailed, not skipped - see test_simple_xfail.
        run_tests(test_increment, rootpath=tmp_path).assert_outcomes(
            passed=2, xfailed=1
        )

    def test_xfail_with_kwarg(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            ("n", "expected"),
            [
                (1, 2),
                pytest.param(1, 3, marks=pytest.mark.xfail(reason="some bug")),
                (2, 3),
            ],
        )
        def test_increment(n, expected):
            assert n + 1 == expected

        # ensemble: xfailed, not skipped - see test_simple_xfail.
        run_tests(test_increment, rootpath=tmp_path).assert_outcomes(
            passed=2, xfailed=1
        )

    def test_xfail_with_arg_and_kwarg(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            ("n", "expected"),
            [
                (1, 2),
                pytest.param(1, 3, marks=pytest.mark.xfail("True", reason="some bug")),
                (2, 3),
            ],
        )
        def test_increment(n, expected):
            assert n + 1 == expected

        # ensemble: xfailed, not skipped - see test_simple_xfail.
        run_tests(test_increment, rootpath=tmp_path).assert_outcomes(
            passed=2, xfailed=1
        )

    @pytest.mark.parametrize("strict", [True, False])
    def test_xfail_passing_is_xpass(self, tmp_path: Path, strict: bool) -> None:
        m = pytest.mark.xfail(
            "sys.version_info > (0, 0, 0)", reason="some bug", strict=strict
        )

        @pytest.mark.parametrize(
            ("n", "expected"),
            [
                (1, 2),
                pytest.param(2, 3, marks=m),
                (3, 4),
            ],
        )
        def test_increment(n, expected):
            assert n + 1 == expected

        record = run_tests(test_increment, rootpath=tmp_path)
        # ensemble: HookRecorder.assertoutcome() counted the non-strict xpass
        # as a plain pass; RunRecord reports it as xpassed.
        if strict:
            record.assert_outcomes(passed=2, failed=1)
        else:
            record.assert_outcomes(passed=2, xpassed=1)

    def test_parametrize_called_in_generate_tests(self, tmp_path: Path) -> None:
        def pytest_generate_tests(metafunc):
            passingTestData = [(1, 2), (2, 3)]
            failingTestData = [(1, 3), (2, 2)]

            testData = passingTestData + [
                pytest.param(*d, marks=pytest.mark.xfail) for d in failingTestData
            ]
            metafunc.parametrize(("n", "expected"), testData)

        def test_increment(n, expected):
            assert n + 1 == expected

        record = run_tests(
            build_module(
                "test_parametrize_called_in_generate_tests",
                pytest_generate_tests,
                test_increment,
            ),
            rootpath=tmp_path,
        )
        # ensemble: xfailed, not skipped - see test_simple_xfail.
        record.assert_outcomes(passed=2, xfailed=2)

    def test_parametrize_ID_generation_string_int_works(self, tmp_path: Path) -> None:
        """#290"""

        @pytest.fixture
        def myfixture():
            return "example"

        @pytest.mark.parametrize("limit", (0, "0"))
        def test_limit(limit, myfixture):
            return

        record = run_tests(
            build_module(
                "test_parametrize_ID_generation_string_int_works",
                myfixture,
                test_limit,
            ),
            rootpath=tmp_path,
        )
        record.assert_outcomes(passed=2)

    @pytest.mark.parametrize("strict", [True, False])
    def test_parametrize_marked_value(self, tmp_path: Path, strict: bool) -> None:
        @pytest.mark.parametrize(
            ("n", "expected"),
            [
                pytest.param(
                    2,
                    3,
                    marks=pytest.mark.xfail(
                        "sys.version_info > (0, 0, 0)",
                        reason="some bug",
                        strict=strict,
                    ),
                ),
                pytest.param(
                    2,
                    3,
                    marks=[
                        pytest.mark.xfail(
                            "sys.version_info > (0, 0, 0)",
                            reason="some bug",
                            strict=strict,
                        )
                    ],
                ),
            ],
        )
        def test_increment(n, expected):
            assert n + 1 == expected

        record = run_tests(test_increment, rootpath=tmp_path)
        # ensemble: HookRecorder.assertoutcome() counted the non-strict xpasses
        # as plain passes; RunRecord reports them as xpassed.
        if strict:
            record.assert_outcomes(failed=2)
        else:
            record.assert_outcomes(xpassed=2)

    def test_pytest_make_parametrize_id(self, tmp_path: Path) -> None:
        # ensemble: a conftest-level hook becomes a plugin object.
        class ConftestPlugin:
            def pytest_make_parametrize_id(self, config, val):
                return str(val * 2)

        @pytest.mark.parametrize("x", range(2))
        def test_func(x):
            pass

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(test_func, spec=spec)
        assert [nodeid.rpartition("::")[2] for nodeid in record.by_test] == [
            "test_func[0]",
            "test_func[2]",
        ]
        record.assert_outcomes(passed=2)

    def test_pytest_make_parametrize_id_with_argname(self, tmp_path: Path) -> None:
        # ensemble: a conftest-level hook becomes a plugin object.
        class ConftestPlugin:
            def pytest_make_parametrize_id(self, config, val, argname):
                return str(val * 2 if argname == "x" else val * 10)

        @pytest.mark.parametrize("x", range(2))
        def test_func_a(x):
            pass

        @pytest.mark.parametrize("y", [1])
        def test_func_b(y):
            pass

        spec = ConfigSpec(rootpath=tmp_path, extra_plugins=(ConftestPlugin(),))
        record = run_tests(test_func_a, test_func_b, spec=spec)
        assert [nodeid.rpartition("::")[2] for nodeid in record.by_test] == [
            "test_func_a[0]",
            "test_func_a[2]",
            "test_func_b[10]",
        ]
        record.assert_outcomes(passed=3)

    def test_parametrize_positional_args(self, tmp_path: Path) -> None:
        """`indirect` and later arguments are keyword-only."""

        @pytest.mark.parametrize("a", [1], False)  # type: ignore[call-arg]
        def test_foo(a):
            pass

        record = run_tests(test_foo, rootpath=tmp_path, capture_output=True)
        record.stdout.fnmatch_lines(["*TypeError*positional argument*"])
        record.assert_outcomes(errors=1)

    def test_parametrize_iterator(self, tmp_path: Path) -> None:
        id_parametrize = pytest.mark.parametrize(  # type: ignore[call-arg]
            ids=(f"param{i}" for i in itertools.count())
        )

        @id_parametrize("y", ["a", "b"])
        def test1(y):
            pass

        @id_parametrize("y", ["a", "b"])
        def test2(y):
            pass

        @pytest.mark.parametrize("a, b", [(1, 2), (3, 4)], ids=itertools.count())
        def test_converted_to_str(a, b):
            pass

        # ensemble: the collection order is the order the members are listed
        # in, which the shared ids iterator depends on.
        record = run_tests(
            build_module(
                "test_parametrize_iterator", test1, test2, test_converted_to_str
            ),
            rootpath=tmp_path,
        )
        assert list(record.by_test) == [
            "test_parametrize_iterator.py::test1[param0]",
            "test_parametrize_iterator.py::test1[param1]",
            "test_parametrize_iterator.py::test2[param0]",
            "test_parametrize_iterator.py::test2[param1]",
            "test_parametrize_iterator.py::test_converted_to_str[0]",
            "test_parametrize_iterator.py::test_converted_to_str[1]",
        ]
        record.assert_outcomes(passed=6)


class TestHiddenParam:
    """Test that pytest.HIDDEN_PARAM works"""

    def test_parametrize_ids(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            ("foo", "bar"),
            [
                ("a", "x"),
                ("b", "y"),
                ("c", "z"),
            ],
            ids=["paramset1", pytest.HIDDEN_PARAM, "paramset3"],
        )
        def test_func(foo, bar):
            pass

        items = collect_tests(test_func, rootpath=tmp_path)
        names = [item.name for item in items]
        assert names == [
            "test_func[paramset1]",
            "test_func",
            "test_func[paramset3]",
        ]

    def test_param_id(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            ("foo", "bar"),
            [
                pytest.param("a", "x", id="paramset1"),
                pytest.param("b", "y", id=pytest.HIDDEN_PARAM),
                ("c", "z"),
            ],
        )
        def test_func(foo, bar):
            pass

        items = collect_tests(test_func, rootpath=tmp_path)
        names = [item.name for item in items]
        assert names == [
            "test_func[paramset1]",
            "test_func",
            "test_func[c-z]",
        ]

    def test_multiple_hidden_param_is_forbidden(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            ("foo", "bar"),
            [
                ("a", "x"),
                ("b", "y"),
            ],
            ids=[pytest.HIDDEN_PARAM, pytest.HIDDEN_PARAM],
        )
        def test_func(foo, bar):
            pass

        # ensemble: the module name is part of the reported nodeid. An
        # ensemble never aborts the session, so the two lines the original
        # matched about that ("! Interrupted: 1 error during collection !" and
        # "no tests collected") have no equivalent; the structured error count
        # is asserted instead.
        module = build_module("test_multiple_hidden_param_is_forbidden", test_func)
        record = run_tests(module, rootpath=tmp_path, capture_output=True)
        record.assert_outcomes(errors=1)
        record.stdout.fnmatch_lines(
            [
                "collected 0 items / 1 error",
                "",
                "*= ERRORS =*",
                "*_ ERROR collecting test_multiple_hidden_param_is_forbidden.py _*",
                "E   Failed: In test_multiple_hidden_param_is_forbidden.py::test_func: multiple instances of "
                "HIDDEN_PARAM cannot be used in the same parametrize call, because the tests names need to be unique.",
            ]
        )

    def test_multiple_hidden_param_is_forbidden_idmaker(self) -> None:
        id_maker = IdMaker(
            ("foo", "bar"),
            [pytest.param("a", "x"), pytest.param("b", "y")],
            None,
            [pytest.HIDDEN_PARAM, pytest.HIDDEN_PARAM],
            None,
            "some_node_id",
        )
        expected = "In some_node_id: multiple instances of HIDDEN_PARAM"
        with pytest.raises(Failed, match=expected):
            id_maker.make_unique_parameterset_ids()

    def test_idmaker_error_without_nodeid(self) -> None:
        id_maker = IdMaker(["a"], [pytest.param("a")], None, [object()], None, None)
        with pytest.raises(Failed, match="ids contains unsupported value"):
            id_maker.make_unique_parameterset_ids()

    def test_multiple_parametrize(self, tmp_path: Path) -> None:
        @pytest.mark.parametrize(
            "bar",
            ["x", "y"],
        )
        @pytest.mark.parametrize(
            "foo",
            ["a", "b"],
            ids=["a", pytest.HIDDEN_PARAM],
        )
        def test_func(foo, bar):
            pass

        items = collect_tests(test_func, rootpath=tmp_path)
        names = [item.name for item in items]
        assert names == [
            "test_func[a-x]",
            "test_func[a-y]",
            "test_func[x]",
            "test_func[y]",
        ]

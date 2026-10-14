# mypy: allow-untyped-defs
from __future__ import annotations

import ast
import linecache
from pathlib import Path
import re
import sys
from types import FrameType
from typing import Any
from unittest import mock

from _pytest._code import Code
from _pytest._code import ExceptionInfo
from _pytest._code import Frame
from _pytest._code import Source
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ReprFuncArgs
import pytest


def test_ne() -> None:
    code1 = Code(compile('foo = "bar"', "", "exec"))
    assert code1 == code1
    code2 = Code(compile('foo = "baz"', "", "exec"))
    assert code2 != code1


def test_code_gives_back_name_for_not_existing_file() -> None:
    name = "abc-123"
    co_code = compile("pass\n", name, "exec")
    assert co_code.co_filename == name
    code = Code(co_code)
    assert str(code.path) == name
    assert code.fullsource is None


def test_code_from_function_with_class() -> None:
    class A:
        pass

    with pytest.raises(TypeError):
        Code.from_function(A)


def x() -> None:
    raise NotImplementedError()


def test_code_fullsource() -> None:
    code = Code.from_function(x)
    full = code.fullsource
    assert "test_code_fullsource()" in str(full)


def test_code_source() -> None:
    code = Code.from_function(x)
    src = code.source()
    expected = """def x() -> None:
    raise NotImplementedError()"""
    assert str(src) == expected


def test_frame_getsourcelineno_myself() -> None:
    def func() -> FrameType:
        return sys._getframe(0)

    f = Frame(func())
    source, lineno = f.code.fullsource, f.lineno
    assert source is not None
    assert source[lineno].startswith("        return sys._getframe(0)")


def test_getstatement_empty_fullsource() -> None:
    def func() -> FrameType:
        return sys._getframe(0)

    f = Frame(func())
    with mock.patch.object(f.code.__class__, "fullsource", None):
        assert f.statement == Source("")


def test_code_from_func() -> None:
    co = Code.from_function(test_frame_getsourcelineno_myself)
    assert co.firstlineno
    assert co.path


def test_unicode_handling() -> None:
    value = "ąć".encode()

    with pytest.raises(Exception) as excinfo:
        raise Exception(value)
    str(excinfo)


def test_code_getargs() -> None:
    def f1(x):
        raise NotImplementedError()

    c1 = Code.from_function(f1)
    assert c1.getargs(var=True) == ("x",)

    def f2(x, *y):
        raise NotImplementedError()

    c2 = Code.from_function(f2)
    assert c2.getargs(var=True) == ("x", "y")

    def f3(x, **z):
        raise NotImplementedError()

    c3 = Code.from_function(f3)
    assert c3.getargs(var=True) == ("x", "z")

    def f4(x, *y, **z):
        raise NotImplementedError()

    c4 = Code.from_function(f4)
    assert c4.getargs(var=True) == ("x", "y", "z")

    def f5(x, *y, **z):
        a1 = a2 = a3 = a4 = a5 = a6 = 1  # noqa: F841

    c5 = Code.from_function(f5)
    f5(1, 2, 3, z=4)  # cover function body
    assert c5.getargs(var=True) == ("x", "y", "z")

    def f6(x, *y, kw=1, **z):
        a1 = a2 = a3 = a4 = a5 = a6 = 1  # noqa: F841

    c6 = Code.from_function(f6)
    f6(1, 2, kw=3, z=4)  # cover function body
    assert c6.getargs(var=True) == ("x", "kw", "y", "z")


def test_frame_getargs() -> None:
    def f1(x) -> FrameType:
        return sys._getframe(0)

    fr1 = Frame(f1("a"))
    assert fr1.getargs(var=True) == [("x", "a")]

    def f2(x, *y) -> FrameType:
        return sys._getframe(0)

    fr2 = Frame(f2("a", "b", "c"))
    assert fr2.getargs(var=True) == [("x", "a"), ("y", ("b", "c"))]

    def f3(x, **z) -> FrameType:
        return sys._getframe(0)

    fr3 = Frame(f3("a", b="c"))
    assert fr3.getargs(var=True) == [("x", "a"), ("z", {"b": "c"})]

    def f4(x, *y, **z) -> FrameType:
        return sys._getframe(0)

    fr4 = Frame(f4("a", "b", c="d"))
    assert fr4.getargs(var=True) == [("x", "a"), ("y", ("b",)), ("z", {"c": "d"})]


class TestExceptionInfo:
    def test_bad_getsource(self) -> None:
        try:
            if False:
                pass  # type: ignore[unreachable]
            else:
                assert False
        except AssertionError:
            exci = ExceptionInfo.from_current()
        assert exci.getrepr()

    def test_from_current_with_missing(self) -> None:
        with pytest.raises(AssertionError, match="no current exception"):
            ExceptionInfo.from_current()


class TestTracebackEntry:
    def test_getsource(self) -> None:
        try:
            if False:
                pass  # type: ignore[unreachable]
            else:
                assert False
        except AssertionError:
            exci = ExceptionInfo.from_current()
        entry = exci.traceback[0]
        source = entry.getsource()
        assert source is not None
        assert len(source) == 6
        assert "assert False" in source[5]

    def test_tb_entry_str(self):
        try:
            assert False
        except AssertionError:
            exci = ExceptionInfo.from_current()
        pattern = r"  File '.*test_code.py':\d+ in test_tb_entry_str\n  assert False"
        entry = str(exci.traceback[0])
        assert re.match(pattern, entry)


class TestReprFuncArgs:
    def test_not_raise_exception_with_mixed_encoding(self, tw_mock) -> None:
        args = [("unicode_string", "São Paulo"), ("utf8_string", b"S\xc3\xa3o Paulo")]

        r = ReprFuncArgs(args)
        r.toterminal(tw_mock)

        assert (
            tw_mock.lines[0]
            == r"unicode_string = São Paulo, utf8_string = b'S\xc3\xa3o Paulo'"
        )


def test_ExceptionChainRepr():
    """Test ExceptionChainRepr, especially with regard to being hashable."""
    try:
        raise ValueError()
    except ValueError:
        excinfo1 = ExceptionInfo.from_current()
        excinfo2 = ExceptionInfo.from_current()

    repr1 = excinfo1.getrepr()
    repr2 = excinfo2.getrepr()
    assert repr1 != repr2

    assert isinstance(repr1, ExceptionChainRepr)
    assert hash(repr1) != hash(repr2)
    assert repr1 is not excinfo1.getrepr()


class TestGetSourceNarrowing:
    """``TracebackEntry.getsource`` parses the enclosing block, not the file.

    Locating the failing statement means parsing source into an AST. Doing
    that for the whole file costs O(file) per rendered traceback entry, which
    is why the block is parsed instead -- with a fallback for the frames whose
    block cannot be determined or does not contain the reported line.
    """

    def test_function_frame_parses_only_the_block(self) -> None:
        astcache: dict[tuple[str | Path, int], ast.AST] = {}
        excinfo = pytest.raises(ValueError, self._boom)
        entry = excinfo.traceback[-1]
        source = entry.getsource(astcache)
        assert source is not None
        assert str(source).endswith('raise ValueError("boom")')

        # The cached tree is the method, not this whole file.
        (cached,) = astcache.values()
        assert isinstance(cached, ast.Module)
        (node,) = cached.body
        assert isinstance(node, ast.FunctionDef)
        assert node.name == "_boom"

    def _boom(self) -> None:
        raise ValueError("boom")

    def test_module_frame_parses_the_whole_file(self) -> None:
        """A module frame has no enclosing block, so it parses the whole file.

        ``getblock`` walks an indented suite only for def/class/decorated
        code; for anything else it stops at the first logical line, which is
        never the whole module.
        """
        astcache: dict[tuple[str | Path, int], ast.AST] = {}
        filename = "<pytest-narrow-module>"
        lines = ["if 1:\n", "    raise ValueError('boom')\n", "x = 2\n"]
        code = compile("".join(lines), filename, "exec")
        with mock.patch.dict(linecache.cache, {filename: (1, None, lines, filename)}):
            excinfo = pytest.raises(ValueError, exec, code, {})
            entry = excinfo.traceback[-1]
            assert entry.frame.code.raw.co_name == "<module>"
            source = entry.getsource(astcache)
        assert source is not None
        assert str(source) == "if 1:\n    raise ValueError('boom')"

        # The whole file, so the statement after the block is in the tree.
        (cached,) = astcache.values()
        assert isinstance(cached, ast.Module)
        assert len(cached.body) == 2

    def test_line_outside_the_block_falls_back(self) -> None:
        """The block can fall short of the frame -- exec'd or generated code,
        a decorator returning a differently shaped callable."""
        filename = "<pytest-narrow-short>"
        # What linecache reports disagrees with what was compiled: the block
        # starting at line 1 is a single line, but the frame reports line 3.
        shown = ["def g(): pass\n", "x = 1\n", "y = 2\n"]
        code = compile(
            "def g():\n    x = 1\n    raise ValueError('boom')\n", filename, "exec"
        )
        ns: dict[str, Any] = {}
        exec(code, ns)
        with mock.patch.dict(linecache.cache, {filename: (1, None, shown, filename)}):
            excinfo = pytest.raises(ValueError, ns["g"])
            source = excinfo.traceback[-1].getsource()
        assert source is not None
        # Narrowing would have cut the file off after line 1.
        assert "y = 2" in str(source)

    def test_unparseable_block_falls_back(self) -> None:
        """``inspect.getblock`` tokenizes, and tokenizing can fail."""
        filename = "<pytest-narrow-broken>"
        shown = ["def g():\n", '    """\n']
        code = compile("def g():\n    raise ValueError('boom')\n", filename, "exec")
        ns: dict[str, Any] = {}
        exec(code, ns)
        with mock.patch.dict(linecache.cache, {filename: (1, None, shown, filename)}):
            excinfo = pytest.raises(ValueError, ns["g"])
            source = excinfo.traceback[-1].getsource()
        assert source is not None

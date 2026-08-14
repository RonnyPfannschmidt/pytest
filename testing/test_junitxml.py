from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from datetime import timezone
import os
from pathlib import Path
import platform
from typing import Any
from typing import cast
from typing import TYPE_CHECKING
from xml.dom import minidom

import xmlschema

from _pytest.config import Config
from _pytest.config import UsageError
from _pytest.ensemble import build_module
from _pytest.ensemble import ConfigSpec
from _pytest.ensemble import DEFAULT_MODULE_NAME
from _pytest.ensemble import module_from_path
from _pytest.ensemble import run_tests
from _pytest.ensemble import RunRecord
from _pytest.ensemble import Source
from _pytest.junitxml import _JunitDurationReport
from _pytest.junitxml import _JunitFamily
from _pytest.junitxml import _JunitLogging
from _pytest.junitxml import bin_xml_escape
from _pytest.junitxml import LogXML
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester
from _pytest.pytester import RunResult
from _pytest.reports import BaseReport
from _pytest.reports import TestReport
from _pytest.stash import Stash
import _pytest.timing
import pytest


#: What ``record_property``/``record_xml_attribute``/``record_testsuite_property``
#: hand to a test.
RecordFunc = Callable[[str, object], None]


@pytest.fixture(scope="session")
def schema() -> xmlschema.XMLSchema:
    """Return an xmlschema.XMLSchema object for the junit-10.xsd file."""
    fn = Path(__file__).parent / "example_scripts/junit-10.xsd"
    with fn.open(encoding="utf-8") as f:
        return xmlschema.XMLSchema(f)


class RunAndParse:
    """Run in-memory sources under ``junitxml`` and parse the XML it wrote.

    The junit report is written from ``pytest_sessionfinish``, which for an
    ensemble is the exit of :func:`run_tests` - so the file only exists once
    that has returned.

    ``name`` is the name of the synthesized module holding loose sources,
    and the XML quotes it as the ``classname`` of every testcase, exactly
    where the pytester version quoted the name of the generated file. It is
    therefore part of what a test asserts, not an implementation detail.
    """

    def __init__(self, tmp_path: Path, schema: xmlschema.XMLSchema) -> None:
        self.tmp_path = tmp_path
        self.schema = schema
        self.xml_path = tmp_path.joinpath("junit.xml")

    def __call__(
        self,
        *sources: Source,
        name: str = DEFAULT_MODULE_NAME,
        args: Sequence[str] = (),
        inicfg: Mapping[str, object] | None = None,
        family: _JunitFamily | None = "xunit1",
        suite_name: str = "pytest",
    ) -> tuple[RunRecord, DomDocument]:
        argv = tuple(args)
        if family:
            argv = ("-o", "junit_family=" + family, *argv)
        spec = ConfigSpec(
            rootpath=self.tmp_path,
            args=(f"--junitxml={self.xml_path}", *argv),
            inicfg=inicfg if inicfg is not None else {},
        ).with_plugins("junitxml")
        record = run_tests(*sources, spec=spec, name=name)
        if family == "xunit2":
            with self.xml_path.open(encoding="utf-8") as f:
                self.schema.validate(f)
        xmldoc = minidom.parse(str(self.xml_path))
        # Ensure the tests attribute of the ``<testsuite>`` element
        # always matches the number of ``<testcase>`` elements (#3580).
        doc = DomDocument(xmldoc)
        testcase_nodes = doc.find_by_tag("testcase")
        test_suite_node = doc.get_first_by_tag("testsuite")
        test_suite_node.assert_attr(name=suite_name, tests=len(testcase_nodes))
        return record, doc


@pytest.fixture
def run_and_parse(tmp_path: Path, schema: xmlschema.XMLSchema) -> RunAndParse:
    """Fixture that returns a function that runs the given in-memory sources
    with ``--junitxml`` and returns the run record plus the parsed
    ``DomNode`` of the root xml node.

    The ``family`` parameter is used to configure the ``junit_family`` of the written report.
    "xunit2" is also automatically validated against the schema.
    """
    return RunAndParse(tmp_path, schema)


class RunAndParsePytester:
    """The pytester-backed original, kept for the tests that need a real
    directory tree, a conftest, a subprocess or captured item output."""

    def __init__(self, pytester: Pytester, schema: xmlschema.XMLSchema) -> None:
        self.pytester = pytester
        self.schema = schema

    def __call__(
        self,
        *args: str | os.PathLike[str],
        family: _JunitFamily | None = "xunit1",
        suite_name: str = "pytest",
    ) -> tuple[RunResult, DomDocument]:
        if family:
            args = ("-o", "junit_family=" + family, *args)
        xml_path = self.pytester.path.joinpath("junit.xml")
        result = self.pytester.runpytest(f"--junitxml={xml_path}", *args)
        if family == "xunit2":
            with xml_path.open(encoding="utf-8") as f:
                self.schema.validate(f)
        xmldoc = minidom.parse(str(xml_path))
        # Ensure the tests attribute of the ``<testsuite>`` element
        # always matches the number of ``<testcase>`` elements (#3580).
        doc = DomDocument(xmldoc)
        testcase_nodes = doc.find_by_tag("testcase")
        test_suite_node = doc.get_first_by_tag("testsuite")
        test_suite_node.assert_attr(name=suite_name, tests=len(testcase_nodes))
        return result, doc


@pytest.fixture
def run_and_parse_pytester(
    pytester: Pytester, schema: xmlschema.XMLSchema
) -> RunAndParsePytester:
    """``run_and_parse`` for the tests that could not move to an ensemble."""
    return RunAndParsePytester(pytester, schema)


def assert_attr(node: minidom.Element, **kwargs: object) -> None:
    __tracebackhide__ = True

    def nodeval(node: minidom.Element, name: str) -> str | None:
        anode = node.getAttributeNode(name)
        return anode.value if anode is not None else None

    expected = {name: str(value) for name, value in kwargs.items()}
    on_node = {name: nodeval(node, name) for name in expected}
    assert on_node == expected


class DomDocument:
    _node: minidom.Document | minidom.Element

    def __init__(self, dom: minidom.Document) -> None:
        self._node = dom

    def find_first_by_tag(self, tag: str) -> DomNode | None:
        return self.find_nth_by_tag(tag, 0)

    def get_first_by_tag(self, tag: str) -> DomNode:
        maybe = self.find_first_by_tag(tag)
        if maybe is None:
            raise LookupError(tag)
        else:
            return maybe

    def find_nth_by_tag(self, tag: str, n: int) -> DomNode | None:
        items = self._node.getElementsByTagName(tag)
        try:
            nth = items[n]
        except IndexError:
            return None
        else:
            return DomNode(nth)

    def find_by_tag(self, tag: str) -> list[DomNode]:
        return [DomNode(x) for x in self._node.getElementsByTagName(tag)]

    @property
    def children(self) -> list[DomNode]:
        return [
            DomNode(x) for x in self._node.childNodes if isinstance(x, minidom.Element)
        ]

    @property
    def get_unique_child(self) -> DomNode:
        children = self.children
        assert len(children) == 1
        return children[0]

    def toxml(self) -> str:
        return self._node.toxml()


class DomNode(DomDocument):
    _node: minidom.Element

    def __init__(self, dom: minidom.Element) -> None:
        self._node = dom

    def __repr__(self) -> str:
        return self.toxml()

    def __getitem__(self, key: str) -> str:
        node = self._node.getAttributeNode(key)
        if node is not None:
            return node.value
        else:
            raise KeyError(key)

    def assert_attr(self, **kwargs: object) -> None:
        __tracebackhide__ = True
        return assert_attr(self._node, **kwargs)

    @property
    def text(self) -> str:
        text = self._node.childNodes[0]
        assert isinstance(text, minidom.Text)
        return text.wholeText

    @property
    def tag(self) -> str:
        return self._node.tagName


class TestJunitHelpers:
    """minimal test to increase coverage for methods that are used in debugging"""

    @pytest.fixture
    def document(self) -> DomDocument:
        doc = minidom.parseString("""
        <root>
          <item name="a"></item>
          <item name="b"></item>
        </root>
""")
        return DomDocument(doc)

    def test_uc_root(self, document: DomDocument) -> None:
        assert document.get_unique_child.tag == "root"

    def test_node_assert_attr(self, document: DomDocument) -> None:
        item = document.get_first_by_tag("item")

        item.assert_attr(name="a")

        with pytest.raises(AssertionError):
            item.assert_attr(missing="foo")

    def test_node_getitem(self, document: DomDocument) -> None:
        item = document.get_first_by_tag("item")
        assert item["name"] == "a"

        with pytest.raises(KeyError, match="missing"):
            item["missing"]

    def test_node_get_first_lookup(self, document: DomDocument) -> None:
        with pytest.raises(LookupError, match="missing"):
            document.get_first_by_tag("missing")

    def test_node_repr(self, document: DomDocument) -> None:
        item = document.get_first_by_tag("item")

        assert repr(item) == item.toxml()
        assert item.toxml() == '<item name="a"/>'


parametrize_families = pytest.mark.parametrize("xunit_family", ["xunit1", "xunit2"])


class TestPython:
    @parametrize_families
    def test_summing_simple(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        def test_pass() -> None:
            pass

        def test_fail() -> None:
            assert 0

        def test_skip() -> None:
            pytest.skip("")

        @pytest.mark.xfail
        def test_xfail() -> None:
            assert 0

        @pytest.mark.xfail
        def test_xpass() -> None:
            assert 1

        record, dom = run_and_parse(
            test_pass,
            test_fail,
            test_skip,
            test_xfail,
            test_xpass,
            family=xunit_family,
        )
        # `assert result.ret` only said "nonzero"; the record says which.
        record.assert_outcomes(passed=1, failed=1, skipped=1, xfailed=1, xpassed=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(name="pytest", errors=0, failures=1, skipped=2, tests=5)

    @parametrize_families
    def test_summing_simple_with_errors(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        @pytest.fixture
        def fixture() -> None:
            raise Exception()

        def test_pass() -> None:
            pass

        def test_fail() -> None:
            assert 0

        def test_error(fixture: None) -> None:
            pass

        @pytest.mark.xfail
        def test_xfail() -> None:
            assert False

        @pytest.mark.xfail(strict=True)
        def test_xpass() -> None:
            assert True

        record, dom = run_and_parse(
            fixture,
            test_pass,
            test_fail,
            test_error,
            test_xfail,
            test_xpass,
            family=xunit_family,
        )
        record.assert_outcomes(
            passed=1, failed=2, errors=1, xfailed=1
        )  # strict xpass is a failure
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(name="pytest", errors=1, failures=2, skipped=1, tests=5)

    @parametrize_families
    def test_hostname_in_xml(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        def test_pass() -> None:
            pass

        _record, dom = run_and_parse(test_pass, family=xunit_family)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(hostname=platform.node())

    @parametrize_families
    def test_timestamp_in_xml(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        def test_pass() -> None:
            pass

        start_time = datetime.now(timezone.utc)
        _record, dom = run_and_parse(test_pass, family=xunit_family)
        node = dom.get_first_by_tag("testsuite")
        timestamp = datetime.fromisoformat(node["timestamp"])
        assert start_time <= timestamp < datetime.now(timezone.utc)

    def test_timing_function(
        self,
        run_and_parse: RunAndParse,
        mock_timing: _pytest.timing.MockTiming,
    ) -> None:
        from _pytest import timing

        def setup_module() -> None:
            timing.sleep(1)

        def teardown_module() -> None:
            timing.sleep(2)

        def test_sleep() -> None:
            timing.sleep(4)

        _record, dom = run_and_parse(
            build_module(
                "test_timing_function", setup_module, teardown_module, test_sleep
            )
        )
        node = dom.get_first_by_tag("testsuite")
        tnode = node.get_first_by_tag("testcase")
        val = tnode["time"]
        assert val is not None
        assert float(val) == 7.0

    @pytest.mark.parametrize("duration_report", ["call", "total"])
    def test_junit_duration_report(
        self,
        monkeypatch: MonkeyPatch,
        duration_report: _JunitDurationReport,
        run_and_parse: RunAndParse,
    ) -> None:
        # mock LogXML.node_reporter so it always sets a known duration to each test report object
        original_node_reporter = LogXML.node_reporter

        def node_reporter_wrapper(s: Any, report: TestReport) -> Any:
            report.duration = 1.0
            reporter = original_node_reporter(s, report)
            return reporter

        monkeypatch.setattr(LogXML, "node_reporter", node_reporter_wrapper)

        def test_foo() -> None:
            pass

        _record, dom = run_and_parse(
            test_foo, args=("-o", f"junit_duration_report={duration_report}")
        )
        node = dom.get_first_by_tag("testsuite")
        tnode = node.get_first_by_tag("testcase")
        val = float(tnode["time"])
        if duration_report == "total":
            assert val == 3.0
        else:
            assert duration_report == "call"
            assert val == 1.0

    @parametrize_families
    def test_setup_error(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        @pytest.fixture
        def arg(request: pytest.FixtureRequest) -> None:
            raise ValueError("Error reason")

        def test_function(arg: None) -> None:
            pass

        record, dom = run_and_parse(
            arg, test_function, name="test_setup_error", family=xunit_family
        )
        # A fixture blowing up at setup is an error, not a failure.
        record.assert_outcomes(errors=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(errors=1, tests=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="test_setup_error", name="test_function")
        fnode = tnode.get_first_by_tag("error")
        fnode.assert_attr(message='failed on setup with "ValueError: Error reason"')
        assert "ValueError" in fnode.toxml()

    @parametrize_families
    def test_teardown_error(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        @pytest.fixture
        def arg() -> Generator[None]:
            yield
            raise ValueError("Error reason")

        def test_function(arg: None) -> None:
            pass

        record, dom = run_and_parse(
            arg, test_function, name="test_teardown_error", family=xunit_family
        )
        # The call passed; the teardown is what errored.
        record.assert_outcomes(passed=1, errors=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(errors=1, tests=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="test_teardown_error", name="test_function")
        fnode = tnode.get_first_by_tag("error")
        fnode.assert_attr(message='failed on teardown with "ValueError: Error reason"')
        assert "ValueError" in fnode.toxml()

    @parametrize_families
    def test_call_failure_teardown_error(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        @pytest.fixture
        def arg() -> Generator[None]:
            yield
            raise Exception("Teardown Exception")

        def test_function(arg: None) -> None:
            raise Exception("Call Exception")

        record, dom = run_and_parse(arg, test_function, family=xunit_family)
        record.assert_outcomes(failed=1, errors=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(errors=1, failures=1, tests=2)
        first, second = dom.find_by_tag("testcase")
        assert first
        assert second
        assert first != second
        fnode = first.get_first_by_tag("failure")
        fnode.assert_attr(message="Exception: Call Exception")
        snode = second.get_first_by_tag("error")
        snode.assert_attr(
            message='failed on teardown with "Exception: Teardown Exception"'
        )

    @parametrize_families
    def test_skip_contains_name_reason(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        def test_skip() -> None:
            pytest.skip("hello23")

        record, dom = run_and_parse(
            test_skip, name="test_skip_contains_name_reason", family=xunit_family
        )
        record.assert_outcomes(skipped=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(skipped=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="test_skip_contains_name_reason", name="test_skip")
        snode = tnode.get_first_by_tag("skipped")
        snode.assert_attr(type="pytest.skip", message="hello23")

    @parametrize_families
    def test_mark_skip_contains_name_reason(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        @pytest.mark.skip(reason="hello24")
        def test_skip() -> None:
            assert True

        record, dom = run_and_parse(
            test_skip, name="test_mark_skip_contains_name_reason", family=xunit_family
        )
        record.assert_outcomes(skipped=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(skipped=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(
            classname="test_mark_skip_contains_name_reason", name="test_skip"
        )
        snode = tnode.get_first_by_tag("skipped")
        snode.assert_attr(type="pytest.skip", message="hello24")

    @parametrize_families
    def test_mark_skipif_contains_name_reason(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        # The module global of the original is a closure variable here; the
        # mark stores the evaluated condition either way.
        global_condition = True

        @pytest.mark.skipif(global_condition, reason="hello25")
        def test_skip() -> None:
            assert True

        record, dom = run_and_parse(
            test_skip, name="test_mark_skipif_contains_name_reason", family=xunit_family
        )
        record.assert_outcomes(skipped=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(skipped=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(
            classname="test_mark_skipif_contains_name_reason", name="test_skip"
        )
        snode = tnode.get_first_by_tag("skipped")
        snode.assert_attr(type="pytest.skip", message="hello25")

    # ensemble: asserts that captured output is *absent*; without item-level
    # capture in an ensemble nothing is ever captured and the assertion would
    # hold for the wrong reason.
    @parametrize_families
    def test_mark_skip_doesnt_capture_output(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        xunit_family: _JunitFamily,
    ) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.mark.skip(reason="foo")
            def test_skip():
                print("bar!")
        """
        )
        result, dom = run_and_parse_pytester(family=xunit_family)
        assert result.ret == 0
        node_xml = dom.get_first_by_tag("testsuite").toxml()
        assert "bar!" not in node_xml

    @parametrize_families
    def test_classname_instance(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        class TestClass:
            def test_method(self) -> None:
                assert 0

        record, dom = run_and_parse(
            TestClass, name="test_classname_instance", family=xunit_family
        )
        record.assert_outcomes(failed=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(failures=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(
            classname="test_classname_instance.TestClass", name="test_method"
        )

    @parametrize_families
    def test_classname_nested_dir(
        self, tmp_path: Path, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        # The classname comes from the nodeid, so the module has to genuinely
        # live in a subdirectory of the rootdir: written to disk and imported
        # with module_from_path, rather than synthesized.
        sub = tmp_path.joinpath("sub")
        sub.mkdir()
        p = sub.joinpath("test_hello.py")
        p.write_text("def test_func(): 0/0", encoding="utf-8")
        record, dom = run_and_parse(module_from_path(p), family=xunit_family)
        record.assert_outcomes(failed=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(failures=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="sub.test_hello", name="test_func")

    # ensemble: an internal error is reported by wrap_session, which an
    # ensemble does not run - the exception escapes the run instead.
    @parametrize_families
    def test_internal_error(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        xunit_family: _JunitFamily,
    ) -> None:
        pytester.makeconftest("def pytest_runtest_protocol(): 0 / 0")
        pytester.makepyfile("def test_function(): pass")
        result, dom = run_and_parse_pytester(family=xunit_family)
        assert result.ret
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(errors=1, tests=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="pytest", name="internal")
        fnode = tnode.get_first_by_tag("error")
        fnode.assert_attr(message="internal error")
        assert "Division" in fnode.toxml()

    # ensemble: the system-out/system-err sections come from the item-level
    # capture an ensemble has no way to start (the CaptureManager is created
    # in pytest_load_initial_conftests, which an ensemble never runs).
    @pytest.mark.parametrize(
        "junit_logging", ["no", "log", "system-out", "system-err", "out-err", "all"]
    )
    @parametrize_families
    def test_failure_function(
        self,
        pytester: Pytester,
        junit_logging: _JunitLogging,
        run_and_parse_pytester: RunAndParsePytester,
        xunit_family: _JunitFamily,
    ) -> None:
        pytester.makepyfile(
            """
            import logging
            import sys

            def test_fail():
                print("hello-stdout")
                sys.stderr.write("hello-stderr\\n")
                logging.info('info msg')
                logging.warning('warning msg')
                raise ValueError(42)
        """
        )

        result, dom = run_and_parse_pytester(
            "-o", f"junit_logging={junit_logging}", family=xunit_family
        )
        assert result.ret, "Expected ret > 0"
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(failures=1, tests=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="test_failure_function", name="test_fail")
        fnode = tnode.get_first_by_tag("failure")
        fnode.assert_attr(message="ValueError: 42")
        assert "ValueError" in fnode.toxml(), "ValueError not included"

        if junit_logging in ["log", "all"]:
            logdata = tnode.get_first_by_tag("system-out")
            log_xml = logdata.toxml()
            assert logdata.tag == "system-out", "Expected tag: system-out"
            assert "info msg" not in log_xml, "Unexpected INFO message"
            assert "warning msg" in log_xml, "Missing WARN message"
        if junit_logging in ["system-out", "out-err", "all"]:
            systemout = tnode.get_first_by_tag("system-out")
            systemout_xml = systemout.toxml()
            assert systemout.tag == "system-out", "Expected tag: system-out"
            assert "info msg" not in systemout_xml, "INFO message found in system-out"
            assert "hello-stdout" in systemout_xml, (
                "Missing 'hello-stdout' in system-out"
            )
        if junit_logging in ["system-err", "out-err", "all"]:
            systemerr = tnode.get_first_by_tag("system-err")
            systemerr_xml = systemerr.toxml()
            assert systemerr.tag == "system-err", "Expected tag: system-err"
            assert "info msg" not in systemerr_xml, "INFO message found in system-err"
            assert "hello-stderr" in systemerr_xml, (
                "Missing 'hello-stderr' in system-err"
            )
            assert "warning msg" not in systemerr_xml, (
                "WARN message found in system-err"
            )
        if junit_logging == "no":
            assert not tnode.find_by_tag("log"), "Found unexpected content: log"
            assert not tnode.find_by_tag("system-out"), (
                "Found unexpected content: system-out"
            )
            assert not tnode.find_by_tag("system-err"), (
                "Found unexpected content: system-err"
            )

    @parametrize_families
    def test_failure_verbose_message(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        def test_fail() -> None:
            assert 0, "An error"

        record, dom = run_and_parse(test_fail, family=xunit_family)
        record.assert_outcomes(failed=1)
        node = dom.get_first_by_tag("testsuite")
        tnode = node.get_first_by_tag("testcase")
        fnode = tnode.get_first_by_tag("failure")
        fnode.assert_attr(message="AssertionError: An error\nassert 0")

    # ensemble: asserts on the captured system-out of each parametrized item.
    @parametrize_families
    def test_failure_escape(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        xunit_family: _JunitFamily,
    ) -> None:
        pytester.makepyfile(
            """
            import pytest
            @pytest.mark.parametrize('arg1', "<&'", ids="<&'")
            def test_func(arg1):
                print(arg1)
                assert 0
        """
        )
        result, dom = run_and_parse_pytester(
            "-o", "junit_logging=system-out", family=xunit_family
        )
        assert result.ret
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(failures=3, tests=3)
        tnodes = node.find_by_tag("testcase")
        for tnode, char in zip(tnodes, "<&'", strict=True):
            tnode.assert_attr(
                classname="test_failure_escape", name=f"test_func[{char}]"
            )
            sysout = tnode.get_first_by_tag("system-out")
            text = sysout.text
            assert f"{char}\n" in text

    @parametrize_families
    def test_junit_prefixing(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        def test_func() -> None:
            assert 0

        class TestHello:
            def test_hello(self) -> None:
                pass

        record, dom = run_and_parse(
            test_func,
            TestHello,
            name="test_junit_prefixing",
            args=("--junitprefix=xyz",),
            family=xunit_family,
        )
        record.assert_outcomes(passed=1, failed=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(failures=1, tests=2)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="xyz.test_junit_prefixing", name="test_func")
        tnode = node.find_by_tag("testcase")[1]
        tnode.assert_attr(
            classname="xyz.test_junit_prefixing.TestHello", name="test_hello"
        )

    @parametrize_families
    def test_xfailure_function(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        def test_xfail() -> None:
            pytest.xfail("42")

        record, dom = run_and_parse(
            test_xfail, name="test_xfailure_function", family=xunit_family
        )
        record.assert_outcomes(xfailed=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(skipped=1, tests=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="test_xfailure_function", name="test_xfail")
        fnode = tnode.get_first_by_tag("skipped")
        fnode.assert_attr(type="pytest.xfail", message="42")

    @parametrize_families
    def test_xfailure_marker(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        @pytest.mark.xfail(reason="42")
        def test_xfail() -> None:
            assert False

        record, dom = run_and_parse(
            test_xfail, name="test_xfailure_marker", family=xunit_family
        )
        record.assert_outcomes(xfailed=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(skipped=1, tests=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="test_xfailure_marker", name="test_xfail")
        fnode = tnode.get_first_by_tag("skipped")
        fnode.assert_attr(type="pytest.xfail", message="42")

    # ensemble: counts the captured output sections of an item.
    @pytest.mark.parametrize(
        "junit_logging", ["no", "log", "system-out", "system-err", "out-err", "all"]
    )
    def test_xfail_captures_output_once(
        self,
        pytester: Pytester,
        junit_logging: _JunitLogging,
        run_and_parse_pytester: RunAndParsePytester,
    ) -> None:
        pytester.makepyfile(
            """
            import sys
            import pytest

            @pytest.mark.xfail()
            def test_fail():
                sys.stdout.write('XFAIL This is stdout')
                sys.stderr.write('XFAIL This is stderr')
                assert 0
        """
        )
        _result, dom = run_and_parse_pytester("-o", f"junit_logging={junit_logging}")
        node = dom.get_first_by_tag("testsuite")
        tnode = node.get_first_by_tag("testcase")

        has_err_logging = junit_logging in ["system-err", "out-err", "all"]
        expected_err_output_len = 1 if has_err_logging else 0
        assert len(tnode.find_by_tag("system-err")) == expected_err_output_len

        has_out_logigng = junit_logging in ("log", "system-out", "out-err", "all")
        expected_out_output_len = 1 if has_out_logigng else 0

        assert len(tnode.find_by_tag("system-out")) == expected_out_output_len

    @parametrize_families
    def test_xfailure_xpass(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        @pytest.mark.xfail
        def test_xpass() -> None:
            pass

        record, dom = run_and_parse(
            test_xpass, name="test_xfailure_xpass", family=xunit_family
        )
        record.assert_outcomes(xpassed=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(skipped=0, tests=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="test_xfailure_xpass", name="test_xpass")

    @parametrize_families
    def test_xfailure_xpass_strict(
        self, run_and_parse: RunAndParse, xunit_family: _JunitFamily
    ) -> None:
        @pytest.mark.xfail(strict=True, reason="This needs to fail!")
        def test_xpass() -> None:
            pass

        record, dom = run_and_parse(
            test_xpass, name="test_xfailure_xpass_strict", family=xunit_family
        )
        # A strict xpass is a plain failure.
        record.assert_outcomes(failed=1)
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(skipped=0, tests=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(classname="test_xfailure_xpass_strict", name="test_xpass")
        fnode = tnode.get_first_by_tag("failure")
        fnode.assert_attr(message="[XPASS(strict)] This needs to fail!")

    # ensemble: needs a module that fails at import time; ensemble sources are
    # already-imported objects.
    @parametrize_families
    def test_collect_error(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        xunit_family: _JunitFamily,
    ) -> None:
        pytester.makepyfile("syntax error")
        result, dom = run_and_parse_pytester(family=xunit_family)
        assert result.ret
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(errors=1, tests=1)
        tnode = node.get_first_by_tag("testcase")
        fnode = tnode.get_first_by_tag("error")
        fnode.assert_attr(message="collection failure")
        assert "SyntaxError" in fnode.toxml()

    def test_unicode(self, tmp_path: Path, run_and_parse: RunAndParse) -> None:
        value = "hx\xc4\x85\xc4\x87\n"
        # The latin1 coding cookie is the point of the test, so the module has
        # to be a real file decoded by the import machinery, not a synthesized
        # one whose source lines would come from this file.
        p = tmp_path.joinpath("test_unicode.py")
        p.write_text(
            f"# coding: latin1\ndef test_hello():\n    print({value!r})\n    assert 0\n",
            encoding="latin1",
        )
        record, dom = run_and_parse(module_from_path(p))
        record.assert_outcomes(failed=1)
        tnode = dom.get_first_by_tag("testcase")
        fnode = tnode.get_first_by_tag("failure")
        assert "hx" in fnode.toxml()

    def test_assertion_binchars(self, run_and_parse: RunAndParse) -> None:
        """This test did fail when the escaping wasn't strict."""
        # Module globals in the original; closure variables here.
        m1 = "\x01\x02\x03\x04"
        m2 = "\x01\x02\x03\x05"

        def test_str_compare() -> None:
            assert m1 == m2

        record, dom = run_and_parse(test_str_compare)
        record.assert_outcomes(failed=1)
        print(dom.toxml())

    # ensemble: needs item-level capture.
    @pytest.mark.parametrize("junit_logging", ["no", "system-out"])
    def test_pass_captures_stdout(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        junit_logging: _JunitLogging,
    ) -> None:
        pytester.makepyfile(
            """
            def test_pass():
                print('hello-stdout')
        """
        )
        _result, dom = run_and_parse_pytester("-o", f"junit_logging={junit_logging}")
        node = dom.get_first_by_tag("testsuite")
        pnode = node.get_first_by_tag("testcase")
        if junit_logging == "no":
            assert not node.find_by_tag("system-out"), (
                "system-out should not be generated"
            )
        if junit_logging == "system-out":
            systemout = pnode.get_first_by_tag("system-out")
            assert "hello-stdout" in systemout.toxml(), (
                "'hello-stdout' should be in system-out"
            )

    # ensemble: needs item-level capture.
    @pytest.mark.parametrize("junit_logging", ["no", "system-err"])
    def test_pass_captures_stderr(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        junit_logging: _JunitLogging,
    ) -> None:
        pytester.makepyfile(
            """
            import sys
            def test_pass():
                sys.stderr.write('hello-stderr')
        """
        )
        _result, dom = run_and_parse_pytester("-o", f"junit_logging={junit_logging}")
        node = dom.get_first_by_tag("testsuite")
        pnode = node.get_first_by_tag("testcase")
        if junit_logging == "no":
            assert not node.find_by_tag("system-err"), (
                "system-err should not be generated"
            )
        if junit_logging == "system-err":
            systemerr = pnode.get_first_by_tag("system-err")
            assert "hello-stderr" in systemerr.toxml(), (
                "'hello-stderr' should be in system-err"
            )

    # ensemble: needs item-level capture.
    @pytest.mark.parametrize("junit_logging", ["no", "system-out"])
    def test_setup_error_captures_stdout(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        junit_logging: _JunitLogging,
    ) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture
            def arg(request):
                print('hello-stdout')
                raise ValueError()
            def test_function(arg):
                pass
        """
        )
        _result, dom = run_and_parse_pytester("-o", f"junit_logging={junit_logging}")
        node = dom.get_first_by_tag("testsuite")
        pnode = node.get_first_by_tag("testcase")
        if junit_logging == "no":
            assert not node.find_by_tag("system-out"), (
                "system-out should not be generated"
            )
        if junit_logging == "system-out":
            systemout = pnode.get_first_by_tag("system-out")
            assert "hello-stdout" in systemout.toxml(), (
                "'hello-stdout' should be in system-out"
            )

    # ensemble: needs item-level capture.
    @pytest.mark.parametrize("junit_logging", ["no", "system-err"])
    def test_setup_error_captures_stderr(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        junit_logging: _JunitLogging,
    ) -> None:
        pytester.makepyfile(
            """
            import sys
            import pytest

            @pytest.fixture
            def arg(request):
                sys.stderr.write('hello-stderr')
                raise ValueError()
            def test_function(arg):
                pass
        """
        )
        _result, dom = run_and_parse_pytester("-o", f"junit_logging={junit_logging}")
        node = dom.get_first_by_tag("testsuite")
        pnode = node.get_first_by_tag("testcase")
        if junit_logging == "no":
            assert not node.find_by_tag("system-err"), (
                "system-err should not be generated"
            )
        if junit_logging == "system-err":
            systemerr = pnode.get_first_by_tag("system-err")
            assert "hello-stderr" in systemerr.toxml(), (
                "'hello-stderr' should be in system-err"
            )

    # ensemble: needs item-level capture.
    @pytest.mark.parametrize("junit_logging", ["no", "system-out"])
    def test_avoid_double_stdout(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        junit_logging: _JunitLogging,
    ) -> None:
        pytester.makepyfile(
            """
            import sys
            import pytest

            @pytest.fixture
            def arg(request):
                yield
                sys.stdout.write('hello-stdout teardown')
                raise ValueError()
            def test_function(arg):
                sys.stdout.write('hello-stdout call')
        """
        )
        _result, dom = run_and_parse_pytester("-o", f"junit_logging={junit_logging}")
        node = dom.get_first_by_tag("testsuite")
        pnode = node.get_first_by_tag("testcase")
        if junit_logging == "no":
            assert not node.find_by_tag("system-out"), (
                "system-out should not be generated"
            )
        if junit_logging == "system-out":
            systemout = pnode.get_first_by_tag("system-out")
            assert "hello-stdout call" in systemout.toxml()
            assert "hello-stdout teardown" in systemout.toxml()


def test_mangle_test_address() -> None:
    from _pytest.junitxml import mangle_test_address

    address = "::".join(["a/my.py.thing.py", "Class", "method", "[a-1-::]"])
    newnames = mangle_test_address(address)
    assert newnames == ["a.my.py.thing", "Class", "method", "[a-1-::]"]


def test_dont_configure_on_workers(tmp_path: Path) -> None:
    gotten: list[object] = []

    class FakeConfig:
        if TYPE_CHECKING:
            workerinput = None

        def __init__(self) -> None:
            self.pluginmanager = self
            self.option = self
            self.stash = Stash()

        def getini(self, name: str) -> str:
            return "pytest"

        junitprefix = None
        # XXX: shouldn't need tmp_path ?
        xmlpath = str(tmp_path.joinpath("junix.xml"))
        register = gotten.append

    fake_config = cast(Config, FakeConfig())
    from _pytest import junitxml

    junitxml.pytest_configure(fake_config)
    assert len(gotten) == 1
    FakeConfig.workerinput = None
    junitxml.pytest_configure(fake_config)
    assert len(gotten) == 1


class TestNonPython:
    # ensemble: collects a non-python file through pytest_collect_file; an
    # ensemble serves preset collectors and never walks the filesystem.
    @parametrize_families
    def test_summing_simple(
        self,
        pytester: Pytester,
        run_and_parse_pytester: RunAndParsePytester,
        xunit_family: _JunitFamily,
    ) -> None:
        pytester.makeconftest(
            """
            import pytest
            def pytest_collect_file(file_path, parent):
                if file_path.suffix == ".xyz":
                    return MyItem.from_parent(name=file_path.name, parent=parent)
            class MyItem(pytest.Item):
                def runtest(self):
                    raise ValueError(42)
                def repr_failure(self, excinfo):
                    return "custom item runtest failed"
        """
        )
        pytester.path.joinpath("myfile.xyz").write_text("hello", encoding="utf-8")
        result, dom = run_and_parse_pytester(family=xunit_family)
        assert result.ret
        node = dom.get_first_by_tag("testsuite")
        node.assert_attr(errors=0, failures=1, skipped=0, tests=1)
        tnode = node.get_first_by_tag("testcase")
        tnode.assert_attr(name="myfile.xyz")
        fnode = tnode.get_first_by_tag("failure")
        fnode.assert_attr(message="custom item runtest failed")
        assert "custom item runtest failed" in fnode.toxml()


# ensemble: needs item-level capture (the null byte reaches the xml through
# the captured stdout section).
@pytest.mark.parametrize("junit_logging", ["no", "system-out"])
def test_nullbyte(pytester: Pytester, junit_logging: _JunitLogging) -> None:
    # A null byte cannot occur in XML (see section 2.2 of the spec)
    pytester.makepyfile(
        """
        import sys
        def test_print_nullbyte():
            sys.stdout.write('Here the null -->' + chr(0) + '<--')
            sys.stdout.write('In repr form -->' + repr(chr(0)) + '<--')
            assert False
    """
    )
    xmlf = pytester.path.joinpath("junit.xml")
    pytester.runpytest(f"--junitxml={xmlf}", "-o", f"junit_logging={junit_logging}")
    text = xmlf.read_text(encoding="utf-8")
    assert "\x00" not in text
    if junit_logging == "system-out":
        assert "#x00" in text
    if junit_logging == "no":
        assert "#x00" not in text


# ensemble: needs item-level capture.
@pytest.mark.parametrize("junit_logging", ["no", "system-out"])
def test_nullbyte_replace(pytester: Pytester, junit_logging: _JunitLogging) -> None:
    # Check if the null byte gets replaced
    pytester.makepyfile(
        """
        import sys
        def test_print_nullbyte():
            sys.stdout.write('Here the null -->' + chr(0) + '<--')
            sys.stdout.write('In repr form -->' + repr(chr(0)) + '<--')
            assert False
    """
    )
    xmlf = pytester.path.joinpath("junit.xml")
    pytester.runpytest(f"--junitxml={xmlf}", "-o", f"junit_logging={junit_logging}")
    text = xmlf.read_text(encoding="utf-8")
    if junit_logging == "system-out":
        assert "#x0" in text
    if junit_logging == "no":
        assert "#x0" not in text


def test_invalid_xml_escape() -> None:
    # Test some more invalid xml chars, the full range should be
    # tested really but let's just test the edges of the ranges
    # instead.
    # XXX Testing 0xD (\r) is tricky as it overwrites the just written
    #     line in the output, so we skip it too.
    invalid = (
        0x00,
        0x1,
        0xB,
        0xC,
        0xE,
        0x19,
        27,  # issue #126
        0xD800,
        0xDFFF,
        0xFFFE,
        0x0FFFF,
    )
    valid = (0x9, 0xA, 0x20, 0xD, 0xD7FF, 0xE000, 0xFFFD, 0x10000, 0x10FFFF)

    for i in invalid:
        got = bin_xml_escape(chr(i))
        if i <= 0xFF:
            expected = f"#x{i:02X}"
        else:
            expected = f"#x{i:04X}"
        assert got == expected
    for i in valid:
        assert chr(i) == bin_xml_escape(chr(i))


def test_logxml_path_expansion(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    home_tilde = Path(os.path.expanduser("~")).joinpath("test.xml")
    xml_tilde = LogXML(Path("~", "test.xml"), None)
    assert xml_tilde.logfile == str(home_tilde)

    monkeypatch.setenv("HOME", str(tmp_path))
    home_var = os.path.normpath(os.path.expandvars("$HOME/test.xml"))
    xml_var = LogXML(Path("$HOME", "test.xml"), None)
    assert xml_var.logfile == str(home_var)


def test_logxml_changingdir(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    def test_func() -> None:
        import os

        os.chdir("a")

    # The relative --junitxml path is resolved against the invocation cwd, so
    # the ensemble has to be run from the rootdir like pytester would.
    tmp_path.joinpath("a").mkdir()
    monkeypatch.chdir(tmp_path)
    spec = ConfigSpec(rootpath=tmp_path, args=("--junitxml=a/x.xml",)).with_plugins(
        "junitxml"
    )
    record = run_tests(test_func, spec=spec)
    record.assert_outcomes(passed=1)
    assert tmp_path.joinpath("a/x.xml").exists()


def test_logxml_makedir(tmp_path: Path) -> None:
    """--junitxml should automatically create directories for the xml file"""

    def test_pass() -> None:
        pass

    spec = ConfigSpec(
        rootpath=tmp_path,
        args=(f"--junitxml={tmp_path.joinpath('path/to/results.xml')}",),
    ).with_plugins("junitxml")
    record = run_tests(test_pass, spec=spec)
    record.assert_outcomes(passed=1)
    assert tmp_path.joinpath("path/to/results.xml").exists()


# ensemble: the UsageError is raised while the args are parsed, and
# `configured()` then masks it with a KeyError - its finally clause reads
# `config.stash[config_warnings_key]`, which is only set further down.
def test_logxml_check_isdir(pytester: Pytester) -> None:
    """Give an error if --junit-xml is a directory (#2089)"""
    result = pytester.runpytest("--junit-xml=.")
    result.stderr.fnmatch_lines(["*--junitxml must be a filename*"])


def test_escaped_parametrized_names_xml(run_and_parse: RunAndParse) -> None:
    @pytest.mark.parametrize("char", ["\x00"])
    def test_func(char: str) -> None:
        assert char

    record, dom = run_and_parse(test_func)
    record.assert_outcomes(passed=1)
    node = dom.get_first_by_tag("testcase")
    node.assert_attr(name="test_func[\\x00]")


def test_double_colon_split_function_issue469(run_and_parse: RunAndParse) -> None:
    @pytest.mark.parametrize("param", ["double::colon"])
    def test_func(param: str) -> None:
        pass

    record, dom = run_and_parse(
        test_func, name="test_double_colon_split_function_issue469"
    )
    record.assert_outcomes(passed=1)
    node = dom.get_first_by_tag("testcase")
    node.assert_attr(classname="test_double_colon_split_function_issue469")
    node.assert_attr(name="test_func[double::colon]")


def test_double_colon_split_method_issue469(run_and_parse: RunAndParse) -> None:
    class TestClass:
        @pytest.mark.parametrize("param", ["double::colon"])
        def test_func(self, param: str) -> None:
            pass

    record, dom = run_and_parse(
        TestClass, name="test_double_colon_split_method_issue469"
    )
    record.assert_outcomes(passed=1)
    node = dom.get_first_by_tag("testcase")
    node.assert_attr(classname="test_double_colon_split_method_issue469.TestClass")
    node.assert_attr(name="test_func[double::colon]")


def test_unicode_issue368(tmp_path: Path) -> None:
    path = tmp_path.joinpath("test.xml")
    log = LogXML(str(path), None)
    ustr = "ВНИ!"

    class Report(BaseReport):
        longrepr = ustr
        sections: list[tuple[str, str]] = []
        nodeid = "something"
        location = "tests/filename.py", 42, "TestClass.method"
        when = "teardown"

    test_report = cast(TestReport, Report())

    # hopefully this is not too brittle ...
    log.pytest_sessionstart()
    node_reporter = log._opentestcase(test_report)
    node_reporter.append_failure(test_report)
    node_reporter.append_collect_error(test_report)
    node_reporter.append_collect_skipped(test_report)
    node_reporter.append_error(test_report)
    test_report.longrepr = "filename", 1, ustr
    node_reporter.append_skipped(test_report)
    test_report.longrepr = "filename", 1, "Skipped: 卡嘣嘣"
    node_reporter.append_skipped(test_report)
    test_report.wasxfail = ustr
    node_reporter.append_skipped(test_report)
    log.pytest_sessionfinish()


def test_record_property(run_and_parse: RunAndParse) -> None:
    @pytest.fixture
    def other(record_property: RecordFunc) -> None:
        record_property("bar", 1)

    def test_record(record_property: RecordFunc, other: None) -> None:
        record_property("foo", "<1")

    record, dom = run_and_parse(other, test_record)
    node = dom.get_first_by_tag("testsuite")
    tnode = node.get_first_by_tag("testcase")
    psnode = tnode.get_first_by_tag("properties")
    pnodes = psnode.find_by_tag("property")
    pnodes[0].assert_attr(name="bar", value="1")
    pnodes[1].assert_attr(name="foo", value="<1")
    # was: result.stdout.fnmatch_lines(["*= 1 passed in *"])
    record.assert_outcomes(passed=1)


def test_record_property_on_test_and_teardown_failure(
    run_and_parse: RunAndParse,
) -> None:
    @pytest.fixture
    def other(record_property: RecordFunc) -> Generator[None]:
        record_property("bar", 1)
        yield
        assert 0

    def test_record(record_property: RecordFunc, other: None) -> None:
        record_property("foo", "<1")
        assert 0

    record, dom = run_and_parse(other, test_record)
    node = dom.get_first_by_tag("testsuite")
    tnodes = node.find_by_tag("testcase")
    for tnode in tnodes:
        psnode = tnode.get_first_by_tag("properties")
        assert psnode, f"testcase didn't had expected properties:\n{tnode}"
        pnodes = psnode.find_by_tag("property")
        pnodes[0].assert_attr(name="bar", value="1")
        pnodes[1].assert_attr(name="foo", value="<1")
    # was: result.stdout.fnmatch_lines(["*= 1 failed, 1 error *"])
    record.assert_outcomes(failed=1, errors=1)


def test_record_property_same_name(run_and_parse: RunAndParse) -> None:
    def test_record_with_same_name(
        record_property: RecordFunc,
    ) -> None:
        record_property("foo", "bar")
        record_property("foo", "baz")

    _record, dom = run_and_parse(test_record_with_same_name)
    node = dom.get_first_by_tag("testsuite")
    tnode = node.get_first_by_tag("testcase")
    psnode = tnode.get_first_by_tag("properties")
    pnodes = psnode.find_by_tag("property")
    pnodes[0].assert_attr(name="foo", value="bar")
    pnodes[1].assert_attr(name="foo", value="baz")


def _record_property_test() -> Callable[..., None]:
    def test_record(record_property: RecordFunc) -> None:
        record_property("foo", "bar")

    return test_record


def _record_xml_attribute_test() -> Callable[..., None]:
    def test_record(record_xml_attribute: RecordFunc) -> None:
        record_xml_attribute("foo", "bar")

    return test_record


#: The two record fixtures, as sources requesting them by real parameter name.
#: A source requests a fixture through its own signature, so the two variants
#: the original built by string formatting are two real functions here.
RECORD_FIXTURE_TESTS = {
    "record_property": _record_property_test,
    "record_xml_attribute": _record_xml_attribute_test,
}


def _record_property_pair() -> tuple[Callable[..., None], Callable[..., None]]:
    @pytest.fixture
    def other(record_property: RecordFunc) -> None:
        record_property("bar", 1)

    def test_record(record_property: RecordFunc, other: None) -> None:
        record_property("foo", "<1")

    return other, test_record


def _record_xml_attribute_pair() -> tuple[Callable[..., None], Callable[..., None]]:
    @pytest.fixture
    def other(record_xml_attribute: RecordFunc) -> None:
        record_xml_attribute("bar", 1)

    def test_record(record_xml_attribute: RecordFunc, other: None) -> None:
        record_xml_attribute("foo", "<1")

    return other, test_record


#: The same two, as the fixture/test pair of ``test_record_fixtures_xunit2``.
RECORD_FIXTURE_PAIRS = {
    "record_property": _record_property_pair,
    "record_xml_attribute": _record_xml_attribute_pair,
}


@pytest.mark.parametrize("fixture_name", ["record_property", "record_xml_attribute"])
def test_record_fixtures_without_junitxml(tmp_path: Path, fixture_name: str) -> None:
    test_record = RECORD_FIXTURE_TESTS[fixture_name]()

    spec = ConfigSpec(rootpath=tmp_path).with_plugins("junitxml")
    record = run_tests(test_record, spec=spec)
    record.assert_outcomes(passed=1)


def test_record_attribute(run_and_parse: RunAndParse) -> None:
    @pytest.fixture
    def other(record_xml_attribute: RecordFunc) -> None:
        record_xml_attribute("bar", 1)

    def test_record(record_xml_attribute: RecordFunc, other: None) -> None:
        record_xml_attribute("foo", "<1")

    record, dom = run_and_parse(
        other,
        test_record,
        family=None,
        # "always" is the ensemble's stand-in for the host-level
        # `@pytest.mark.filterwarnings("default")` of the original: process
        # global filters are inherited, and this suite both errors on
        # warnings and ignores PytestExperimentalApiWarning.
        inicfg={"junit_family": "xunit1", "filterwarnings": ["always"]},
    )
    node = dom.get_first_by_tag("testsuite")
    tnode = node.get_first_by_tag("testcase")
    tnode.assert_attr(bar="1")
    tnode.assert_attr(foo="<1")
    # The rendered warning quoted the generated file and line, which for an
    # ensemble source would be this file; the recorded warning is the same
    # warning, asserted as an object rather than as a line of output.
    assert [str(w.message) for w in record.warnings] == [
        "record_xml_attribute is an experimental feature"
    ]


@pytest.mark.parametrize("fixture_name", ["record_xml_attribute", "record_property"])
def test_record_fixtures_xunit2(fixture_name: str, run_and_parse: RunAndParse) -> None:
    """Ensure record_xml_attribute and record_property drop values when outside of legacy family."""
    other, test_record = RECORD_FIXTURE_PAIRS[fixture_name]()

    record, _dom = run_and_parse(
        other,
        test_record,
        family=None,
        inicfg={"junit_family": "xunit2", "filterwarnings": ["always"]},
    )
    expected = [
        f"{fixture_name} is incompatible "
        "with junit_family 'xunit2' (use 'legacy' or 'xunit1')"
    ]
    if fixture_name == "record_xml_attribute":
        expected.insert(0, "record_xml_attribute is an experimental feature")
    # The original only ever asserted the last line it built; both warnings
    # are checked here.
    assert [str(w.message) for w in record.warnings] == expected


# ensemble: xdist.
def test_random_report_log_xdist(
    pytester: Pytester,
    monkeypatch: MonkeyPatch,
    run_and_parse_pytester: RunAndParsePytester,
) -> None:
    """`xdist` calls pytest_runtest_logreport as they are executed by the workers,
    with nodes from several nodes overlapping, so junitxml must cope with that
    to produce correct reports (#1064)."""
    pytest.importorskip("xdist")
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
    pytester.makepyfile(
        """
        import pytest, time
        @pytest.mark.parametrize('i', list(range(30)))
        def test_x(i):
            assert i != 22
    """
    )
    _, dom = run_and_parse_pytester("-n2")
    suite_node = dom.get_first_by_tag("testsuite")
    failed = []
    for case_node in suite_node.find_by_tag("testcase"):
        if case_node.find_first_by_tag("failure"):
            failed.append(case_node["name"])

    assert failed == ["test_x[22]"]


@parametrize_families
def test_root_testsuites_tag(
    run_and_parse: RunAndParse, xunit_family: _JunitFamily
) -> None:
    def test_x() -> None:
        pass

    record, dom = run_and_parse(test_x, family=xunit_family)
    record.assert_outcomes(passed=1)
    root = dom.get_unique_child
    assert root.tag == "testsuites"
    root.assert_attr(name="pytest tests")
    suite_node = root.get_unique_child
    assert suite_node.tag == "testsuite"


def test_runs_twice(run_and_parse: RunAndParse) -> None:
    def test_pass() -> None:
        pass

    # `--keep-duplicates` plus the same file twice is how the original got one
    # module collected twice; handing the same module object over twice is the
    # ensemble equivalent, and produces the same duplicated nodeids that
    # junitxml has to cope with.
    module = build_module("test_runs_twice", test_pass)
    record, dom = run_and_parse(module, module)
    # was: result.stdout.no_fnmatch_line("*INTERNALERROR*")
    record.assert_outcomes(passed=2)
    first, second = (x["classname"] for x in dom.find_by_tag("testcase"))
    assert first == second


# ensemble: xdist.
def test_runs_twice_xdist(
    pytester: Pytester,
    monkeypatch: MonkeyPatch,
    run_and_parse_pytester: RunAndParsePytester,
) -> None:
    pytest.importorskip("xdist")
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
    f = pytester.makepyfile(
        """
        def test_pass():
            pass
    """
    )

    result, dom = run_and_parse_pytester(f, "--dist", "each", "--tx", "2*popen")
    result.stdout.no_fnmatch_line("*INTERNALERROR*")
    first, second = (x["classname"] for x in dom.find_by_tag("testcase"))
    assert first == second


# ensemble: custom File/Item collectors served from pytest_collect_file, which
# needs the filesystem collection an ensemble replaces.
def test_fancy_items_regression(
    pytester: Pytester, run_and_parse_pytester: RunAndParsePytester
) -> None:
    # issue 1259
    pytester.makeconftest(
        """
        import pytest
        class FunItem(pytest.Item):
            def runtest(self):
                pass
        class NoFunItem(pytest.Item):
            def runtest(self):
                pass

        class FunCollector(pytest.File):
            def collect(self):
                return [
                    FunItem.from_parent(name='a', parent=self),
                    NoFunItem.from_parent(name='a', parent=self),
                    NoFunItem.from_parent(name='b', parent=self),
                ]

        def pytest_collect_file(file_path, parent):
            if file_path.suffix == '.py':
                return FunCollector.from_parent(path=file_path, parent=parent)
    """
    )

    pytester.makepyfile(
        """
        def test_pass():
            pass
    """
    )

    result, dom = run_and_parse_pytester()

    result.stdout.no_fnmatch_line("*INTERNALERROR*")

    items = sorted(
        f"{x['classname']} {x['name']}"
        # dom is a DomNode not a mapping, it's not possible to ** it.
        for x in dom.find_by_tag("testcase")
    )

    import pprint

    pprint.pprint(items)
    assert items == [
        "conftest a",
        "conftest a",
        "conftest b",
        "test_fancy_items_regression a",
        "test_fancy_items_regression a",
        "test_fancy_items_regression b",
        "test_fancy_items_regression test_pass",
    ]


@parametrize_families
def test_global_properties(tmp_path: Path, xunit_family: _JunitFamily) -> None:
    path = tmp_path.joinpath("test_global_properties.xml")
    log = LogXML(str(path), None, family=xunit_family)

    class Report(BaseReport):
        sections: list[tuple[str, str]] = []
        nodeid = "test_node_id"

    log.pytest_sessionstart()
    log.add_global_property("foo", "1")
    log.add_global_property("bar", "2")
    log.pytest_sessionfinish()

    dom = minidom.parse(str(path))

    properties = dom.getElementsByTagName("properties")

    assert properties.length == 1, "There must be one <properties> node"

    property_list = dom.getElementsByTagName("property")

    assert property_list.length == 2, "There most be only 2 property nodes"

    expected = {"foo": "1", "bar": "2"}
    actual = {}

    for p in property_list:
        k = str(p.getAttribute("name"))
        v = str(p.getAttribute("value"))
        actual[k] = v

    assert actual == expected


def test_url_property(tmp_path: Path) -> None:
    test_url = "http://www.github.com/pytest-dev"
    path = tmp_path.joinpath("test_url_property.xml")
    log = LogXML(str(path), None)

    class Report(BaseReport):
        longrepr = "FooBarBaz"
        sections: list[tuple[str, str]] = []
        nodeid = "something"
        location = "tests/filename.py", 42, "TestClass.method"
        url = test_url

    test_report = cast(TestReport, Report())

    log.pytest_sessionstart()
    node_reporter = log._opentestcase(test_report)
    node_reporter.append_failure(test_report)
    log.pytest_sessionfinish()

    test_case = minidom.parse(str(path)).getElementsByTagName("testcase")[0]

    assert test_case.getAttribute("url") == test_url, (
        "The URL did not get written to the xml"
    )


@parametrize_families
def test_record_testsuite_property(
    run_and_parse: RunAndParse, xunit_family: _JunitFamily
) -> None:
    def test_func1(record_testsuite_property: RecordFunc) -> None:
        record_testsuite_property("stats", "all good")

    def test_func2(record_testsuite_property: RecordFunc) -> None:
        record_testsuite_property("stats", 10)

    record, dom = run_and_parse(test_func1, test_func2, family=xunit_family)
    record.assert_outcomes(passed=2)
    node = dom.get_first_by_tag("testsuite")
    properties_node = node.get_first_by_tag("properties")
    p1_node, p2_node = properties_node.find_by_tag(
        "property",
    )[:2]
    p1_node.assert_attr(name="stats", value="all good")
    p2_node.assert_attr(name="stats", value="10")


def test_record_testsuite_property_junit_disabled(tmp_path: Path) -> None:
    def test_func1(record_testsuite_property: RecordFunc) -> None:
        record_testsuite_property("stats", "all good")

    spec = ConfigSpec(rootpath=tmp_path).with_plugins("junitxml")
    record = run_tests(test_func1, spec=spec)
    record.assert_outcomes(passed=1)


@pytest.mark.parametrize("junit", [True, False])
def test_record_testsuite_property_type_checking(tmp_path: Path, junit: bool) -> None:
    def test_func1(record_testsuite_property: RecordFunc) -> None:
        record_testsuite_property(1, 2)  # type: ignore[arg-type]

    args = (f"--junitxml={tmp_path.joinpath('tests.xml')}",) if junit else ()
    spec = ConfigSpec(rootpath=tmp_path, args=args).with_plugins("junitxml")
    record = run_tests(test_func1, spec=spec, capture_output=True)
    record.assert_outcomes(failed=1)
    record.stdout.fnmatch_lines(
        ["*TypeError: name parameter needs to be a string, but int given"]
    )


@pytest.mark.parametrize("suite_name", ["my_suite", ""])
@parametrize_families
def test_set_suite_name(
    suite_name: str,
    run_and_parse: RunAndParse,
    xunit_family: _JunitFamily,
) -> None:
    inicfg: dict[str, object] = {}
    if suite_name:
        inicfg = {"junit_suite_name": suite_name, "junit_family": xunit_family}
        expected = suite_name
    else:
        expected = "pytest"

    def test_func() -> None:
        pass

    record, dom = run_and_parse(
        test_func, inicfg=inicfg, family=xunit_family, suite_name=expected
    )
    record.assert_outcomes(passed=1)
    node = dom.get_first_by_tag("testsuite")
    node.assert_attr(name=expected)


def test_escaped_skipreason_issue3533(run_and_parse: RunAndParse) -> None:
    @pytest.mark.skip(reason="1 <> 2")
    def test_skip() -> None:
        pass

    record, dom = run_and_parse(test_skip)
    record.assert_outcomes(skipped=1)
    node = dom.get_first_by_tag("testcase")
    snode = node.get_first_by_tag("skipped")
    assert "1 <> 2" in snode.text
    snode.assert_attr(message="1 <> 2")


def test_bin_escaped_skipreason(run_and_parse: RunAndParse) -> None:
    """Escape special characters from mark.skip reason (#11842)."""

    @pytest.mark.skip("\33[31;1mred\33[0m")
    def test_skip() -> None:
        pass

    record, dom = run_and_parse(test_skip)
    record.assert_outcomes(skipped=1)
    node = dom.get_first_by_tag("testcase")
    snode = node.get_first_by_tag("skipped")
    assert "#x1B[31;1mred#x1B[0m" in snode.text
    snode.assert_attr(message="#x1B[31;1mred#x1B[0m")


def test_escaped_setup_teardown_error(run_and_parse: RunAndParse) -> None:
    @pytest.fixture
    def my_setup() -> None:
        raise Exception("error: \033[31mred\033[m")

    def test_esc(my_setup: None) -> None:
        pass

    record, dom = run_and_parse(my_setup, test_esc)
    record.assert_outcomes(errors=1)
    node = dom.get_first_by_tag("testcase")
    snode = node.get_first_by_tag("error")
    assert "#x1B[31mred#x1B[m" in snode["message"]
    assert "#x1B[31mred#x1B[m" in snode.text


# ensemble: junit_log_passing_tests only shows in the presence of item-level
# capture; with nothing captured the assertions would hold vacuously.
@parametrize_families
def test_logging_passing_tests_disabled_does_not_log_test_output(
    pytester: Pytester,
    run_and_parse_pytester: RunAndParsePytester,
    xunit_family: _JunitFamily,
) -> None:
    pytester.makeini(
        f"""
        [pytest]
        junit_log_passing_tests=False
        junit_logging=system-out
        junit_family={xunit_family}
    """
    )
    pytester.makepyfile(
        """
        import pytest
        import logging
        import sys

        def test_func():
            sys.stdout.write('This is stdout')
            sys.stderr.write('This is stderr')
            logging.warning('hello')
    """
    )
    result, dom = run_and_parse_pytester(family=xunit_family)
    assert result.ret == 0
    node = dom.get_first_by_tag("testcase")
    assert len(node.find_by_tag("system-err")) == 0
    assert len(node.find_by_tag("system-out")) == 0


# ensemble: needs item-level capture.
@parametrize_families
@pytest.mark.parametrize("junit_logging", ["no", "system-out", "system-err"])
def test_logging_passing_tests_disabled_logs_output_for_failing_test_issue5430(
    pytester: Pytester,
    junit_logging: _JunitLogging,
    run_and_parse_pytester: RunAndParsePytester,
    xunit_family: _JunitFamily,
) -> None:
    pytester.makeini(
        f"""
        [pytest]
        junit_log_passing_tests=False
        junit_family={xunit_family}
    """
    )
    pytester.makepyfile(
        """
        import pytest
        import logging
        import sys

        def test_func():
            logging.warning('hello')
            assert 0
    """
    )
    result, dom = run_and_parse_pytester(
        "-o", f"junit_logging={junit_logging}", family=xunit_family
    )
    assert result.ret == 1
    node = dom.get_first_by_tag("testcase")
    if junit_logging == "system-out":
        assert len(node.find_by_tag("system-err")) == 0
        assert len(node.find_by_tag("system-out")) == 1
    elif junit_logging == "system-err":
        assert len(node.find_by_tag("system-err")) == 1
        assert len(node.find_by_tag("system-out")) == 0
    else:
        assert junit_logging == "no"
        assert len(node.find_by_tag("system-err")) == 0
        assert len(node.find_by_tag("system-out")) == 0


def test_no_message_quiet(tmp_path: Path) -> None:
    """Do not show the summary banner when --quiet is given (#13700)."""

    def test() -> None:
        pass

    xml = tmp_path.joinpath("pytest.xml")
    spec = ConfigSpec(rootpath=tmp_path, args=(f"--junitxml={xml}",)).with_plugins(
        "junitxml"
    )
    record = run_tests(test, spec=spec, capture_output=True)
    record.stdout.fnmatch_lines(["* generated xml file: *"])

    spec = ConfigSpec(
        rootpath=tmp_path, args=(f"--junitxml={xml}", "--quiet")
    ).with_plugins("junitxml")
    record = run_tests(test, spec=spec, capture_output=True)
    record.stdout.no_fnmatch_line("* generated xml file: *")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("junit_logging", "stdout"),
        ("junit_duration_report", "setup"),
        ("junit_family", "xunit3"),
    ],
)
def test_invalid_junit_option_value(tmp_path: Path, name: str, value: str) -> None:
    """Invalid junit option values fail with a clean usage error."""
    spec = ConfigSpec(
        rootpath=tmp_path,
        args=(f"--junitxml={tmp_path.joinpath('junit.xml')}",),
        inicfg={name: value},
    ).with_plugins("junitxml")
    # pytester saw the usage error rendered on stderr and the USAGE_ERROR exit
    # code; the ensemble sees the UsageError itself, raised while junitxml is
    # being configured.
    with pytest.raises(
        UsageError, match=f"config option '{name}' expects one of .*, got '{value}'"
    ):
        run_tests(spec=spec)

# mypy: allow-untyped-defs
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import dataclasses
from typing import Literal
import warnings

from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import parse_warning_filter
from _pytest.config import UsageError
from _pytest.main import Session
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
from _pytest.tracemalloc import tracemalloc_message
from _pytest.warning_defer import DEFER_ACTION
from _pytest.warning_defer import defer_state_key
from _pytest.warning_defer import DeferredWarning
from _pytest.warning_defer import DeferState
from _pytest.warning_defer import install_warning_filter
from _pytest.warning_defer import should_defer
from _pytest.warning_defer import to_deferred
import pytest


@contextmanager
def catch_warnings_for_item(
    config: Config,
    ihook,
    when: Literal["config", "collect", "runtest"],
    item: Item | None,
    *,
    record: bool = True,
) -> Generator[None]:
    """Context manager that catches warnings generated in the contained execution block.

    ``item`` can be None if we are not in the context of an item execution.

    Each warning captured triggers the ``pytest_warning_recorded`` hook.
    """
    with config._catch_configured_warnings(record=record) as log:
        # apply filters from "filterwarnings" marks
        nodeid = "" if item is None else item.nodeid
        state = config.stash.setdefault(defer_state_key, DeferState())
        if item is not None:
            for mark in item.iter_markers(name="filterwarnings"):
                for arg in mark.args:
                    parsed = parse_warning_filter(arg, escape=False)
                    install_warning_filter(parsed)
                    state.filters.append(parsed)

        # record=True means log is not None; mypy can't infer that.
        recording = _Recording(log=log, nodeid=nodeid) if log is not None else None
        recordings = config.stash.setdefault(_recordings_key, [])
        if recording is not None:
            recordings.append(recording)
        try:
            yield
        finally:
            if recording is not None:
                recordings.remove(recording)
                # Anything not already drained by a runtest phase boundary can
                # only be reported at the end of the session.
                state.collected.extend(_drain(config, recording))

            if record:
                # mypy can't infer that record=True means log is not None; help it.
                assert log is not None

                for warning_message in log:
                    ihook.pytest_warning_recorded.call_historic(
                        kwargs=dict(
                            warning_message=warning_message,
                            nodeid=nodeid,
                            when=when,
                            location=None,
                        )
                    )


@dataclasses.dataclass
class _Recording:
    """A live ``catch_warnings(record=True)`` log and how much of it was drained."""

    log: list[warnings.WarningMessage]
    nodeid: str
    cursor: int = 0


#: Active recordings, innermost last.
_recordings_key: StashKey[list[_Recording]] = StashKey()


def _drain(config: Config, recording: _Recording) -> list[DeferredWarning]:
    """Take the deferred warnings recorded since the last drain."""
    state = config.stash[defer_state_key]
    deferred = [
        to_deferred(warning_message, recording.nodeid)
        for warning_message in recording.log[recording.cursor :]
        if should_defer(warning_message, state.filters)
    ]
    recording.cursor = len(recording.log)
    return deferred


def _collect_deferred(item: Item) -> None:
    """Fail the current test phase if it produced deferred warnings."""
    __tracebackhide__ = True
    recordings = item.config.stash.get(_recordings_key, None)
    if not recordings:
        return
    deferred = _drain(item.config, recordings[-1])
    if not deferred:
        return
    if _deferred_report_mode(item.config) == "summary":
        item.config.stash[defer_state_key].collected.extend(deferred)
        return
    plural = "s" if len(deferred) > 1 else ""
    lines = "\n".join(w.format() for w in deferred)
    # The warning's own location is in the message; the frames between here and
    # the emitting code are pytest's, so there is no traceback worth showing.
    fail(f"{len(deferred)} deferred warning{plural}:\n{lines}", pytrace=False)


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item: Item) -> None:
    _collect_deferred(item)


@pytest.hookimpl(trylast=True)
def pytest_runtest_call(item: Item) -> None:
    _collect_deferred(item)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: Item) -> None:
    _collect_deferred(item)


def warning_record_to_str(warning_message: warnings.WarningMessage) -> str:
    """Convert a warnings.WarningMessage to a string."""
    return warnings.formatwarning(
        str(warning_message.message),
        warning_message.category,
        warning_message.filename,
        warning_message.lineno,
        warning_message.line,
    ) + tracemalloc_message(warning_message.source)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item: Item) -> Generator[None, object, object]:
    with catch_warnings_for_item(
        config=item.config, ihook=item.ihook, when="runtest", item=item
    ):
        return (yield)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection(session: Session) -> Generator[None, object, object]:
    config = session.config
    with catch_warnings_for_item(
        config=config, ihook=config.hook, when="collect", item=None
    ):
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
) -> Generator[None]:
    config = terminalreporter.config
    with catch_warnings_for_item(
        config=config, ihook=config.hook, when="config", item=None
    ):
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_sessionfinish(session: Session) -> Generator[None]:
    config = session.config
    with catch_warnings_for_item(
        config=config, ihook=config.hook, when="config", item=None
    ):
        try:
            return (yield)
        finally:
            _report_deferred_warnings(session)


def _report_deferred_warnings(session: Session) -> None:
    """Report deferred warnings that no test phase could be failed for.

    Under ``summary`` this is every deferred warning; under ``eager`` it is the
    ones raised outside a test phase, such as during collection.
    """
    state = session.config.stash.get(defer_state_key, None)
    if state is None or not state.collected:
        return
    terminalreporter = session.config.pluginmanager.getplugin("terminalreporter")
    if terminalreporter is not None:
        terminalreporter.write_sep("=", "deferred warnings", yellow=True, bold=True)
        for deferred in state.collected:
            terminalreporter.write_line(deferred.format(with_nodeid=True))
    if session.exitstatus == ExitCode.OK:
        session.exitstatus = ExitCode.DEFERRED_WARNINGS_ERROR


@pytest.hookimpl(wrapper=True)
def pytest_load_initial_conftests(
    early_config: Config,
) -> Generator[None]:
    with catch_warnings_for_item(
        config=early_config, ihook=early_config.hook, when="config", item=None
    ):
        return (yield)


DEFERRED_REPORT_MODES = ("eager", "summary")


def _deferred_report_mode(config: Config) -> str:
    mode: str = config.getini("deferred_warnings_report")
    if mode not in DEFERRED_REPORT_MODES:
        raise UsageError(
            f"Invalid deferred_warnings_report value {mode!r}, "
            f"expected one of {', '.join(DEFERRED_REPORT_MODES)}"
        )
    return mode


def pytest_report_header(config: Config) -> str | None:
    """Point users configuring ``error`` filters at the ``defer`` action.

    Erroring at the ``warnings.warn()`` call site stays a perfectly good choice,
    so this is advice rather than a warning -- and it is not itself emitted
    through :mod:`warnings`, which an ``error`` filter would turn into a failure.
    """
    if config.pluginmanager.is_blocked("warnings"):
        return None
    actions = set()
    for args, escape in (
        (config.getini("filterwarnings"), False),
        (config.known_args_namespace.pythonwarnings or [], True),
    ):
        for arg in args:
            try:
                actions.add(parse_warning_filter(arg, escape=escape)[0])
            except Exception:
                # Reported properly when the filters are applied.
                continue
    if "error" not in actions or DEFER_ACTION in actions:
        return None
    return (
        "warnings: 'error' filters configured; the 'defer' action instead reports "
        "them once the test finishes, see --help for deferred_warnings_report"
    )


def pytest_configure(config: Config) -> None:
    # Fail early on a bad value rather than once a test defers a warning.
    _deferred_report_mode(config)
    config.addinivalue_line(
        "markers",
        "filterwarnings(warning): add a warning filter to the given test. "
        "see https://docs.pytest.org/en/stable/how-to/capture-warnings.html#pytest-mark-filterwarnings ",
    )

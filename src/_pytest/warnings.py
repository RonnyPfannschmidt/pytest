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
from _pytest.warning_late_error import ERROR_LATER_ACTION
from _pytest.warning_late_error import install_warning_filter
from _pytest.warning_late_error import late_warning_state_key
from _pytest.warning_late_error import LateWarning
from _pytest.warning_late_error import LateWarningState
from _pytest.warning_late_error import should_error_later
from _pytest.warning_late_error import to_late_warning
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
        state = config.stash.setdefault(late_warning_state_key, LateWarningState())
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


def _drain(config: Config, recording: _Recording) -> list[LateWarning]:
    """Take the warnings recorded since the last drain that must error later."""
    state = config.stash[late_warning_state_key]
    late = [
        to_late_warning(warning_message, recording.nodeid)
        for warning_message in recording.log[recording.cursor :]
        if should_error_later(warning_message, state.filters)
    ]
    recording.cursor = len(recording.log)
    return late


def _collect_late_warnings(item: Item) -> None:
    """Fail the current test phase for any warning that must error later."""
    __tracebackhide__ = True
    recordings = item.config.stash.get(_recordings_key, None)
    if not recordings:
        return
    late = _drain(item.config, recordings[-1])
    if not late:
        return
    if _error_later_report_mode(item.config) == "session":
        item.config.stash[late_warning_state_key].collected.extend(late)
        return
    plural = "s" if len(late) > 1 else ""
    lines = "\n".join(w.format() for w in late)
    # The warning's own location is in the message; the frames between here and
    # the emitting code are pytest's, so there is no traceback worth showing.
    fail(
        f"{len(late)} warning{plural} matched an 'error_later' filter:\n{lines}",
        pytrace=False,
    )


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item: Item) -> None:
    _collect_late_warnings(item)


@pytest.hookimpl(trylast=True)
def pytest_runtest_call(item: Item) -> None:
    _collect_late_warnings(item)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: Item) -> None:
    _collect_late_warnings(item)


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
            _report_late_warnings(session)


def _report_late_warnings(session: Session) -> None:
    """Error on the warnings that no test phase could be failed for.

    Under ``session`` that is every one of them; under ``test`` it is the ones
    emitted outside a test phase, such as during collection.
    """
    state = session.config.stash.get(late_warning_state_key, None)
    if state is None or not state.collected:
        return
    terminalreporter = session.config.pluginmanager.getplugin("terminalreporter")
    if terminalreporter is not None:
        terminalreporter.write_sep("=", "late warning errors", yellow=True, bold=True)
        for late in state.collected:
            terminalreporter.write_line(late.format(with_nodeid=True))
    if session.exitstatus == ExitCode.OK:
        session.exitstatus = ExitCode.LATE_WARNING_ERROR


@pytest.hookimpl(wrapper=True)
def pytest_load_initial_conftests(
    early_config: Config,
) -> Generator[None]:
    with catch_warnings_for_item(
        config=early_config, ihook=early_config.hook, when="config", item=None
    ):
        return (yield)


LATE_ERROR_REPORT_MODES = ("test", "session")


def _error_later_report_mode(config: Config) -> str:
    mode: str = config.getini("error_later_report")
    if mode not in LATE_ERROR_REPORT_MODES:
        raise UsageError(
            f"Invalid error_later_report value {mode!r}, "
            f"expected one of {', '.join(LATE_ERROR_REPORT_MODES)}"
        )
    return mode


def pytest_report_header(config: Config) -> str | None:
    """Point users configuring ``error`` filters at the ``error_later`` action.

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
    if "error" not in actions or ERROR_LATER_ACTION in actions:
        return None
    return (
        "warnings: 'error' filters configured; 'error_later' errors on them once the "
        "test finishes instead of inside it, see --help for error_later_report"
    )


def pytest_configure(config: Config) -> None:
    # Fail early on a bad value rather than once a warning matches the filter.
    _error_later_report_mode(config)
    config.addinivalue_line(
        "markers",
        "filterwarnings(warning): add a warning filter to the given test. "
        "see https://docs.pytest.org/en/stable/how-to/capture-warnings.html#pytest-mark-filterwarnings ",
    )

"""Warnings that become errors after the fact.

The ``error`` warning filter raises at the ``warnings.warn()`` call site, which
aborts whatever the code under test was doing halfway through and reports the
failure at the frame that emitted the warning. The ``error_later`` action is
pytest's alternative: the warning still becomes an error, but it is recorded
first, the code under test runs to completion, and pytest raises afterwards at a
defined point.

This module holds the pieces both :mod:`_pytest.config` and :mod:`_pytest.warnings`
need, so that neither has to import the other.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Final
from typing import Literal
import warnings

from _pytest.stash import StashKey


#: Filter action implemented by pytest rather than by the :mod:`warnings` module.
ERROR_LATER_ACTION: Final = "error_later"

#: A parsed warning filter: action, message regex, category, module regex, lineno.
WarningFilter = tuple[
    "warnings._ActionKind | Literal['error_later']", str, type[Warning], str, int
]


@dataclasses.dataclass(frozen=True)
class LateWarning:
    """A warning that matched an ``error_later`` filter, rendered down to plain data.

    Everything is pre-rendered so that no warning instance, and nothing the
    warning referenced, is kept alive until the report is written.
    """

    message: str
    category: str
    filename: str
    lineno: int
    nodeid: str

    def format(self, *, with_nodeid: bool = False) -> str:
        location = f"{self.filename}:{self.lineno}"
        if with_nodeid and self.nodeid:
            location = f"{location} ({self.nodeid})"
        return f"{location}: {self.category}: {self.message}"


@dataclasses.dataclass
class LateWarningState:
    """Per-session state for the ``error_later`` action."""

    #: Filters pytest applied to the current ``catch_warnings`` context, in
    #: application order. Later entries take precedence, as in ``warnings.filters``.
    filters: list[WarningFilter] = dataclasses.field(default_factory=list)
    #: Warnings held over to the end of the session.
    collected: list[LateWarning] = dataclasses.field(default_factory=list)


late_warning_state_key: StashKey[LateWarningState] = StashKey()


def install_warning_filter(filter_: WarningFilter) -> None:
    """Apply a parsed filter to the :mod:`warnings` module.

    ``error_later`` is not an action the :mod:`warnings` module knows about; it is
    installed as ``always`` so that the warning is recorded rather than raised,
    and :func:`should_error_later` decides afterwards what to do with it.
    """
    action, message, category, module, lineno = filter_
    if action == ERROR_LATER_ACTION:
        warnings.filterwarnings("always", message, category, module, lineno)
    else:
        warnings.filterwarnings(action, message, category, module, lineno)


def should_error_later(
    warning_message: warnings.WarningMessage, filters: list[WarningFilter]
) -> bool:
    """Whether a recorded warning matched an ``error_later`` filter.

    ``filters`` is in application order, so it is walked backwards: the last
    filter applied has the highest precedence, exactly as in ``warnings.filters``.

    Filters constrained by ``module`` or ``lineno`` are skipped, because a
    recorded warning carries a filename rather than the emitting module's name.
    Those filters still take full effect inside the :mod:`warnings` module; the
    only thing they cannot do is select ``error_later``. Such filters are rejected
    at parse time if they carry either, so a skipped filter is never a ``error_later``.
    """
    text = str(warning_message.message)
    for action, message, category, module, lineno in reversed(filters):
        if module or lineno:
            continue
        if not issubclass(warning_message.category, category):
            continue
        if message and not re.compile(message).match(text):
            continue
        return action == ERROR_LATER_ACTION
    return False


def to_late_warning(
    warning_message: warnings.WarningMessage, nodeid: str
) -> LateWarning:
    return LateWarning(
        message=str(warning_message.message),
        category=warning_message.category.__name__,
        filename=warning_message.filename,
        lineno=warning_message.lineno,
        nodeid=nodeid,
    )

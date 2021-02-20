"""Deprecation messages and bits of code used elsewhere in the codebase that
is planned to be removed in the next pytest release.

Keeping it in a central location makes it easy to track what is deprecated and should
be removed when the time comes.

All constants defined in this module should be either instances of
:class:`PytestWarning`, or :class:`UnformattedWarning`
in case of warnings which need to format their messages.
"""
from warnings import warn

from _pytest.warning_types import PytestDeprecationWarning
from _pytest.warning_types import PytestRemovedIn8Warning
from _pytest.warning_types import UnformattedWarning

# set of plugins which have been integrated into the core; we use this list to ignore
# them during registration to avoid conflicts
DEPRECATED_EXTERNAL_PLUGINS = {
    "pytest_catchlog",
    "pytest_capturelog",
    "pytest_faulthandler",
}


# This can be* removed pytest 8, but it's harmless and common, so no rush to remove.
# * If you're in the future: "could have been".
YIELD_FIXTURE = PytestDeprecationWarning(
    "@pytest.yield_fixture is deprecated.\n"
    "Use @pytest.fixture instead; they are the same."
)

WARNING_CMDLINE_PREPARSE_HOOK = PytestRemovedIn8Warning(
    "The pytest_cmdline_preparse hook is deprecated and will be removed in a future release. \n"
    "Please use pytest_load_initial_conftests hook instead."
)

FSCOLLECTOR_GETHOOKPROXY_ISINITPATH = PytestRemovedIn8Warning(
    "The gethookproxy() and isinitpath() methods of FSCollector and Package are deprecated; "
    "use self.session.gethookproxy() and self.session.isinitpath() instead. "
)

STRICT_OPTION = PytestRemovedIn8Warning(
    "The --strict option is deprecated, use --strict-markers instead."
)

# This deprecation is never really meant to be removed.
PRIVATE = PytestDeprecationWarning("A private pytest class or function was used.")

ARGUMENT_PERCENT_DEFAULT = PytestRemovedIn8Warning(
    'pytest now uses argparse. "%default" should be changed to "%(default)s"',
)

ARGUMENT_TYPE_STR_CHOICE = UnformattedWarning(
    PytestRemovedIn8Warning,
    "`type` argument to addoption() is the string {typ!r}."
    " For choices this is optional and can be omitted, "
    " but when supplied should be a type (for example `str` or `int`)."
    " (options: {names})",
)

ARGUMENT_TYPE_STR = UnformattedWarning(
    PytestRemovedIn8Warning,
    "`type` argument to addoption() is the string {typ!r}, "
    " but when supplied should be a type (for example `str` or `int`)."
    " (options: {names})",
)

SETUP_CFG_CONFIG = PytestDeprecationWarning(
    "configuring pytest in setup.cfg has been deprecated \n"
    "as pytest and setuptools do not share he same config parser\n"
    "please consider pytest.ini/tox.ini or pyproject.toml"
)


HOOK_LEGACY_PATH_ARG = UnformattedWarning(
    PytestRemovedIn8Warning,
    "The ({pylib_path_arg}: py.path.local) argument is deprecated, please use ({pathlib_path_arg}: pathlib.Path)\n"
    "see https://docs.pytest.org/en/latest/deprecations.html"
    "#py-path-local-arguments-for-hooks-replaced-with-pathlib-path",
)

NODE_CTOR_FSPATH_ARG = UnformattedWarning(
    PytestRemovedIn8Warning,
    "The (fspath: py.path.local) argument to {node_type_name} is deprecated. "
    "Please use the (path: pathlib.Path) argument instead.\n"
    "See https://docs.pytest.org/en/latest/deprecations.html"
    "#fspath-argument-for-node-constructors-replaced-with-pathlib-path",
)

WARNS_NONE_ARG = PytestRemovedIn8Warning(
    "Passing None has been deprecated.\n"
    "See https://docs.pytest.org/en/latest/how-to/capture-warnings.html"
    "#additional-use-cases-of-warnings-in-tests"
    " for alternatives in common use cases."
)

KEYWORD_MSG_ARG = UnformattedWarning(
    PytestRemovedIn8Warning,
    "pytest.{func}(msg=...) is now deprecated, use pytest.{func}(reason=...) instead",
)

INSTANCE_COLLECTOR = PytestRemovedIn8Warning(
    "The pytest.Instance collector type is deprecated and is no longer used. "
    "See https://docs.pytest.org/en/latest/deprecations.html#the-pytest-instance-collector",
)

# You want to make some `__init__` or function "private".
#
#   def my_private_function(some, args):
#       ...
#
# Do this:
#
#   def my_private_function(some, args, *, _ispytest: bool = False):
#       check_ispytest(_ispytest)
#       ...
#
# Change all internal/allowed calls to
#
#   my_private_function(some, args, _ispytest=True)
#
# All other calls will get the default _ispytest=False and trigger
# the warning (possibly error in the future).


def check_ispytest(ispytest: bool) -> None:
    if not ispytest:
        warn(PRIVATE, stacklevel=3)

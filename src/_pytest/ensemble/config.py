"""Nested config construction for ensembles (EXPERIMENTAL)."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
import contextlib
import copy
import dataclasses
import pathlib
from typing import Final
from typing import TextIO
import warnings

from _pytest.config import Config
from _pytest.config import essential_plugins
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.config.findpaths import ConfigValue
from _pytest.config.findpaths import parse_override_ini
from _pytest.stash import StashKey
from _pytest.terminal import terminal_file_key
from _pytest.tmpdir import TempPathFactory
from _pytest.unraisableexception import gc_collect_iterations_key


#: Warnings raised while the ensemble config was being configured or
#: unconfigured. They are recorded rather than allowed out, so that an
#: ensemble cannot make its *host* fail - this suite runs with
#: ``filterwarnings = error``.
config_warnings_key = StashKey[list[warnings.WarningMessage]]()


#: Plugins loaded into an ensemble config by default: the essential core
#: plus the plugins that give tests their usual semantics, deliberately
#: excluding everything that renders output, captures io, or installs
#: process-global state (terminal, capture, cacheprovider, assertion,
#: debugging, faulthandler, logging, threadexception, unraisableexception, ...).
#:
#: ``tmpdir`` is not here either: it would allocate the ensemble its own
#: numbered base temp directory under the global ``pytest-of-<user>`` root,
#: which means scanning that root once per ensemble and leaving a directory
#: behind for every one ever configured. It is loaded only when
#: :attr:`ConfigSpec.tmp_path_factory` supplies a preconfigured factory.
DEFAULT_PLUGINS: Final[tuple[str, ...]] = (
    *essential_plugins,  # mark, main, runner, fixtures, helpconfig
    "python",
    "skipping",
    "warnings",
    "reports",
    "unittest",
    "monkeypatch",
    "recwarn",
    # Not for the rewriting - that is installed from Config._preparse, which
    # an ensemble never runs - but for the failure *explanation*. Without
    # this plugin ``assertion.util._reprcompare`` stays bound to whatever the
    # host installed, so an ensemble silently renders its assertions with the
    # host's verbosity, ini values and pytest_assertrepr_compare hooks.
    "assertion",
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigSpec:
    """Declarative description of a nested pytest configuration.

    A spec is plain data: build it in a fixture, parametrize it, or derive
    variants via :meth:`replace`/:meth:`with_plugins`. State is only
    acquired when the spec is passed to :func:`configured`.
    """

    #: Root directory anchoring nodeids and synthetic module paths.
    #: Must be an existing directory; it is never read from or written to.
    rootpath: pathlib.Path | None = None

    #: Command line arguments, taken verbatim (never coerced to paths).
    args: tuple[str, ...] = ()

    #: Ini configuration values, authoritative (no config file is read).
    #: Values are plain ``str``/``list[str]`` (ini mode) or preconstructed
    #: :class:`ConfigValue` instances.
    inicfg: Mapping[str, object] = dataclasses.field(default_factory=dict)

    #: Built-in plugin names to load.
    plugins: tuple[str, ...] = DEFAULT_PLUGINS

    #: Additional plugins: importable module names or plugin objects.
    #: Plugin objects are the ensemble replacement for conftest files.
    extra_plugins: tuple[str | object, ...] = ()

    #: Invocation directory; defaults to ``rootpath``, never the implicit cwd.
    invocation_dir: pathlib.Path | None = None

    #: Not supported yet; ensemble configs never load conftest files.
    load_conftests: bool = False

    #: Preconfigured temp path factory. Supplying one loads the ``tmpdir``
    #: plugin and binds this factory instead of the one it would build from
    #: the config, so an ensemble's ``tmp_path`` lives wherever the caller
    #: decided - normally inside the *host* test's own ``tmp_path``, which
    #: costs no root scan and is cleaned up with the host. Build one with
    #: :func:`make_tmp_path_factory`.
    tmp_path_factory: TempPathFactory | None = None

    #: How many ``gc.collect()`` passes ``unraisableexception`` makes, when
    #: that plugin is opted into at all. It is not in :data:`DEFAULT_PLUGINS`,
    #: and even when loaded an ensemble does not collect by default: the heap
    #: it would walk is the *host* process's, so a full pass costs whatever
    #: the host happens to be holding rather than anything the ensemble owns.
    #: Raise it only for a test that needs finalizers flushed before an
    #: unraisable exception can surface.
    gc_collect_iterations: int = 0

    #: Stream the terminal plugin writes to, when it is loaded at all. An
    #: ensemble must never be given the stdout of whatever is running it,
    #: so this is bound at construction rather than redirected around it.
    output: TextIO | None = None

    def replace(self, **kw: object) -> ConfigSpec:
        return dataclasses.replace(self, **kw)  # type: ignore[arg-type]

    def with_plugins(self, *plugins: str | object) -> ConfigSpec:
        """Return a spec with the given plugins added.

        Built-in plugin names extend :attr:`plugins`; anything else is
        appended to :attr:`extra_plugins`.
        """
        from _pytest.config import builtin_plugins

        names = tuple(p for p in plugins if isinstance(p, str) and p in builtin_plugins)
        extras = tuple(
            p for p in plugins if not (isinstance(p, str) and p in builtin_plugins)
        )
        return self.replace(
            plugins=self.plugins + names,
            extra_plugins=self.extra_plugins + extras,
        )

    def without_plugins(self, *names: str) -> ConfigSpec:
        """Return a spec with the given built-in plugin names removed."""
        return self.replace(plugins=tuple(p for p in self.plugins if p not in names))


def make_tmp_path_factory(basetemp: pathlib.Path) -> TempPathFactory:
    """Build a :class:`TempPathFactory` for an ensemble, rooted at *basetemp*.

    *basetemp* is treated the way ``--basetemp`` is: it is removed if it
    already exists, then created. Pass a path that is yours to destroy - a
    subdirectory of the host test's ``tmp_path`` is the intended use.

    The point is what this does *not* do. A factory built from a config
    allocates a numbered directory under the global ``pytest-of-<user>``
    root, which scans that root and every sibling run's leftovers, registers
    a cleanup lock, and leaves the directory behind afterwards. Ensembles are
    built in the hundreds, so paying that per ensemble is not viable.
    """
    return TempPathFactory(
        given_basetemp=basetemp,
        trace=lambda *args, **kwargs: None,
        retention_count=0,
        retention_policy="all",
        _ispytest=True,
    )


class _BindTmpPathFactory:
    """Bind a caller-supplied factory over the one ``tmpdir`` builds.

    The ``tmpdir`` plugin creates its factory in ``pytest_configure``; this
    runs last and replaces it, so the fixtures, the retention handling and
    the ``pytest_sessionfinish`` cleanup all stay the plugin's own.
    """

    __pytest_no_fixtures__ = True

    def __init__(self, factory: TempPathFactory) -> None:
        self._factory = factory

    @hookimpl(trylast=True)
    def pytest_configure(self, config: Config) -> None:
        config._tmp_path_factory = self._factory  # type: ignore[attr-defined]


def _own(value: object) -> ConfigValue:
    """Wrap a spec's ini value in a ConfigValue the config may own.

    Mutable values are copied: ``Config.addinivalue_line`` appends to the
    cached list, and the cache would otherwise hold the caller's own object,
    so a reused (frozen!) spec would grow every time it was configured.
    """
    if isinstance(value, ConfigValue):
        if isinstance(value.value, list):
            value = dataclasses.replace(value, value=list(value.value))
        return value
    if isinstance(value, list):
        value = list(value)
    return ConfigValue(value, origin="file", mode="ini")


@contextlib.contextmanager
def configured(spec: ConfigSpec) -> Iterator[Config]:
    """Build a parsed *and* configured :class:`Config` from a spec.

    The config is constructed from the spec's explicit values through the
    same parse phases a command line invocation uses, but without rootdir
    discovery, config file reading, conftest loading, plugin autoloading,
    or environment variable consultation.

    On exit, ``pytest_unconfigure`` and the config cleanup stack run.
    """
    if spec.rootpath is None:
        raise ValueError("ConfigSpec.rootpath is required to build a config")
    if spec.load_conftests:
        raise NotImplementedError(
            "loading conftest files is not supported in ensemble configs yet"
        )
    if not spec.rootpath.is_dir():
        raise ValueError(f"ConfigSpec.rootpath is not a directory: {spec.rootpath}")
    missing = [name for name in essential_plugins if name not in spec.plugins]
    if missing:
        raise ValueError(
            f"ConfigSpec.plugins must include the essential plugins, missing: {missing}"
        )

    invocation_dir = (
        spec.invocation_dir if spec.invocation_dir is not None else spec.rootpath
    )
    pluginmanager = PytestPluginManager()
    config = Config(
        pluginmanager,
        invocation_params=Config.InvocationParams(
            args=spec.args,
            plugins=spec.extra_plugins or None,
            dir=invocation_dir,
        ),
    )
    # Before anything that can fail: the teardown below reads this, and a
    # UsageError raised while parsing args would otherwise surface as a
    # KeyError with the real error buried in __context__.
    config.stash[config_warnings_key] = []
    try:
        for name in spec.plugins:
            pluginmanager.import_plugin(name)
        if spec.tmp_path_factory is not None:
            pluginmanager.import_plugin("tmpdir")
            pluginmanager.register(_BindTmpPathFactory(spec.tmp_path_factory))
        for plugin in spec.extra_plugins:
            if isinstance(plugin, str):
                pluginmanager.import_plugin(plugin)
            else:
                pluginmanager.register(plugin)

        config.hook.pytest_addhooks.call_historic(
            kwargs=dict(pluginmanager=pluginmanager)
        )
        inicfg = {name: _own(value) for name, value in spec.inicfg.items()}
        config._apply_rootdir(
            rootpath=spec.rootpath,
            inipath=None,
            inicfg=inicfg,
            ignored_config_files=(),
        )
        config._register_core_ini_options()

        # Mirror the addopts/override-ini handling of Config.parse: without
        # it every OverrideIniAction option (--strict-markers, --strict-config,
        # ...) and every -o name=value would be parsed into the namespace and
        # then silently dropped, so a spec asking for them would get a config
        # that quietly ignores them.
        args = config._validate_args(config.getini("addopts"), "via addopts config")
        args += spec.args
        config.known_args_namespace = config._parser.parse_known_args(
            args, namespace=copy.copy(config.option)
        )
        if overrides := parse_override_ini(config.known_args_namespace.override_ini):
            config._inicfg.update(overrides)
            config._inicache.clear()

        config._finalize_parse(args, decide_args=False)
        # Read by ``unraisableexception`` at configure, cleanup and
        # unconfigure time; harmless when that plugin was not opted into.
        config.stash[gc_collect_iterations_key] = spec.gc_collect_iterations
        if spec.output is not None:
            # Must be stashed before configure: the terminal reporter binds
            # its stream when it is constructed, and must never bind ours.
            config.stash[terminal_file_key] = spec.output
        with warnings.catch_warnings(record=True) as caught:
            config._do_configure()
        config.stash[config_warnings_key].extend(caught)
        yield config
    finally:
        with warnings.catch_warnings(record=True) as caught:
            config._ensure_unconfigure()
        config.stash[config_warnings_key].extend(caught)

"""Compare pytester and ``_pytest.ensemble`` on the *same real files*.

``bench/ensemble_vs_pytester.py`` compares the two harnesses on generated
sources, which measures the harness but says nothing about the usual shape
of a pytester test: take a script that already exists under
``testing/example_scripts``, put it somewhere, and run it.

The two ways of doing that are:

``copy`` + run
    What ``pytester.copy_example`` does: put the file in the pytester
    tmpdir, then run it with ``runpytest_inprocess``/``inline_run``/
    ``runpytest_subprocess``. The script is imported from the copy, so its
    reported paths are the copy's. The copy is done with ``shutil.copy``
    rather than ``copy_example`` itself, which picks one fixed destination
    and so cannot be called in a loop.
``module_from_path``
    ``module_from_path(path)`` imports the script where it lives - without
    registering it in ``sys.modules`` or writing bytecode beside it - and
    ``run_tests`` collects the resulting module. The script is never
    copied, and its items report their real paths.

Both run the same code, so the difference is the harness, not the work.

Run with::

    python bench/ensemble_vs_pytester_examples.py
    pytest bench/ensemble_vs_pytester_examples.py -s    # equivalent
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import io
from pathlib import Path
import shutil
import time

from _pytest.ensemble import module_from_path
from _pytest.ensemble import run_tests
from _pytest.pytester import Pytester


#: Real example scripts, relative to ``testing/example_scripts``. Examples
#: that deliberately fail at import time are not benchmark subjects - the
#: two harnesses would not be doing the same work.
EXAMPLES = [
    "fixtures/fill_fixtures/test_funcarg_basic.py",
    "fixtures/fill_fixtures/test_funcarg_lookup_modulelevel.py",
    "fixtures/fill_fixtures/test_funcarg_lookup_classlevel.py",
    "fixtures/fill_fixtures/test_extend_fixture_module_class.py",
    "unittest/test_setup_skip.py",
    "unittest/test_setup_skip_class.py",
    "unittest/test_setup_skip_module.py",
]

ITERATIONS = {"subprocess": 3}
DEFAULT_ITERATIONS = 10

EXAMPLE_ROOT = Path(__file__).parent.parent / "testing" / "example_scripts"


def _files(path: Path) -> int:
    """Files below *path*, ignoring bytecode caches."""
    return sum(
        1
        for p in path.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )


def _measure(
    run: Callable[[int], None], iterations: int, watched: Path
) -> tuple[float, float]:
    """Return (seconds, files created below *watched*) per iteration."""
    with contextlib.redirect_stdout(io.StringIO()):
        run(0)  # warm up
        before = _files(watched)
        start = time.perf_counter()
        for i in range(1, iterations + 1):
            run(i)
        elapsed = time.perf_counter() - start
    return elapsed / iterations, (_files(watched) - before) / iterations


def _arms(pytester: Pytester, rel: str) -> dict[str, Callable[[int], None]]:
    source = EXAMPLE_ROOT / rel

    def copy(i: int) -> Path:
        # Each iteration gets its own directory, which is what a real
        # pytester test gets too.
        target = pytester.path / f"run{i}"
        target.mkdir(exist_ok=True)
        dest = target / source.name
        shutil.copy(source, dest)
        return dest

    def subprocess_(i: int) -> None:
        pytester.runpytest_subprocess(copy(i))

    def inprocess(i: int) -> None:
        pytester.runpytest_inprocess(copy(i))

    def inline(i: int) -> None:
        pytester.inline_run(copy(i))

    def copy_only(i: int) -> None:
        copy(i)

    def ensemble(i: int) -> None:
        run_tests(module_from_path(source), rootpath=source.parent)

    def import_only(i: int) -> None:
        module_from_path(source)

    return {
        "subprocess": subprocess_,
        "inprocess": inprocess,
        "inline": inline,
        "ensemble": ensemble,
        "copy only": copy_only,
        "import only": import_only,
    }


def test_ensemble_vs_pytester_on_examples(pytester: Pytester) -> None:
    """Not an assertion test - run with ``-s`` and read the table."""
    print()
    totals: dict[str, float] = {}
    for rel in EXAMPLES:
        results = {}
        for name, run in _arms(pytester, rel).items():
            iterations = ITERATIONS.get(name, DEFAULT_ITERATIONS)
            results[name] = _measure(run, iterations, pytester.path)
            totals[name] = totals.get(name, 0.0) + results[name][0]

        baseline = results["ensemble"][0]
        print(f"\n{rel}")
        print(f"  {'arm':<12} {'per run':>10} {'files':>8} {'vs ensemble':>13}")
        for name, (seconds, files) in results.items():
            ratio = f"{seconds / baseline:.1f}x" if baseline else "n/a"
            print(f"  {name:<12} {seconds * 1000:>8.2f}ms {files:>8.1f} {ratio:>13}")

    print(f"\n{'=' * 56}\nacross all {len(EXAMPLES)} examples")
    base = totals["ensemble"]
    print(f"  {'arm':<12} {'total':>10} {'vs ensemble':>13}")
    for name, seconds in totals.items():
        print(f"  {name:<12} {seconds * 1000:>8.2f}ms {seconds / base:>12.1f}x")
    print(
        "\n(files counted below the pytester tmpdir, excluding __pycache__;\n"
        " the ensemble arm writes nothing and never leaves the source tree)"
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-s", "-q", "-p", "no:randomly"]))

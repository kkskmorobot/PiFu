"""Redirect copied historical entrypoints to the portable, repeatable runners.

The original implementations remain in place for provenance and reuse. This
utility deliberately refuses the source project and only edits an explicitly
provided reproduction copy. It is used during packaging, not for training.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

RUNS = ("run18", "run22", "run24", "run21", "run23", "run25", "run26", "run27", "run28", "run33", "lg1", "lg2", "lgx", "lg3", "ab1", "ab2")


def redirect(path, arguments, legacy_prepare):
    text = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    guards = [node for node in tree.body if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"]
    if len(guards) != 1:
        raise RuntimeError(f"Expected one __main__ entrypoint: {path}")
    node = guards[0]
    lines = text.splitlines(keepends=True)
    replacement = (
        'if __name__ == "__main__":\n'
        '    # Portable copy: reruns use new outputs; historical release gates are audit records.\n'
        '    import sys as _repro_sys\n'
        '    from pathlib import Path as _ReproPath\n'
        '    _repro_sys.path.insert(0, str(_ReproPath(__file__).resolve().parents[2] / "reproduction"))\n'
        '    from run import main as _repro_main\n'
        f'    _repro_args = [{arguments}] + [("--prepare" if a == {legacy_prepare!r} else a) for a in _repro_sys.argv[1:]]\n'
        '    _repro_main(_repro_args)\n'
    )
    result = "".join(lines[:node.lineno - 1]) + replacement + "".join(lines[node.end_lineno:])
    ast.parse(result)
    path.write_text(result, encoding="utf-8", newline="\n")


def patch_entrypoints(project):
    project = Path(project).resolve()
    forbidden = Path(r"E:\App\DaiMa\JetBrains\Project\PiFuV2").resolve()
    if project == forbidden or forbidden in project.parents:
        raise RuntimeError("Source project must never be modified")
    for run in RUNS:
        path = project / "tools" / run / f"train_{run}_from_pycharm.py"
        redirect(path, f'"train", "--run", "{run}"', "--prepare-only")
    for suite in ("test1", "test2"):
        redirect(project / "tools" / suite / "test.py", f'"evaluate", "--suite", "{suite}"', "--prepare")
    return 18


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    print(f"Redirected {patch_entrypoints(args.project)} entrypoints in {args.project.resolve()}")

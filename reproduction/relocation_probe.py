"""Run a command while refusing to read the old project's code, data, or weights."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

OLD = os.path.normcase(r'E:\App\DaiMa\JetBrains\Project\PiFuV2')


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
    if path.startswith(OLD + os.sep) and not path.startswith(OLD + os.sep + '.venv' + os.sep):
        raise RuntimeError(f'Relocation check blocked old project access: {path}')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    if len(sys.argv)<2:
        raise SystemExit('Usage: python reproduction/relocation_probe.py train|evaluate|validation|check ...')
    sys.argv = [str(Path(__file__).with_name('run.py'))] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')

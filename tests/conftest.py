"""Make `import data_prep` work in tests by loading the data-prep notebook.

data_prep is a Jupyter notebook (data_prep.ipynb). Here we read its code cells,
execute them into a fresh module, and register it as `data_prep` in sys.modules
so the test file can import it like any other module.
"""

import os
import sys
import types
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # the notebook resolves paths from Path.cwd()


def _load_notebook_as_module(name: str, path: Path) -> types.ModuleType:
    nb = nbformat.read(path, as_version=4)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    code = "\n\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    exec(compile(code, str(path), "exec"), module.__dict__)  # noqa: S102
    sys.modules[name] = module
    return module


# Run from the repo root so the notebook's Path.cwd()-based paths resolve.
_load_notebook_as_module("data_prep", ROOT / "data_prep.ipynb")

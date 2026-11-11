"""Optional companion code (baselines and paper analyses) that is not part of the REWIND release.

The public pipeline runs without it; where a stage can also compute baseline signals, it imports the
companion module here and skips that part with a one-line note when the module is not present.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

_NOTED = set()


def optional_module(name: str, what: str) -> Optional[ModuleType]:
    """Import `name`, or return None (printing once that `what` is skipped) when its package is absent."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if not error.name or not (name == error.name or name.startswith(error.name + ".")):
            raise  # a missing dependency inside an existing module is a real error
        if name not in _NOTED:
            _NOTED.add(name)
            print(f"[rewind] {name} not found: skipping {what}", flush=True)
        return None

import json
from pathlib import Path
from typing import Any


def get_root() -> Path:
    """
    Return the root directory of the project.
    """
    ROOT = Path(__file__).resolve().parents[3]
    return ROOT


def read_json(fname: str | Path) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Read a json file.
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=dict)


def write_json(content: Any, fname: str | Path) -> None:
    """
    Write the content to a json file.
    """
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

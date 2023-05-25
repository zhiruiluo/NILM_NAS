from __future__ import annotations

from pathlib import Path

__all__ = [
    "get_project_root",
    "get_relative_path_to_project_root",
]


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_relative_path_to_project_root(path) -> Path:
    return Path(path).relative_to(get_project_root())

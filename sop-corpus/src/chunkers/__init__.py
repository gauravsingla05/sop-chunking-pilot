"""Chunker registry.

Adding a new chunker: import its module here and add it to `ALL`.
Each module must expose: NAME (str) and chunk(doc) -> list[Chunk].
"""

from . import fixed, recursive, semantic, layout_aware, pac

ALL = {
    fixed.NAME:        fixed,
    recursive.NAME:    recursive,
    semantic.NAME:     semantic,
    layout_aware.NAME: layout_aware,
    pac.NAME:          pac,
}


def get(name: str):
    if name not in ALL:
        raise KeyError(f"unknown chunker {name!r}; choices: {sorted(ALL)}")
    return ALL[name]

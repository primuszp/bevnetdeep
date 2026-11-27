"""Shared PEP 562 lazy-attribute plumbing for the package ``__init__`` files.

Each package maps public names to the submodule that defines them, so importing
the package stays cheap: a submodule (and its dependencies) is only imported
when one of its symbols is first accessed.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any


def lazy_accessors(
    package: str,
    lazy: dict[str, str],
    namespace: dict[str, Any],
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build the ``__getattr__``/``__dir__`` pair for a lazily loaded package.

    Args:
        package: the importing package's ``__name__``.
        lazy: public symbol -> defining submodule, relative to *package*.
        namespace: the package's ``globals()``, used to cache resolved symbols.

    Returns:
        ``(__getattr__, __dir__)`` to be bound in the package module.
    """

    def __getattr__(name: str) -> Any:
        module_name = lazy.get(name)
        if module_name is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(import_module(f".{module_name}", package), name)
        namespace[name] = value  # cache: subsequent lookups skip __getattr__
        return value

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(lazy))

    return __getattr__, __dir__

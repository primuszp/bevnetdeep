"""Run-local diagnostics and warning routing."""
from __future__ import annotations

import time
import warnings
from pathlib import Path


def setup_diagnostics(out_dir: Path):
    """Route warnings and diagnostics to ``<run>/diagnostics.log``."""
    diag_path = out_dir / "diagnostics.log"
    diag_path.parent.mkdir(parents=True, exist_ok=True)

    def diag(level: str, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with diag_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] [{level.upper()}] {message}\n")

    def showwarning(message, category, filename, lineno, file=None, line=None):
        _ = (file, line)  # Required by the warnings.showwarning callback contract.
        diag("WARNING", f"{category.__name__}: {message} ({filename}:{lineno})")

    warnings.showwarning = showwarning
    return diag_path, diag

"""Run the executable cells of lab.ipynb without requiring Jupyter.

The notebook remains the primary deliverable.  This helper is useful on a
minimal Python installation and runs exactly its code cells in order.
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path

import matplotlib

# The helper is non-interactive.  Notebook execution still renders figures
# inline, while this runner writes the same PNG files without blocking on show.
matplotlib.use("Agg")


def display(value):
    """Notebook-like display fallback for tabular values."""
    print(value.to_string(index=False) if hasattr(value, "to_string") else value)


builtins.display = display
notebook_path = Path(__file__).with_name("lab.ipynb")
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__", "__file__": str(notebook_path)}
for position, cell in enumerate(notebook["cells"], start=1):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    print(f"\n===== cell {position} =====")
    exec(compile(source, f"{notebook_path}:cell-{position}", "exec"), namespace)

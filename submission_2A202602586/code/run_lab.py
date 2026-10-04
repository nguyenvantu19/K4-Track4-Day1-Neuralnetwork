"""Run lab.ipynb in a fresh kernel and save real outputs (Restart & Run All).

Run from any working directory. Requires nbclient, nbformat and ipykernel.
"""
from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient


def main():
    notebook_path = Path(__file__).with_name("lab.ipynb").resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    runtime_dir = notebook_path.parents[2] / ".lab_runtime"
    runtime_dir.mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    os.environ["IPYTHONDIR"] = str(runtime_dir / "ipython")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

    def report_cell(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    client = NotebookClient(
        notebook, timeout=1800, kernel_name="python3", allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
        on_cell_start=report_cell,
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    print(f"Saved executed notebook: {notebook_path}", flush=True)


if __name__ == "__main__":
    main()

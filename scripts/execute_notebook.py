"""Execute the completed notebook with a timeout and preserve outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


parser = argparse.ArgumentParser()
parser.add_argument(
    "notebook", nargs="?", default="Project-Definition2-traffic-vehicle.ipynb"
)
parser.add_argument("--timeout", type=int, default=1800)
parser.add_argument("--kernel", default="traffic-vehicle-python312")
args = parser.parse_args()

path = Path(args.notebook).resolve()
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=args.timeout,
    kernel_name=args.kernel,
    resources={"metadata": {"path": str(path.parent)}},
)
client.execute()
nbformat.write(notebook, path)
print(f"Executed {path}")

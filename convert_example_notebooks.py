import os
import jupytext
from pathlib import Path

EXAMPLE_DIR = os.path.join(".", "examples")


for notebook_md_file in Path(EXAMPLE_DIR).rglob("*.md"):
    if os.path.basename(notebook_md_file) == "README.md":
        continue
    md_notebook = jupytext.read(notebook_md_file)
    path_without_ext = os.path.splitext(notebook_md_file)[0]
    nb_out_path = f"{path_without_ext}.ipynb"
    jupytext.write(md_notebook, nb_out_path)

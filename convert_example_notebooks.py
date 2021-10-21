import os
import jupytext
from glob import glob

EXAMPLE_DIR = os.path.join(".", "examples")
GLOB_PATTERN = os.path.abspath(os.path.join(EXAMPLE_DIR, "*.md"))


for notebook_md_file in glob(GLOB_PATTERN):
    md_notebook = jupytext.read(notebook_md_file)
    path_without_ext = os.path.splitext(notebook_md_file)[0]
    nb_out_path = os.path.join(EXAMPLE_DIR, f"{path_without_ext}.ipynb")
    jupytext.write(md_notebook, nb_out_path)

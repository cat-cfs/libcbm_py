# Copyright (C) Her Majesty the Queen in Right of Canada,
# as represented by the Minister of Natural Resources Canada

import git
import os
import sys


def load_cbm3_python(branch="master"):
    """Clone the cbm3_python repository from Github if it does not exist locally
    already.  If it does already exist locally do a git pull to get the latest
    revisions. Appends the cb3_python dir to sys.path

    Args:
        branch (str, optional): The git branch to clone or pull. Defaults
            to "master".

    Returns:
        str: the directory where cbm3_python is cloned
    """
    cbm3_python_dir = os.path.abspath(os.path.join(".", "cbm3_python_git"))
    if not os.path.exists(cbm3_python_dir):
        git.Repo.clone_from(
            'https://github.com/cat-cfs/cbm3_python.git', cbm3_python_dir,
            branch=branch)
    else:
        g = git.cmd.Git(cbm3_python_dir)
        g.pull()
        if branch:
            g.checkout(branch)

    if cbm3_python_dir not in sys.path:
        sys.path.append(os.path.abspath(cbm3_python_dir))

    return cbm3_python_dir


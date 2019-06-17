#Copyright (C) Her Majesty the Queen in Right of Canada,
#as represented by the Minister of Natural Resources Canada

import git
import os,sys
def load_cbm3_python(branch="master"):
    '''
    Clone the cbm3_python repository from Github if it does not exist locally already.  
    If it does already exist locally do a git pull to get the latest revisions
    '''
    cbm3_python_dir = os.path.abspath(os.path.join(".","cbm3_python_git"))
    if not os.path.exists(cbm3_python_dir):
        git.Repo.clone_from('https://github.com/cat-cfs/cbm3_python.git', cbm3_python_dir, branch=branch)
    else:
        g = git.cmd.Git(cbm3_python_dir)
        g.pull()
        if branch:
            g.checkout(branch)
    
    if not cbm3_python_dir in sys.path:
        sys.path.append(os.path.abspath(cbm3_python_dir))

    return cbm3_python_dir


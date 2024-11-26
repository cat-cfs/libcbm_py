# Running `libcbm` examples in `python` with `jupyter` or `R` with `R-markdown`

A step by step guide on one of many methods to run the `jupyter` notebook examples in this directory.

## Python

### Download and Install `python` and `git`

[https://www.python.org/downloads/](https://www.python.org/downloads/)

[https://git-scm.com/downloads](https://git-scm.com/downloads)

## (optional) Set up a python virtual environment

See [Creation of virtual environments](https://docs.python.org/3/library/venv.html)

```
python -m venv my_venv
```

### Install Packages

Install the following packages.  If using a virtual environment, activate the environment first.

* [`jupyter`](https://jupyter.org/)
* [`jupytext`](https://github.com/mwouts/jupytext) - lets `jupyter` load markdown formatted notebooks
* [`matplotlib`](https://matplotlib.org/) - for plotting graphs

```
pip install jupyter
pip install jupytext
pip install matplotlib
```

### Clone and install `libcbm`

Install

```
pip install libcbm
```

OR Install from github source

```
git clone https://github.com/cat-cfs/libcbm_py
cd libcbm_py
pip install .
```



### Run `jupyter` from the examples directory

Any of the  [example notebooks](https://github.com/cat-cfs/libcbm_py/tree/master/examples) can be run via the `jupyter` user interface

```
cd libcbm_py/examples
jupyter notebook
```

## R

See the `*.rmd` formatted examples in this directory

Calling libcbm functions, which are coded in `python`, from `R` requires the `Reticulate` package.

See https://rstudio.github.io/reticulate/

Example method to set up python on your system:

```R
library(reticulate)
version <- "3.10:latest"
install_python(version)
virtualenv_create("my-environment", version = version)
use_virtualenv("my-environment")
```

In subsequent R sessions you can re-use the environment: it is not necessary to use `install_python` and `virtualenv_create` for each R session.

```R
library(reticulate)
use_virtualenv("my-environment")
```

Now use the RStudio terminal to install libcbm_py using pip:

```
python -m pip install libcbm
```

If this gives an error, it may be because the python version installed in `my-environment` is incorrect.
To check that your R virtual environment is using the correct version of python, type this into the terminal:

```
python -m pip --version
```

Should return:
`pip 24.3.1 from C:\Users\user\Documents\.virtualenvs\my-environment\lib\site-packages\pip (python 3.10)`
(The version number of pip, the path to the version of pip being called, and the version of python calling pip)

If this shows the wrong version of python, you can delete the virtual environment folder (e.g. `C:\Users\user\Documents\.virtualenvs\my-environment`) and run `install_python("3.10:latest")` and `virtualenv_create("my-environment", version = "3.10:latest")` again to reinstall the correct version of python.

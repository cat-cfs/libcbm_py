# Running `libcbm` examples in `jupyter`

A step by step guide on one of many methods to run the `jupyter` notebook examples in this directory.  

## Download and Install `python` and `git`

[https://www.python.org/downloads/](https://www.python.org/downloads/)

[https://git-scm.com/downloads](https://git-scm.com/downloads)

## (optional) Set up a python virtual environment

See [Creation of virtual environments](https://docs.python.org/3/library/venv.html)

```
python -m venv my_venv
```

## Install Packages 

Install the following packages.  If using a virtual environment, activate the environment first.

* [`jupyter`](https://jupyter.org/)
* [`jupytext`](https://github.com/mwouts/jupytext) - lets `jupyter` load markdown formatted notebooks
* [`matplotlib`](https://matplotlib.org/) - for plotting graphs

```
pip install jupyter
pip install jupytext
pip install matplotlib
```

## Clone and install `libcbm`

```
git clone https://github.com/cat-cfs/libcbm_py
cd libcbm_py
pip install .
```

## Run `jupyter` from the examples directory

Any of the  [example notebooks](https://github.com/cat-cfs/libcbm_py/tree/master/examples) can be run via the `jupyter` user interface

```
cd libcbm_py/examples
jupyter notebook
```




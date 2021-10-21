:: note to include numba routines in coverage, set disable it with the option
SET NUMBA_DISABLE_JIT=1
:: can also be done in the .numba_config.yml file in this dir
pytest --cov=libcbm --cov-report html
python ./convert_example_notebooks.py
:: make the notebooks look at the local dir
SET PYTHONPATH=%CD%
pytest --nbval-lax ./examples --cov=libcbm --cov-report html  --cov-append
coverage-badge -o coverage.svg -f
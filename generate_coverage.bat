:: note to include numba routines in coverage, set disable it with the option
SET NUMBA_DISABLE_JIT=1
:: can also be done in the .numba_config.yml file in this dir
pytest --cov=libcbm --cov-report html
:: make the notebooks look at the local dir
SET PYTHONPATH=%CD%
pytest --nbval-lax ./examples --cov=libcbm --cov-report html  --cov-append
:: run the notebooks in the docs dir as well
pytest --nbval-lax ./docs --cov=libcbm --cov-report html  --cov-append
coverage-badge -o coverage.svg -f
:: note to include numba routines in coverage, set disable it with the option
:: DISABLE_JIT: True
:: in the .numba_config.yml file in this dir
pytest --cov=libcbm --cov-report html:cov_html
coverage-badge -o coverage.svg -f
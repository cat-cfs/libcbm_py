#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script to run the current feature that is being developed and test it.

Typically you would run this file from a command line like this:

     ipython3.exe -i -- /deploy/libcbm_py/current.py
"""

# Built-in modules #

# Third party modules #
from tqdm import tqdm

# First party modules #

# Internal modules #
from libcbm_runner.core.continent import continent

################################################################################
scenario = continent.scenarios['historical']
runners  = [rs[-1] for k,rs in scenario.runners.items() if k=='ZZ']
runner   = runners[0]
runner.run()

################################################################################
#config_path = "/home/sinclair/repos/libcbm_py/examples/sit/growth_only/sit_config.json"
#from libcbm.input.sit import sit_cbm_factory
#sit = sit_cbm_factory.load_sit(config_path)
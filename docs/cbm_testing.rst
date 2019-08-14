.. _cbm-testing:


CBM testing
===========================

Since LibCBM is partially intended as a fully functioning rewrite of the
CBM-CFS3_ model, it is written with rigorous testing methods for comparing
CBM-CFS3 results for numerically similar Carbon dynamics and identical
stand-state variables.  LibCBM supports fully automated simulation of test
cases with CBM-CFS3 and side-by-side comparison with LibCBM results.

The testing approach is:

  1. generate randomized test cases
     (see :py:func:`libcbm.test.cbm.case_generation`) which include:

    - initially forested stands, deforestation, and afforestation
    - growth curves including all species, and multiple species stands
    - random stand initial conditions
    - random disturbance events

  2. simulate randomized test cases with both of:

    - CBM-CFS3 (see: :py:mod:`libcbm.test.cbm.cbm3_support.cbm3_simulator`)
    - LibCBM (see: :py:mod:`libcbm.test.cbm.test_case_simulator`)

  3. Compare results for all randomized test cases:

    - compare pool, flux, and state variables for each case and time step combination


Test case generator
-------------------

The functions in :py:mod:`libcbm.test.cbm.case_generation` support the
creation of randomized test cases for simulating with both LibCBM and
CBM-CFS3.

.. automodule:: libcbm.test.cbm.case_generation
    :members:


Test case simulator
-------------------

The functions in :py:mod:`libcbm.test.cbm.test_case_simulator` simulate the
randomized test cases produced by :py:mod:`libcbm.test.cbm.case_generation`
using LibCBM.

.. automodule:: libcbm.test.cbm.test_case_simulator
    :members:


CBM-CFS3 support
----------------

The functions in :py:mod:`libcbm.test.cbm.cbm3_support.cbm3_simulator`
simulate the randomized test cases produced by
:py:mod:`libcbm.test.cbm.case_generation` using CBM-CFS3.

Simulating with CBM-CFS3 requires cbm3_python_ and StandardImportToolPlugin_
for automation of the CBM-CFS3 toolbox to execute test cases.

.. automodule:: libcbm.test.cbm.cbm3_support.cbm3_python_helper
    :members:

.. automodule:: libcbm.test.cbm.cbm3_support.cbm3_simulator
    :members:


Pool comparison
---------------

The functions in :py:mod:`libcbm.test.cbm.cbm3_support.pool_comparison`
support the pool by pool comparison of test case simulation results run by
LibCBM versus CBM-CFS3.

.. automodule:: libcbm.test.cbm.pool_comparison
    :members:



.. _CBM-CFS3: https://www.nrcan.gc.ca/climate-change/impacts-adaptations/climate-change-impacts-forests/carbon-accounting/carbon-budget-model/13107
.. _cbm3_python: https://github.com/cat-cfs/cbm3_python
.. _StandardImportToolPlugin: https://github.com/cat-cfs/StandardImportToolPlugin

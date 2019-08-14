CBM implementation in LibCBM
==================================

LibCBM has an implementation of CBM which is comparable with CBM-CFS3_.  See: :ref:`cbm-testing`.

.. _CBM-CFS3: https://www.nrcan.gc.ca/climate-change/impacts-adaptations/climate-change-impacts-forests/carbon-accounting/carbon-budget-model/13107

The LibCBM implementation of the CBM model is designed with externalized
model state.

This also means that consumers of the CBM class allocate and have full
control and access to all CBM variables between calls to CBM functions.

This goes hand in hand with externalized model stepping which allows user
control of stepping through CBM time.

With the above externalizations CBM provides a powerful API for integrating
other processes or feedbacks to work with or inform the CBM model on a
time step basis.

All CBM variables are allocated using standard Pandas data frames or numpy
ndarrays, and many functions support either.

The CBM class consists of 3 methods for simulating Carbon dynamics

    - spinup: used to initialize Carbon pools
        :py:func:`libcbm.model.cbm.cbm_model.CBM.spinup`
    - init: used to initialize CBM state variables
        :py:func:`libcbm.model.cbm.cbm_model.CBM.spinup`
    - step: advances CBM state and pools through 1 time step and
        summarizes fluxes
        :py:func:`libcbm.model.cbm.cbm_model.CBM.spinup`

An instance of the CBM model can be created with the
:py:func:`libcbm.model.cbm.cbm_factory` function.


.. autoclass:: libcbm.model.cbm.cbm_model.CBM
    :members:

.. automodule:: libcbm.model.cbm.cbm_factory
    :members:
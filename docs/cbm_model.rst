.. _cbm-model:

CBM
===

LibCBM has an implementation of CBM which is comparable with CBM-CFS3_.  See: :ref:`cbm-testing`.

.. _CBM-CFS3: https://www.nrcan.gc.ca/climate-change/impacts-adaptations/climate-change-impacts-forests/carbon-accounting/carbon-budget-model/13107

Usage
-----

A CBM instance can be initialized by using a composed factory method.



.. automodule:: libcbm.model.cbm.cbm_factory
    :members:

The CBM class
-------------
The LibCBM implementation of the CBM model is designed all model state is
user defined and passed to the CBM functions.

This means that consumers of the CBM class allocate and have full
control and access to all CBM variables between calls to CBM functions.

CBM provides a step function which allows caller control of stepping
through CBM time, whereas in CBM-CFS3 stepping is an internal process
that does not offer the ability to interact with other processes on a
time step basis.

With the above features CBM provides a powerful API for integrating or
informing other processes or feedbacks to work with or inform the CBM model
as it simulates Carbon dynamics.

All CBM variables are allocated using standard Pandas data frames or numpy
ndarrays, and many functions support either.

The CBM class consists of 3 methods for simulating Carbon dynamics

    - spinup: used to initialize Carbon pools
        :py:func:`libcbm.model.cbm.cbm_model.CBM.spinup`
    - init: used to initialize CBM state variables
        :py:func:`libcbm.model.cbm.cbm_model.CBM.init`
    - step: advances CBM state and pools through 1 time step and
        summarizes fluxes
        :py:func:`libcbm.model.cbm.cbm_model.CBM.step`

An instance of the CBM model can be created with the
:py:func:`libcbm.model.cbm.cbm_factory` function.


.. autoclass:: libcbm.model.cbm.cbm_model.CBM
    :members:


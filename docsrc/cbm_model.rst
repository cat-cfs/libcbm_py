.. _cbm-model:

CBM
===

LibCBM has an implementation of CBM which is comparable with CBM-CFS3_.

.. _CBM-CFS3: https://www.nrcan.gc.ca/climate-change/impacts-adaptations/climate-change-impacts-forests/carbon-accounting/carbon-budget-model/13107

Usage
-----

A CBM instance can be initialized by using a composed factory method.



.. automodule:: libcbm.model.cbm.cbm_factory
    :members:

The CBM class
-------------

The CBM class is a set of functions that run the CBM model including spinup,
variable initialization and model stepping.  It replicates the Carbon dynamics
and stand state of the CBM-CFS3 model.

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

Simulator
---------

.. automodule:: libcbm.model.cbm.cbm_simulator
    :members:

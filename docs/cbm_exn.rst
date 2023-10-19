:tocdepth: 2

.. _cbm-exn:

cbm-exn
=======

CBM-EXN is a CBM-CFS3 equivalent model, except that it is driven by net C
increments.  Therefore the difference between CBM-EXN, and CBM-CFS3 is that
CBM-EXN runs without the merchantable volume to biomass conversion routines
and instead accepts C increments for the MerchC foliage C and other C pools
directly.

The other difference is that the CBM3 built-in HW, SW structure is removed,
and each simulation record represents an individual cohort with age and
species. This allows the flexibility of user-defined multi-cohort stand
structuring via external grouping.

`cbm_exn` is built using the :ref:`model_definition` interfaces and classes

.. _cbm-exn-custom-ops: cbm_exn_custom_ops.ipynb

See cbm-exn-custom-ops_ for a step by step example for cbm_exn including
overriding the default C ops

Functions and classes
---------------------


.. automodule:: libcbm.model.cbm_exn.cbm_exn_land_state
    :members:

.. currentmodule:: libcbm.model.cbm_exn.cbm_exn_model

.. autofunction:: initialize

.. autoclass:: SpinupReporter
    :members:

.. autoclass:: CBMEXNModel
    :members:

.. autoclass:: libcbm.model.cbm_exn.cbm_exn_parameters.CBMEXNParameters
    :members:

.. automodule:: libcbm.model.cbm_exn.cbm_exn_spinup
    :members:

.. automodule:: libcbm.model.cbm_exn.cbm_exn_step
    :members:

.. automodule:: libcbm.model.cbm_exn.cbm_exn_variables
    :members:

.. _cbm-input-and-variables:

CBM Input and Variables
=======================


.. currentmodule:: libcbm.model.cbm.cbm_variables

Inventory
---------

.. autofunction:: initialize_inventory

Pools
-----

.. autofunction:: initialize_pools

Flux
----

.. autofunction:: initialize_flux


Spinup parameters
-----------------

.. autofunction:: initialize_spinup_parameters

Spinup variables
----------------

.. autofunction:: initialize_spinup_variables

CBM time step parameters
------------------------

.. autofunction:: initialize_cbm_parameters

CBM state variables
-------------------

.. autofunction:: initialize_cbm_state_variables


Classifiers
-----------

.. currentmodule:: libcbm.model.cbm.cbm_config

.. autofunction:: classifier

.. autofunction:: classifier_value

.. autofunction:: classifier_config


Merchantable Volume Curves
--------------------------

.. autofunction:: merch_volume_curve

.. autofunction:: merch_volume_to_biomass_config


CBM default parameters
----------------------

The parameters in this section are the simulation-constant model parameters.
These are used to initialize the CBM class.

.. automodule:: libcbm.model.cbm.cbm_defaults
    :members:

.. automodule:: libcbm.model.cbm.cbm_defaults_queries
    :members:

CBM default parameters reference
--------------------------------

.. autoclass:: libcbm.model.cbm.cbm_defaults_reference.CBMDefaultsReference
    :members:
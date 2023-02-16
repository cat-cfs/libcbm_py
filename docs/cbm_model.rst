:tocdepth: 2

.. _cbm-model:


CBM-CFS3 support in libcbm
==========================

LibCBM has an implementation of CBM which is comparable with CBM-CFS3_.

.. _CBM-CFS3: https://www.nrcan.gc.ca/climate-change/impacts-adaptations/climate-change-impacts-forests/carbon-accounting/carbon-budget-model/13107
.. _cbm3-tutorial2: cbm3_tutorial2.ipynb
.. _multi-stand-modelling: multi_stand_modelling.ipynb

libcbm supports 2 primary methods for running the CBM-CFS3 implementation.
These are via the CBM standard import tool format (SIT) and through a
more basic multi-stand modelling framework.

Standard import tool (CBM-SIT) format
-------------------------------------

See cbm3-tutorial2_ for a step by step example for running libcbm via
SIT format in a jupyter notebook.

Code documentation for the SIT implementation :ref:`cbm-sit`


Multi-Stand level modelling support
-----------------------------------

Libcbm can also run a simplified version of CBM, without the rule-based
disturbances and transitions built into the SIT.  This method may be more
suitable for larger scale simulations with direct, many-to-one linkages for
disturbance events to stands across timesteps.

See multi-stand-modelling_ for a step by step example of running libcbm this way.

Internal CBM Classes and functions
----------------------------------

A CBM instance can be initialized by using a composed factory method.

.. automodule:: libcbm.model.cbm.cbm_factory
    :members:

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

CBM Input and Variables
-----------------------

.. currentmodule:: libcbm.model.cbm.cbm_variables

CBM Variables class (cbm_vars)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CBMVariables
    :members:

CBM Spinup variables Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: initialize_spinup_variables

.. autofunction:: initialize_spinup_parameters


CBM Step variables Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: initialize_simulation_variables

Output processing
-----------------

.. autoclass:: libcbm.model.cbm.cbm_output.CBMOutput
    :members:

Configuration Details
---------------------


Classifiers
^^^^^^^^^^^

.. currentmodule:: libcbm.model.cbm.cbm_config

.. autofunction:: classifier

.. autofunction:: classifier_value

.. autofunction:: classifier_config


Merchantable Volume Curves
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: merch_volume_curve

.. autofunction:: merch_volume_to_biomass_config


CBM default parameters
^^^^^^^^^^^^^^^^^^^^^^

The parameters in this section are the simulation-constant model parameters.
These are used to initialize the CBM class.

.. automodule:: libcbm.model.cbm.cbm_defaults
    :members:


CBM default parameters reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: libcbm.model.cbm.cbm_defaults_reference.CBMDefaultsReference
    :members:


CBM rule based disturbances and transition rules
------------------------------------------------

.. autoclass:: libcbm.model.cbm.rule_based.classifier_filter.ClassifierFilter
    :members:

.. automodule:: libcbm.model.cbm.rule_based.event_processor
    :members:

.. automodule:: libcbm.model.cbm.rule_based.rule_filter
    :members:

.. automodule:: libcbm.model.cbm.rule_based.rule_target
    :members:

.. automodule:: libcbm.model.cbm.rule_based.transition_rule_processor
    :members:

.. autoclass:: libcbm.model.cbm.rule_based.transition_rule_processor.TransitionRuleProcessor
    :members:

SIT Specific rule-based disturbance and transition rules functionality
----------------------------------------------------------------------

.. autoclass:: libcbm.model.cbm.rule_based.sit.sit_event_processor.SITEventProcessor
    :members:

.. automodule:: libcbm.model.cbm.rule_based.sit.sit_stand_filter
    :members:

.. automodule:: libcbm.model.cbm.rule_based.sit.sit_stand_target
    :members:

.. automodule:: libcbm.model.cbm.rule_based.sit.sit_transition_rule_processor
    :members:

.. autoclass:: libcbm.model.cbm.rule_based.sit.sit_transition_rule_processor.SITTransitionRuleProcessor
    :members:
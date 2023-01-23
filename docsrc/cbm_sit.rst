.. _cbm-sit:

CBM Standard Import tool format
===============================

See Chapter 3 of the Operational-Scale CBM-CFS3-Manual_ for a detailed
description of this format.

Note the SIT format does not define specifically names to column headers.
It instead interprets column meaning based on column ordering, so any column
labels in DataFrames passed to SIT parse functions function will be ignored
by the parsing functions here.

.. _CBM-CFS3-Manual: http://www.cfs.nrcan.gc.ca/pubwarehouse/pdfs/35717.pdf

Classifiers
-----------

.. automodule:: libcbm.input.sit.sit_classifier_parser
    :members:

Age Classes
-----------

.. automodule:: libcbm.input.sit.sit_age_class_parser
    :members:

Disturbance Types
-----------------

.. automodule:: libcbm.input.sit.sit_disturbance_type_parser
    :members:

Inventory
---------

.. automodule:: libcbm.input.sit.sit_inventory_parser
    :members:

Growth and Yield
----------------

.. automodule:: libcbm.input.sit.sit_yield_parser
    :members:

Disturbance Events
------------------

.. automodule:: libcbm.input.sit.sit_disturbance_event_parser
    :members:

Transition Rules
------------------

.. automodule:: libcbm.input.sit.sit_transition_rule_parser
    :members:

The SIT Format
--------------

.. automodule:: libcbm.input.sit.sit_parser
    :members:

.. automodule:: libcbm.input.sit.sit_format
    :members:

Simulating SIT input
--------------------

.. automodule:: libcbm.input.sit.sit_cbm_factory
    :members:

Reading SIT input
-----------------

.. automodule:: libcbm.input.sit.sit_reader
    :members:

SIT Mapping
-----------

.. autoclass:: libcbm.input.sit.sit_mapping.SITMapping
    :members:

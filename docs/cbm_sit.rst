.. _cbm-sit:

CBM Standard Import tool format
===============================

See Chapter 3 of the Operational-Scale CBM-CFS3-Manual_ for a detailed
description of this format.

Note the SIT format is a no column header, ordered column format, so any column
labels in DataFrames passed to SIT parse functions function will be ignored
by this function.

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

.. automodule:: libcbm.input.sit.sit_transition_rules_parser
    :members:
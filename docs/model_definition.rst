:tocdepth: 2

.. _model_definition:

Model definition
================

The model_definition package of libcbm can be used to define a CBM-like model using python code.


.. currentmodule:: libcbm.model.model_definition.model

.. autofunction:: initialize

.. autoclass:: CBMModel
    :members:


.. currentmodule:: libcbm.model.model_definition.model_handle

.. autofunction::create_model_handle

.. autoclass:: ModelHandle
    :members:

.. autoclass:: libcbm.model.model_definition.model_variables.ModelVariables
    :members:

.. autoclass:: libcbm.model.model_definition.output_processor.ModelOutputProcessor
    :members:

.. currentmodule:: libcbm.model.model_definition.spinup_engine

.. autoclass:: SpinupState
    :members:

.. autofunction:: advance_spinup_state

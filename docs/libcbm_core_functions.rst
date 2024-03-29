:tocdepth: 2

.. _libcbm-core-functions:

libcbm core functionality
=========================

Dataframe abstraction
---------------------

libcbm has an abstraction layer for dataframes for isolating interactions with
external libraries such as numpy, pandas and dask.  Internally libcbm
functionality calls this abstraction layer rather than the 3rd party libraries directly.

Dataframe
^^^^^^^^^

.. automodule:: libcbm.storage.dataframe
    :members:

Series
^^^^^^

.. automodule:: libcbm.storage.series
    :members:

Backends
^^^^^^^^

.. automodule:: libcbm.storage.backends
    :members:

C++ library wrapper functions
-----------------------------

The libcbm core functions are pool and flux functions that are generally useful for CBM-like models.

.. autoclass:: libcbm.wrapper.libcbm_ctypes.LibCBM_ctypes
    :members:

.. autoclass:: libcbm.wrapper.libcbm_error.LibCBM_Error
    :members:

.. autoclass:: libcbm.wrapper.libcbm_handle.LibCBMHandle
    :members:

.. autoclass:: libcbm.wrapper.libcbm_matrix.LibCBM_Matrix_Base
    :members:

.. autoclass:: libcbm.wrapper.libcbm_matrix.LibCBM_Matrix
    :members:

.. autoclass:: libcbm.wrapper.libcbm_matrix.LibCBM_Matrix_Int
    :members:

.. autoclass:: libcbm.wrapper.libcbm_wrapper.LibCBMWrapper
    :members:

CBM3-Specific C++ library wrapper functions
-------------------------------------------


The CBM core functionality is the set of low level functions that compose the CBM model.

.. automodule:: libcbm.wrapper.cbm.cbm_ctypes
    :members:

.. autoclass:: libcbm.wrapper.cbm.cbm_wrapper.CBMWrapper
    :members:

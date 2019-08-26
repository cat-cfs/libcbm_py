LibCBM
======

LibCBM is a next generation version of the CBM-CFS3 model.  It also has useful
functions for extending CBM, or developing new CBM-like models.

Key Features
------------

 - User allocated model state using numpy/Pandas and user controlled model stepping See: :ref:`cbm-model`
 - Core CBM dynamics and state functions. See: :ref:`cbm-core-functions`
 - CBM-CFS3 results comparison/test functions. See: :ref:`cbm-testing`
 - General purpose matrix-based pool/flux compute methods. See: :ref:`libcbm-core-functions`
 - Example jupyter notebooks including CBM-CFS3 testing (see /examples)

Requirements
------------
 - python3x
 - The libcbm compiled library, or libcbm c/c++ source
 - numpy
 - pandas
 - sqlite3

In order to run the CBM-CFS3 tests there are additional requirements:
 - Windows OS (CBM-CFS3 is windows only)
 - pyodbc and a working 64 bit MS Access driver
 - gitpython

Running the notebooks in /examples Also have additional requirements:
 - scipy
 - jupyter
 - jupytext


Code Documentation
------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   cbm_model
   cbm_variables
   cbm_parameters
   cbm_testing
   cbm_core_functions
   libcbm_core_functions
   other_functions

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

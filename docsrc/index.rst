libcbm
======

version |version|

libcbm is a next generation version of the CBM-CFS3 model.  It also has useful
functions for extending CBM, or developing new CBM-like models.

github_

.. _github: http://www.github.com/cat-cfs/libcbm_py
.. _moss-c-publication: https://doi.org/10.1139/cjfr-2015-0512

Key Features
------------

 - User allocated model state using numpy/Pandas and user controlled model stepping See: :ref:`cbm-model`
 - Core CBM dynamics and state functions. See: :ref:`cbm-core-functions`
 - General purpose matrix-based pool/flux compute methods. See: :ref:`libcbm-core-functions`
 - Example jupyter notebooks (see /examples)
 - Full support for the CBM-CFS3 standard import tool format
 - Implementation of the moss C model See: moss-c-publication_
 - Support for running CBM-CFS3 dynamics using net above ground biomass C Increment: :ref:`cbm-exn`

Requirements
------------
 - python3x
 - The libcbm compiled library, or libcbm c/c++ source
 - numpy
 - pandas
 - sqlite3
 - numexpr

Running the notebooks in /examples Also have additional requirements:
 - scipy
 - jupyter
 - jupytext


Code Documentation
------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   cbm_model
   cbm_input_and_variables
   cbm_sit
   rule_based
   cbm3_tutorial2
   cbm_core_functions
   model_definition
   moss_c_model
   cbm_exn

   libcbm_core_functions
   package_resources

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

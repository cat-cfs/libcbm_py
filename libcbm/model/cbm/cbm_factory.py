# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import json
from libcbm.model.cbm.cbm_model import CBM
from libcbm.wrapper.cbm.cbm_wrapper import CBMWrapper
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle


def create(dll_path, dll_config_factory, cbm_parameters_factory,
           merch_volume_to_biomass_factory, classifiers_factory):
    """Create and initialize an instance of the CBM model

    Args:
        dll_path (str): path to the libcbm compiled library
        dll_config_factory (func): a function that returns dll configuration.
        cbm_parameters_factory (func): a function that returns CBM
            parameters.
        merch_volume_to_biomass_factory (func): function that creates a
            valid merchantable volume to biomass configuration for CBM (see:
            :py:func:`libcbm.model.cbm.cbm_config.merch_volume_to_biomass_config`)
        classifiers_factory (func): function that creates a valid classifier
            configuration for CBM (see:
            :py:func:`libcbm.model.cbm.cbm_config.classifier_config`)

    In the following example a CBM instance is built with a single growth
    curve, and classifier set.  The :py:mod:`libcbm.model.cbm.cbm_defaults`
    module is use to construct the other factory methods

    Example::

        from libcbm.model.cbm import cbm_defaults
        from libcbm.model.cbm import cbm_factory
        from libcbm.model.cbm import cbm_config

        db_path = "cbm_defaults.db"
        dll_path = "LibCBM.dll"

        classifiers = lambda : cbm_config.classifier_config([
            cbm_config.classifier(
                "c1",
                values=[cbm_config.classifier_value("c1_v1")])
        ])

        merch_volumes = lambda : cbm_config.merch_volume_to_biomass_config(
            db_path=db_path,
            merch_volume_curves=[
                cbm_config.merch_volume_curve(
                classifier_set=["c1_v1"],
                merch_volumes=[{
                    "species_id": 1,
                    "age_volume_pairs": [[0,0],[50,100],[100,150],[150,200]]
                }]
            )]
        )

        cbm = cbm_factory.create(
            dll_path=dll_path,
            dll_config_factory=cbm_defaults.get_libcbm_configuration_factory(
                db_path),
            cbm_parameters_factory=cbm_defaults.get_cbm_parameters_factory(
                db_path),
            merch_volume_to_biomass_factory=merch_volumes,
            classifiers_factory=classifiers)

    Returns:
        :py:class:`libcbm.model.cbm.CBM`: an initialized instance of the CBM
            model
    """

    configuration_string = json.dumps(dll_config_factory())
    libcbm_handle = LibCBMHandle(dll_path, configuration_string)
    libcbm_wrapper = LibCBMWrapper(libcbm_handle)

    merch_volume_to_biomass_config = merch_volume_to_biomass_factory()
    classifiers_config = classifiers_factory()
    cbm_config = {
        "cbm_defaults": cbm_parameters_factory(),
        "merch_volume_to_biomass": merch_volume_to_biomass_config,
        "classifiers": classifiers_config["classifiers"],
        "classifier_values": classifiers_config["classifier_values"],
    }

    cbm_config_string = json.dumps(cbm_config)
    cbm_wrapper = CBMWrapper(libcbm_handle, cbm_config_string)
    return CBM(libcbm_wrapper, cbm_wrapper)

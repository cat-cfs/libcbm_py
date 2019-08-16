
import ctypes
import json
import pandas as pd
from libcbm import data_helpers
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int
from libcbm.wrapper.libcbm_ctypes import LibCBM_ctypes


class CBMWrapper(LibCBM_ctypes):
    """Exposes low level ctypes wrapper to regular python, for CBM
    specific libcbm functions.

    The base class is :py:class:`libcbm.wrapper.libcbm_ctypes.LibCBM_ctypes`

    Args:
        handle (libcbm.wrapper.libcbm_handle.LibCBMHandle): handle to the low
            level function library
        config (str): A json formatted string containing CBM
            configuration.

            See :py:mod:`libcbm.model.cbm.cbm_defaults` for
            construction of the "cbm_defaults" value, and
            :py:mod:`libcbm.model.cbm.cbm_config` for helper methods.

            Example::

                {
                    "cbm_defaults": {"p1": {}, "p2": {}, ..., "pN": {}},
                    "classifiers": [
                        {"id": 1, "name": "a"},
                        {"id": 2, "name": "b"},
                        {"id": 3, "name": "c"}
                    ],
                    "classifier_values": [
                        {
                            "id": 1,
                            "classifier_id": 1,
                            "value": "a1",
                            "description": "a1"
                        },
                        {
                            "id": 2,
                            "classifier_id": 2,
                            "value": "b2",
                            "description": "b2"
                        },
                        {
                            "id": 3,
                            "classifier_id": 3,
                            "value": "c1",
                            "description": "c1"
                        }
                    ],
                    "merch_volume_to_biomass": {
                        'db_path': './cbm_defaults.db',
                        'merch_volume_curves': [
                            {
                                'classifier_set': {
                                    'type': 'name',
                                    'values': ['a1','b2','c1']
                                },
                                'components': [
                                    {
                                    'species_id': 1,
                                    'age_volume_pairs': [(age0, vol0),
                                                         (age1, vol0),
                                                         (ageN, volN)]
                                    },
                                    {
                                    'species_id': 2,
                                    'age_volume_pairs': [(age0, vol0),
                                                         (age1, vol0),
                                                         (ageN, volN)]
                                    }
                                ]
                            }
                        ]
                    },
                    "transitions": [
                        {
                            "id": 1,
                            "classifier_set": {
                                'type': 'name', 'values': ['a1','b2','?']
                            },
                            "regeneration_delay": 0,
                            "reset_age": 0
                        }
                    ]
                }
    """
    def __init__(self, handle, config):
        self.handle = handle
        p_config = ctypes.c_char_p(config.encode("UTF-8"))
        self.handle.call("LibCBM_Initialize_CBM", p_config)

    def AdvanceStandState(self, inventory, state_variables, parameters):
        """Advances CBM stand variables through a timestep based on the
        current simulation state.

        Args:
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Will not be modified by this function.
                See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_inventory`
                for a compatible definition
            state_variables (pandas.DataFrame): simulation variables which
                define all non-pool state in the CBM model.  Altered by this
                function call.  See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_cbm_state_variables`
                for a compatible definition
            parameters (object): Read-only parameters used in a CBM timestep.
                See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_cbm_parameters`
                for a compatible definition.
        """
        i = data_helpers.unpack_ndarrays(inventory)
        v = data_helpers.unpack_ndarrays(state_variables)
        p = data_helpers.unpack_ndarrays(parameters)

        n = i.classifiers.shape[0]
        classifiersMat = LibCBM_Matrix_Int(i.classifiers)

        self.handle.call(
            "LibCBM_AdvanceStandState", n, classifiersMat,
            i.spatial_unit, p.disturbance_type, p.transition_rule_id,
            v.last_disturbance_type, v.time_since_last_disturbance,
            v.time_since_land_class_change, v.growth_enabled, v.enabled,
            v.land_class, v.regeneration_delay, v.age)

    def EndStep(self, state_variables):
        """Applies end-of-timestep changes to the CBM state

        Args:
            state_variables (pandas.DataFrame): simulation variables which
                define all non-pool state in the CBM model.  This
                function call will alter this variable with end-of-step
                changes. See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_cbm_state_variables`
                for a compatible definition
        """
        v = data_helpers.unpack_ndarrays(state_variables)
        n = v.age.shape[0]
        self.handle.call(
            "LibCBM_EndStep", n, v.age, v.regeneration_delay, v.enabled)

    def InitializeLandState(self, inventory, pools, state_variables):
        """Initializes CBM state to values appropriate for after running
        spinup and before starting CBM stepping

        Args:
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Will not be modified by this function.
                See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_inventory`
                for a compatible definition.
            pools (numpy.ndarray or pandas.DataFrame): matrix of shape
                n_stands by n_pools. The values in this matrix are updated by
                this function for stands that have an afforestation pre-type
                defined.
            state_variables (pandas.DataFrame): simulation variables which
                define all non-pool state in the CBM model.  This
                function call will alter this variable with CBM initial state
                values. See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_cbm_state_variables`
                for a compatible definition.

        """
        i = data_helpers.unpack_ndarrays(inventory)
        v = data_helpers.unpack_ndarrays(state_variables)
        n = i.last_pass_disturbance_type.shape[0]
        poolMat = LibCBM_Matrix(data_helpers.get_ndarray(pools))

        self.handle.call(
            "LibCBM_InitializeLandState", n, i.last_pass_disturbance_type,
            i.delay, i.age, i.spatial_unit, i.afforestation_pre_type_id,
            poolMat, v.last_disturbance_type, v.time_since_last_disturbance,
            v.time_since_land_class_change, v.growth_enabled, v.enabled,
            v.land_class, v.age)

    def AdvanceSpinupState(self, inventory, variables, parameters):
        """Advances spinup state variables through one spinup step.

        Args:
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Will not be modified by this function.
                See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_inventory`
                for a compatible definition
            variables (object): Spinup working variables.  Defines all
                non-pool simulation state during spinup.  See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_spinup_variables`
                for a compatible definition
            parameters (object): spinup parameters. See:
                :py:func:`libcbm.model.cbm.cbm_variables.initialize_spinup_parameters`
                for a compatible definition

        Returns:
            int: The number of stands finished running the spinup routine
            as of the end of this call.
        """

        i = data_helpers.unpack_ndarrays(inventory)
        p = data_helpers.unpack_ndarrays(parameters)
        v = data_helpers.unpack_ndarrays(variables)
        n = i.spatial_unit.shape[0]

        # If return_interval, min_rotations, max_rotations are explicitly
        # set by the user, ignore the spatial unit, which is used to set
        # default value for these 3 variables.
        return_interval = data_helpers.get_nullable_ndarray(
            p.return_interval, type=ctypes.c_int)
        min_rotations = data_helpers.get_nullable_ndarray(
            p.min_rotations, type=ctypes.c_int)
        max_rotations = data_helpers.get_nullable_ndarray(
            p.max_rotations, type=ctypes.c_int)
        spatial_unit = None
        if return_interval is None or min_rotations is None \
           or max_rotations is None:
            spatial_unit = data_helpers.get_nullable_ndarray(
                i.spatial_unit, type=ctypes.c_int)

        n_finished = self.handle.call(
            "LibCBM_AdvanceSpinupState", n, spatial_unit, return_interval,
            min_rotations, max_rotations, i.age, i.delay, v.slow_pools,
            i.historical_disturbance_type, i.last_pass_disturbance_type,
            i.afforestation_pre_type_id, v.spinup_state, v.disturbance_type,
            v.rotation, v.step, v.last_rotation_slow_C, v.enabled)

        return n_finished

    def EndSpinupStep(self, pools, variables):
        """Applies end-of-timestep changes to the spinup state

        Args:
            pools (numpy.ndarray or pandas.DataFrame): matrix of shape
                n_stands by n_pools. The values in this matrix are used to
                compute a criteria for exiting the spinup routing.  The
                biomass pools are also zeroed for historical and last pass
                disturbances.
            variables (object): Spinup working variables.  Defines all
                non-pool simulation state during spinup.  Set to an
                end-of-timestep state by this function. See:
                :py:func:`libcbm.model.cbm_variables.initialize_spinup_variables`
                for a compatible definition

        """
        v = data_helpers.unpack_ndarrays(variables)
        n = v.age.shape[0]
        poolMat = LibCBM_Matrix(data_helpers.get_ndarray(pools))
        self.handle.call(
            "LibCBM_EndSpinupStep", n, v.spinup_state, v.disturbance_type,
            poolMat, v.age, v.slow_pools, v.growth_enabled)

    def GetMerchVolumeGrowthOps(self, growth_op, inventory, pools,
                                state_variables):
        """Computes CBM merchantable growth as a bulk matrix operation.

        Args:
            growth_op (int): Handle for a block of matrices as allocated by
                the :py:func:`AllocateOp` function. Used to compute merch
                volume growth operations.
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Used by this function to find correct
                parameters from the set of merch volume growth parameters
                passed to library initialization, and to find a yield curve
                associated with inventory classifier sets. Will not be
                modified by this function. See:
                :py:func:`libcbm.model.cbm_variables.initialize_inventory`
                for a compatible definition.
            pools (numpy.ndarray or pandas.DataFrame): matrix of shape
                n_stands by n_pools. Used by this function to compute a root
                increment, and also to limit negative growth increments such
                that a negative biomass pools are prevented.  This parameter
                is not modified by this function.
            state_variables (pandas.DataFrame): simulation variables which
                define all non-pool state in the CBM model.  This function
                call will not alter this parameter. See:
                :py:func:`libcbm.model.cbm_variables.initialize_cbm_state_variables`
                for a compatible definition
        """
        n = pools.shape[0]
        poolMat = LibCBM_Matrix(data_helpers.get_ndarray(pools))

        opIds = (ctypes.c_size_t * (1))(*[growth_op])
        i = data_helpers.unpack_ndarrays(inventory)
        classifiersMat = LibCBM_Matrix_Int(
            data_helpers.get_ndarray(i.classifiers))
        v = data_helpers.unpack_ndarrays(state_variables)

        last_disturbance_type = data_helpers.get_nullable_ndarray(
            v.last_disturbance_type, type=ctypes.c_int)
        time_since_last_disturbance = data_helpers.get_nullable_ndarray(
            v.time_since_last_disturbance, type=ctypes.c_int)
        growth_multiplier = data_helpers.get_nullable_ndarray(
            v.growth_multiplier, type=ctypes.c_double)
        growth_enabled = data_helpers.get_nullable_ndarray(
            v.growth_enabled, type=ctypes.c_int)

        self.handle.call(
            "LibCBM_GetMerchVolumeGrowthOps", opIds, n, classifiersMat,
            poolMat, v.age, i.spatial_unit, last_disturbance_type,
            time_since_last_disturbance, growth_multiplier, growth_enabled)

    def GetTurnoverOps(self, biomass_turnover_op, snag_turnover_op,
                       inventory):
        """Computes biomass turnovers and dead organic matter turnovers as
        bulk matrix operations.

        Args:
            biomass_turnover_op (int): Handle for a block of matrices as
                allocated by the :py:func:`AllocateOp` function. Used to
                compute biomass turnover operations.
            snag_turnover_op (int): Handle for a block of matrices as
                allocated by the :py:func:`AllocateOp` function. Used to
                compute dom (specifically snags) turnover operations.
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Used by this function to find correct
                parameters from the set of turnover parameters passed to
                library initialization. Will not be modified by this
                function. See:
                :py:func:`libcbm.model.cbm_variables.initialize_inventory`
                for a compatible definition.
        """
        i = data_helpers.unpack_ndarrays(inventory)
        n = i.spatial_unit.shape[0]
        opIds = (ctypes.c_size_t * (2))(
            *[biomass_turnover_op, snag_turnover_op])

        self.handle.call(
            "LibCBM_GetTurnoverOps", opIds, n, i.spatial_unit)

    def GetDecayOps(self, dom_decay_op, slow_decay_op, slow_mixing_op,
                    inventory, parameters, historical_mean_annual_temp=False):
        """Prepares dead organic matter decay bulk matrix operations.

        Args:
            dom_decay_op (int): Handle for a block of matrices as
                allocated by the :py:func:`AllocateOp` function. Used to
                compute dom decay operations.
            slow_decay_op (int): Handle for a block of matrices as
                allocated by the :py:func:`AllocateOp` function. Used to
                compute slow pool decay operations.
            slow_mixing_op (int): Handle for a block of matrices as
                allocated by the :py:func:`AllocateOp` function. Used to
                compute slow pool mixing operations.
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Used by this function to find correct
                parameters from the set of decay parameters passed to library
                initialization. Will not be modified by this
                function. See:
                :py:func:`libcbm.model.cbm_variables.initialize_inventory`
                for a compatible definition
            parameters (object): [description]
            historical_mean_annual_temp (bool, optional): If set to true, the
                historical default mean annual temperature is used. This is
                intended for spinup.  If explicit mean annual temperature
                is provided via the parameters argument, this parameter will
                be ignored, and the explicit mean annual temp will be used.
                Defaults to False.
        """
        i = data_helpers.unpack_ndarrays(inventory)
        p = data_helpers.unpack_ndarrays(parameters)
        n = i.spatial_unit.shape[0]
        opIds = (ctypes.c_size_t * (3))(
            *[dom_decay_op, slow_decay_op, slow_mixing_op])
        spatial_unit = data_helpers.get_nullable_ndarray(
            i.spatial_unit, ctypes.c_int)
        mean_annual_temp = data_helpers.get_nullable_ndarray(
            p.mean_annual_temp)
        self.handle.call(
            "LibCBM_GetDecayOps", opIds, n, spatial_unit,
            historical_mean_annual_temp, mean_annual_temp)

    def GetDisturbanceOps(self, disturbance_op, inventory,
                          parameters):
        """Sets up CBM disturbance matrices as a bulk matrix operations.

        Args:
            disturbance_op (int): Handle for a block of matrices as
                allocated by the :py:func:`AllocateOp` function. Used to
                compute disturbance event pool flows.
            inventory (object): Data comprised of classifier sets
                and cbm inventory data. Used by this function to find correct
                parameters from the set of disturbance parameters passed to
                library initialization. Will not be modified by this function.
                See: :py:func:`libcbm.model.cbm_variables.initialize_inventory`
                for a compatible definition
            parameters (object): Read-only parameters used to set
                disturbance type id to fetch the appropriate disturbance
                matrix. See:
                :py:func:`libcbm.model.cbm_variables.initialize_cbm_parameters`
                for a compatible definition.

        """
        spatial_unit = data_helpers.unpack_ndarrays(inventory).spatial_unit
        disturbance_type = data_helpers.unpack_ndarrays(
            parameters).disturbance_type
        n = spatial_unit.shape[0]
        opIds = (ctypes.c_size_t * (1))(*[disturbance_op])

        self.handle.call(
            "LibCBM_GetDisturbanceOps", opIds, n, spatial_unit,
            disturbance_type)

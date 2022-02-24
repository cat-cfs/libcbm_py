# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int
from libcbm.wrapper.libcbm_ctypes import LibCBM_ctypes
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm.storage.dataframe import DataFrame


class CBMWrapper(LibCBM_ctypes):
    """Exposes low level ctypes wrapper to regular python, for CBM
    specific libcbm functions.

    The base class is :py:class:`libcbm.wrapper.libcbm_ctypes.LibCBM_ctypes`

    Args:
        handle (LibCBMHandle): handle to the low
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
                    }
                }
    """

    def __init__(self, handle: LibCBMHandle, config: str):
        self.handle = handle
        p_config = ctypes.c_char_p(config.encode("UTF-8"))
        self.handle.call("LibCBM_Initialize_CBM", p_config)

    def advance_stand_state(
        self,
        classifiers: DataFrame,
        inventory: DataFrame,
        state_variables: DataFrame,
        parameters: DataFrame,
    ):
        """Advances CBM stand variables through a timestep based on the
        current simulation state.

        Args:
            classifiers (DataFrame): classifier values associated with
                the inventory
            inventory (DataFrame): CBM inventory data. Will not be modified by
                this function.
            state_variables (DataFrame): simulation variables which
                define all non-pool state in the CBM model.  Altered by this
                function call.
            parameters (DataFrame): Read-only parameters used in a CBM
                timestep.
        """

        self.handle.call(
            "LibCBM_AdvanceStandState",
            inventory.n_rows,
            LibCBM_Matrix_Int(classifiers.to_c_contiguous_numpy_array()),
            inventory["spatial_unit"].to_numpy(),
            parameters["disturbance_type"].to_numpy(),
            parameters["reset_age"].to_numpy(),
            state_variables["last_disturbance_type"].to_numpy(),
            state_variables["time_since_last_disturbance"].to_numpy(),
            state_variables["time_since_land_class_change"].to_numpy(),
            state_variables["growth_enabled"].to_numpy(),
            state_variables["enabled"].to_numpy(),
            state_variables["land_class"].to_numpy(),
            state_variables["regeneration_delay"].to_numpy(),
            state_variables["age"].to_numpy(),
        )

    def end_step(self, state_variables: DataFrame):
        """Applies end-of-timestep changes to the CBM state

        Args:
            state_variables (DataFrame): simulation variables which
                define all non-pool state in the CBM model.  This
                function call will alter this variable with end-of-step
                changes.
        """

        self.handle.call(
            "LibCBM_EndStep",
            state_variables.n_rows,
            state_variables["enabled"],
            state_variables["growth_enabled"],
            state_variables["age"],
            state_variables["regeneration_delay"],
            state_variables["time_since_last_disturbance"],
            state_variables["time_since_land_class_change"],
        )

    def initialize_land_state(
        self,
        inventory: DataFrame,
        pools: DataFrame,
        state_variables: DataFrame,
    ):
        """Initializes CBM state to values appropriate for after running
        spinup and before starting CBM stepping

        Args:
            inventory (DataFrame): CBM inventory data. Will not be modified by
                this function.
            pools (DataFrame): matrix of shape
                n_stands by n_pools. The values in this matrix are updated by
                this function for stands that have an afforestation pre-type
                defined.
            state_variables (DataFrame): simulation variables which
                define all non-pool state in the CBM model.  This
                function call will alter this variable with CBM initial state
                values.

        """
        self.handle.call(
            "LibCBM_InitializeLandState",
            inventory.n_rows,
            inventory["last_pass_disturbance_type"].to_numpy(),
            inventory["delay"].to_numpy(),
            inventory["age"].to_numpy(),
            inventory["spatial_unit"].to_numpy(),
            inventory["afforestation_pre_type_id"].to_numpy(),
            LibCBM_Matrix(pools.to_c_contiguous_numpy_array()),
            state_variables["last_disturbance_type"].to_numpy(),
            state_variables["time_since_last_disturbance"].to_numpy(),
            state_variables["time_since_land_class_change"].to_numpy(),
            state_variables["growth_enabled"].to_numpy(),
            state_variables["enabled"].to_numpy(),
            state_variables["land_class"].to_numpy(),
            state_variables["age"].to_numpy(),
        )

    def advance_spinup_state(
        self, inventory: DataFrame, variables: DataFrame, parameters: DataFrame
    ) -> int:
        """Advances spinup state variables through one spinup step.

        Args:
            inventory (DataFrame): CBM inventory data. Will not be modified by
                this function.
            variables (DataFrame): Spinup working variables.  Defines all
                non-pool simulation state during spinup.
            parameters (DataFrame): spinup parameters.

        Returns:
            int: The number of stands finished running the spinup routine
            as of the end of this call.
        """

        # If return_interval, min_rotations, max_rotations are explicitly
        # set by the user, ignore the spatial unit, which is used to set
        # default value for these 3 variables.
        return_interval = parameters["return_interval"].to_numpy_ptr()
        min_rotations = parameters["min_rotations"].to_numpy_ptr()
        max_rotations = parameters["max_rotations"].to_numpy_ptr()

        include_spatial_unit = (
            return_interval is None
            or min_rotations is None
            or max_rotations is None
        )
        spatial_unit = (
            inventory["spatial_unit"].to_numpy_ptr()
            if include_spatial_unit
            else None
        )

        n_finished = self.handle.call(
            "LibCBM_AdvanceSpinupState",
            inventory.n_rows,
            spatial_unit,
            return_interval,
            min_rotations,
            max_rotations,
            inventory["age"].to_numpy(),
            inventory["delay"].to_numpy(),
            variables["slow_pools"].to_numpy(),
            inventory["historical_disturbance_type"].to_numpy(),
            inventory["last_pass_disturbance_type"].to_numpy(),
            inventory["afforestation_pre_type_id"].to_numpy(),
            variables["spinup_state"].to_numpy(),
            variables["disturbance_type"].to_numpy(),
            variables["rotation"].to_numpy(),
            variables["step"].to_numpy(),
            variables["last_rotation_slow_C"].to_numpy(),
            variables["growth_enabled"].to_numpy(),
            variables["enabled"].to_numpy(),
        )

        return n_finished

    def end_spinup_step(self, pools: DataFrame, variables: DataFrame):
        """Applies end-of-timestep changes to the spinup state

        Args:
            pools (DataFrame): matrix of shape
                n_stands by n_pools. The values in this matrix are used to
                compute a criteria for exiting the spinup routine.  The
                biomass pools are also zeroed for historical and last pass
                disturbances.
            variables (DataFrame): Spinup working variables.  Defines all
                non-pool simulation state during spinup.  Set to an
                end-of-timestep state by this function.

        """
        self.handle.call(
            "LibCBM_EndSpinupStep",
            variables.n_rows,
            variables["spinup_state"].to_numpy(),
            variables["disturbance_type"].to_numpy(),
            LibCBM_Matrix(pools.to_c_contiguous_numpy_array()),
            variables["age"].to_numpy(),
            variables["slow_pools"].to_numpy(),
            variables["growth_enabled"].to_numpy(),
        )

    def get_merch_volume_growth_ops(
        self,
        growth_op: int,
        overmature_decline_op: int,
        classifiers: DataFrame,
        inventory: DataFrame,
        pools: DataFrame,
        state_variables: DataFrame,
    ):
        """Computes CBM merchantable growth as a bulk matrix operation.

        Args:
            growth_op (int): Handle for a block of matrices as allocated by
                the :py:func:`AllocateOp` function. Used to compute merch
                volume growth operations.
            overmature_decline_op (int): Handle for a block of matrices as
                allocated by the :py:func:`AllocateOp` function. Used to
                compute merch volume growth operations.
            classifiers DataFrame): matrix of classifier ids
                associated with yield tables.
            inventory (DataFrame): Used by this function to find correct
                spatial parameters from the set of merch volume growth
                parameters. Will not be modified by this function.
            pools (DataFrame): matrix of shape
                n_stands by n_pools. Used by this function to compute a root
                increment, and also to limit negative growth increments such
                that a negative biomass pools are prevented.  This parameter
                is not modified by this function.
            state_variables (DataFrame): simulation variables which
                define all non-pool state in the CBM model.  This function
                call will not alter this parameter.
        """

        op_ids = (ctypes.c_size_t * (2))(*[growth_op, overmature_decline_op])

        self.handle.call(
            "LibCBM_GetMerchVolumeGrowthOps",
            op_ids,
            inventory.n_rows,
            LibCBM_Matrix_Int(classifiers.to_c_contiguous_numpy_array()),
            LibCBM_Matrix(pools.to_c_contiguous_numpy_array()),
            state_variables["age"].to_numpy(),
            inventory["spatial_unit"].to_numpy(),
            state_variables["last_disturbance_type"].to_numpy_ptr(),
            state_variables["time_since_last_disturbance"].to_numpy_ptr(),
            state_variables["growth_multiplier"].to_numpy_ptr(),
            state_variables["growth_enabled"].to_numpy_ptr(),
        )

    def get_turnover_ops(
        self,
        biomass_turnover_op: int,
        snag_turnover_op: int,
        inventory: DataFrame,
    ):
        """Computes biomass turnovers and dead organic matter turnovers as
        bulk matrix operations.

        Args:
            biomass_turnover_op (int): Handle for a block of matrices as
                allocated by the :py:func:`allocate_op` function. Used to
                compute biomass turnover operations.
            snag_turnover_op (int): Handle for a block of matrices as
                allocated by the :py:func:`allocate_op` function. Used to
                compute dom (specifically snags) turnover operations.
            inventory (DataFrame): CBM inventory data. Used by this function
                to find correct parameters from the set of turnover parameters
                passed to library initialization. Will not be modified by this
                function.
        """
        op_ids = (ctypes.c_size_t * (2))(
            *[biomass_turnover_op, snag_turnover_op]
        )

        self.handle.call(
            "LibCBM_GetTurnoverOps",
            op_ids,
            inventory.n_rows,
            inventory["spatial_unit"].to_numpy(),
        )

    def get_decay_ops(
        self,
        dom_decay_op: int,
        slow_decay_op: int,
        slow_mixing_op: int,
        inventory: DataFrame,
        parameters: DataFrame,
        historical_mean_annual_temp: bool = False,
    ):
        """Prepares dead organic matter decay bulk matrix operations.

        Args:
            dom_decay_op (int): Handle for a block of matrices as
                allocated by the :py:func:`allocate_op` function. Used to
                compute dom decay operations.
            slow_decay_op (int): Handle for a block of matrices as
                allocated by the :py:func:`allocate_op` function. Used to
                compute slow pool decay operations.
            slow_mixing_op (int): Handle for a block of matrices as
                allocated by the :py:func:`allocate_op` function. Used to
                compute slow pool mixing operations.
            inventory (object): CBM inventory data. Used by this function to
                find correct parameters from the set of decay parameters
                passed to library initialization. Will not be modified by this
                function.
            parameters (object): parameters for this timestep
            historical_mean_annual_temp (bool, optional): If set to true, the
                historical default mean annual temperature is used. This is
                intended for spinup.  If explicit mean annual temperature
                is provided via the parameters argument, this parameter will
                be ignored, and the explicit mean annual temp will be used.
                Defaults to False.
        """

        op_ids = (ctypes.c_size_t * (3))(
            *[dom_decay_op, slow_decay_op, slow_mixing_op]
        )

        mean_annual_temp = parameters["mean_annual_temp"].to_numpy_ptr()

        if mean_annual_temp is not None:
            # If the mean annual temperature is specified, then omit the
            # spatial unit, whose only purpose in the context of the decay
            # routine is to fetch the mean annual temperature value from
            # the CBM defaults database
            spatial_unit = None
        else:
            spatial_unit = inventory["spatial_unit"].to_numpy_ptr()

        self.handle.call(
            "LibCBM_GetDecayOps",
            op_ids,
            inventory.n_rows,
            spatial_unit,
            historical_mean_annual_temp,
            mean_annual_temp,
        )

    def get_disturbance_ops(
        self, disturbance_op: int, inventory: DataFrame, parameters: DataFrame
    ):
        """Sets up CBM disturbance matrices as a bulk matrix operations.

        Args:
            disturbance_op (int): Handle for a block of matrices as
                allocated by the :py:func:`AllocateOp` function. Used to
                compute disturbance event pool flows.
            inventory (DataFrame): CBM inventory data. Used by this function to
                find correct parameters from the set of disturbance parameters
                passed to library initialization. Will not be modified by this
                function.
            parameters (DataFrame): Read-only parameters used to set
                disturbance type id to fetch the appropriate disturbance
                matrix.
        """
        op_ids = (ctypes.c_size_t * (1))(*[disturbance_op])

        self.handle.call(
            "LibCBM_GetDisturbanceOps",
            op_ids,
            inventory.n_rows,
            inventory["spatial_unit"].to_numpy(),
            parameters["disturbance_type"].to_numpy(),
        )

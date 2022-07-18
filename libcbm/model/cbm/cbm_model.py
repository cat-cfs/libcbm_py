# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from typing import Callable
from typing import Union
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.cbm.cbm_wrapper import CBMWrapper
from libcbm.storage.series import Series
from libcbm.storage.series import SeriesDef
from libcbm.storage import dataframe


def get_op_names() -> list[str]:
    """Gets the names of the CBM dynamics operations

    Returns:
        list: the list of string names describing the dynamics ops
    """
    return [
        "growth",
        "snag_turnover",
        "biomass_turnover",
        "overmature_decline",
        "dom_decay",
        "slow_decay",
        "slow_mixing",
        "disturbance",
    ]


def get_op_processes() -> dict[str, int]:
    """Gets a dictionary of operation name to process id, which is
    used to group flux indicators

    Returns:
        dict: dictionary of key: operation name, value: process id
    """
    return {
        "growth": 1,
        "overmature_decline": 1,
        "snag_turnover": 1,
        "biomass_turnover": 1,
        "dom_decay": 2,
        "slow_decay": 2,
        "slow_mixing": 2,
        "disturbance": 3,
    }


class CBM:
    """The CBM model.

    Args:
        compute_functions (LibCBMWrapper): an instance of LibCBMWrapper.
        model_functions (CBMWrapper): an instance of CBMWrapper
        pool_codes (list): list of pool code names (non localizable)
        flux_indicator_codes (list): list of flux indicator code names (non
            localizable)
    """

    def __init__(
        self,
        compute_functions: LibCBMWrapper,
        model_functions: CBMWrapper,
        pool_codes: list[str],
        flux_indicator_codes: list[str],
    ):

        self.compute_functions = compute_functions
        self.model_functions = model_functions

        self.op_names = get_op_names()

        self.op_processes = get_op_processes()

        self.pool_codes = pool_codes
        self.flux_indicator_codes = flux_indicator_codes

    def spinup(
        self,
        cbm_vars: CBMVariables,
        reporting_func: Callable[[int, CBMVariables], None] = None,
    ) -> CBMVariables:
        """Run the CBM-CFS3 spinup function on an array of stands,
        initializing the specified variables.

        See :py:mod:`libcbm.model.cbm.cbm_variables` for initialization
        routines for the cbm_vars object.

        Args:
            cbm_vars (CBMVariables): spinup CBM variables
            reporting_func (function): a function which accepts the spinup
                iteration spinup variables for reporting results by spinup
                iteration. The function returns None.

        Returns:
            CBMVariables: cbm_vars
        """
        if cbm_vars.flux is not None and reporting_func is None:
            # can use reporting func without flux, but it does not make any
            # sense to compute flux and not use reporting func since the result
            # will not be visible
            raise ValueError("flux specified without reporting_func")

        n_stands = cbm_vars.pools.n_rows

        ops = {
            x: self.compute_functions.allocate_op(n_stands)
            for x in self.op_names
        }

        self.model_functions.get_turnover_ops(
            ops["snag_turnover"], ops["biomass_turnover"], cbm_vars.inventory
        )

        self.model_functions.get_decay_ops(
            ops["dom_decay"],
            ops["slow_decay"],
            ops["slow_mixing"],
            cbm_vars.inventory,
            cbm_vars.parameters,
            historical_mean_annual_temp=True,
        )

        op_schedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "overmature_decline",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing",
            "disturbance",
        ]

        iteration = 0

        while True:

            n_finished = self.model_functions.advance_spinup_state(
                cbm_vars.inventory, cbm_vars.state, cbm_vars.parameters
            )

            if n_finished == n_stands:
                break

            self.model_functions.get_merch_volume_growth_ops(
                ops["growth"],
                ops["overmature_decline"],
                cbm_vars.classifiers,
                cbm_vars.inventory,
                cbm_vars.pools,
                cbm_vars.state,
            )

            self.model_functions.get_disturbance_ops(
                ops["disturbance"], cbm_vars.inventory, cbm_vars.state
            )

            if cbm_vars.flux is None:
                self.compute_functions.compute_pools(
                    [ops[x] for x in op_schedule],
                    cbm_vars.pools,
                    cbm_vars.state["enabled"],
                )
            else:
                cbm_vars.flux.zero()
                self.compute_functions.compute_flux(
                    [ops[x] for x in op_schedule],
                    [self.op_processes[x] for x in op_schedule],
                    cbm_vars.pools,
                    cbm_vars.flux,
                    cbm_vars.state["enabled"],
                )
            if reporting_func:
                reporting_func(iteration, cbm_vars)

            self.model_functions.end_spinup_step(
                cbm_vars.pools, cbm_vars.state
            )
            iteration = iteration + 1

        for op_name in self.op_names:
            self.compute_functions.free_op(ops[op_name])
        return cbm_vars

    def init(self, cbm_vars: CBMVariables) -> CBMVariables:
        """Set the initial state of CBM variables after spinup and prior
        to starting CBM simulation stepping

        Args:
            cbm_vars (CBMVariables): cbm vars

        Returns:
            CBMVariables: cbm_vars
        """

        # the following line is tricky, and needs some more thought:
        # 1. defining the landclass as FL_FL is not problematic for regular
        #    forest cases.
        # 2. defining the landclass as a non forest landclass is not
        #    problematic for afforestation, and using a forested landclass
        #    is problematic.  This is handled by libcbm, in that an error is
        #    thrown if a forested land class is used with an afforestation
        #    pre-type.
        # 3. using a deforestation disturbance for
        #    inventory.last_pass_disturbance_type type should overwrite the
        #    state variable landclass since each deforestation disturbance
        #    type is explicitly associated with a transitional land class.
        #    This means that inventory.last_pass_disturbance_type along with
        #    inventory.delay indicates the correct value for
        #    state_variables.land_class at CBM startup, and it can potentially
        #    contradict the value of inventory.land_class. libcbm does not
        #    attempt to throw an error if this situation is detected.
        cbm_vars.state["land_class"].assign(cbm_vars.inventory["land_class"])

        self.model_functions.initialize_land_state(
            cbm_vars.inventory, cbm_vars.pools, cbm_vars.state
        )
        return cbm_vars

    def step_start(self, cbm_vars: CBMVariables) -> CBMVariables:
        """Advance stand state and initialize variables prior to disturbance
        and annual process steps.  Should be called one time at the start of
        each CBM time step.

        Args:
            cbm_vars (CBMVariables): cbm vars

        Returns:
            CBMVariables: cbm_vars
        """

        cbm_vars.flux.zero()

        self.model_functions.advance_stand_state(
            cbm_vars.classifiers,
            cbm_vars.inventory,
            cbm_vars.state,
            cbm_vars.parameters,
        )
        return cbm_vars

    def compute_disturbance_production(
        self,
        cbm_vars: CBMVariables,
        disturbance_type: Union[Series, int] = None,
        eligible: Series = None,
        density: bool = True,
    ):
        """Computes a series of disturbance production values based on the
        current pools in cbm_vars, and disturbance matrices associated with
        cbm_vars.parameters.disturbance type by default, and the specified
        disturbance_type scalar, or array otherwise.  Does not change values
        in cbm_vars.

        Args:
            cbm_vars (CBMVariables): object containing current simulation state
            disturbance_type (Series, int, optional): The integer code
                or array specifying the disturbance type(s). If not specified,
                the value of cbm_vars.parameters.disturbance_type is used.
            eligible (Series, optional): Bit values where True
                specifies the index is eligible for the disturbance, and
                false the opposite. In the returned result False indices
                will be set with 0's.  Specifying None is equivant to an
                full array of True values. Defaults to None.
            density (bool, optional): if set to True the return value is
                expressed in units of tonnes Carbon/hectare, and if False
                the return value is expressed in units of tonnes Carbon.
                Defaults to True.

        Returns:
            DataFrame: dataframe describing the C production associated
                with applying the specified disturbance type on the specified
                pools.  All columns are expressed as area density values
                in units tonnes C/ha.

                Fields:

                    - DisturbanceSoftProduction: the softwood C production
                    - DisturbanceSoftProduction: the hardwood C production
                    - DisturbanceDOMProduction: the dead organic matter C
                        production
                    - Total: the row sums of the above three values

        """
        # this is by convention in the cbm_defaults database
        disturbance_op_process_id = get_op_processes()["disturbance"]

        # The number of stands is the number of rows in the inventory table.
        n_stands = cbm_vars.inventory.n_rows

        # allocate space for computing the Carbon flows
        disturbance_op = self.compute_functions.allocate_op(n_stands)

        if (
            not isinstance(disturbance_type, Series)
            and disturbance_type is not None
        ):
            # set the disturbance type for all records
            parameters = dataframe.from_series_list(
                [
                    SeriesDef("disturbance_type", disturbance_type, "int32"),
                ],
                cbm_vars.inventory.n_rows,
                cbm_vars.inventory.backend_type,
            )
        else:
            # else: just use the provided array data
            parameters = dataframe.from_series_list(
                [
                    cbm_vars.parameters["disturbance_type"],
                ],
                cbm_vars.inventory.n_rows,
                cbm_vars.inventory.backend_type,
            )

        self.model_functions.get_disturbance_ops(
            disturbance_op,
            cbm_vars.inventory,
            parameters,
        )

        flux = dataframe.numeric_dataframe(
            cols=self.flux_indicator_codes,
            nrows=n_stands,
            back_end=cbm_vars.inventory.backend_type,
        )

        pools_copy = cbm_vars.pools.copy()

        # compute the flux based on the specified disturbance type
        self.compute_functions.compute_flux(
            [disturbance_op],
            [disturbance_op_process_id],
            pools_copy,
            flux,
            enabled=(eligible if eligible is not None else None),
        )

        self.compute_functions.free_op(disturbance_op)
        # computes C harvested by applying the disturbance matrix to the
        # specified carbon pools
        total_series = (
            flux["DisturbanceSoftProduction"]
            + flux["DisturbanceHardProduction"]
            + flux["DisturbanceDOMProduction"]
        )
        total_series.name = "Total"
        df = dataframe.from_series_list(
            [
                flux["DisturbanceSoftProduction"],
                flux["DisturbanceHardProduction"],
                flux["DisturbanceDOMProduction"],
                total_series,
            ],
            nrows=flux.n_rows,
            back_end=flux.backend_type,
        )
        if density:
            return df
        else:
            return df.multiply(cbm_vars.inventory["area"])

    def step_disturbance(self, cbm_vars: CBMVariables) -> CBMVariables:
        """Compute disturbance dynamics and compute disturbance flux on the
        current value of cbm_vars.pools.  The values stored in array
        `cbm_vars.parameters.disturbance_type` determines the disturbance
        type applied to each stand row.

        If conformance to CBM-CFS3 methodolgy is required, this function can
        safely be called multiple times per timestep, but note CBM-CFS does
        not support multiple disturbances to a single stand per timestep.
        This situation could arise if care is not taken to exclude stand
        indexes disturbed in prior calls to this function, within a given
        timestep, from all subsequent calls within that timestep.

        Args:
            cbm_vars (CBMVariables): cbm_vars object

        Returns:
            CBMVariables: cbm_vars
        """
        n_stands = cbm_vars.pools.n_rows
        disturbance_op = self.compute_functions.allocate_op(n_stands)
        self.model_functions.get_disturbance_ops(
            disturbance_op, cbm_vars.inventory, cbm_vars.parameters
        )

        self.compute_functions.compute_flux(
            [disturbance_op],
            [self.op_processes["disturbance"]],
            cbm_vars.pools,
            cbm_vars.flux,
            enabled=None,
        )
        # enabled = none on line above is due to a possible bug in CBM3. This
        # is very much an edge case:
        # stands can be disturbed despite having all other C-dynamics processes
        # disabled (which happens in peatland)
        self.compute_functions.free_op(disturbance_op)
        return cbm_vars

    def step_annual_process(self, cbm_vars: CBMVariables) -> CBMVariables:
        """Compute CBM annual process dynamics: growth, turnover and decay.
        This updates the cbm_vars.pools value and cbm_vars.flux values with
        computed annual process dynamics

        Args:
            cbm_vars (CBMVariables): cbm_vars object

        Returns:
            CBMVariables: cbm_vars
        """
        n_stands = cbm_vars.pools.n_rows

        ops = {
            x: self.compute_functions.allocate_op(n_stands)
            for x in self.op_names
        }

        self.model_functions.get_merch_volume_growth_ops(
            ops["growth"],
            ops["overmature_decline"],
            cbm_vars.classifiers,
            cbm_vars.inventory,
            cbm_vars.pools,
            cbm_vars.state,
        )

        self.model_functions.get_turnover_ops(
            ops["snag_turnover"], ops["biomass_turnover"], cbm_vars.inventory
        )

        self.model_functions.get_decay_ops(
            ops["dom_decay"],
            ops["slow_decay"],
            ops["slow_mixing"],
            cbm_vars.inventory,
            cbm_vars.parameters,
        )

        annual_process_op_schedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "overmature_decline",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing",
        ]

        self.compute_functions.compute_flux(
            [ops[x] for x in annual_process_op_schedule],
            [self.op_processes[x] for x in annual_process_op_schedule],
            cbm_vars.pools,
            cbm_vars.flux,
            cbm_vars.state["enabled"],
        )
        for op_name in self.op_names:
            self.compute_functions.free_op(ops[op_name])
        return cbm_vars

    def step_end(self, cbm_vars: CBMVariables) -> CBMVariables:
        """Apply end of timestep state changes.  Updates values in
        cbm_vars.state.

        Args:
            cbm_vars (CBMVariables): cbm_vars object

        Returns:
            CBMVariables: cbm_vars
        """
        self.model_functions.end_step(cbm_vars.state)
        return cbm_vars

    def step(self, cbm_vars: CBMVariables) -> CBMVariables:
        """Run all default cbm step methods.  It is assumed that any
        records in the specified cbm_vars that require the spinup routine
        have been passed into the :py:func:`init` and :py:func:`spinup`
        and methods in this class.

        Args:
            cbm_vars (CBMVariables): cbm_vars object

        Returns:
            CBMVariables: cbm_vars
        """
        cbm_vars = self.step_start(cbm_vars)
        cbm_vars = self.step_disturbance(cbm_vars)
        cbm_vars = self.step_annual_process(cbm_vars)
        cbm_vars = self.step_end(cbm_vars)
        return cbm_vars

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import pandas as pd
import numpy as np
from types import SimpleNamespace
from libcbm.model.cbm import cbm_variables


def get_op_names():
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
        "disturbance"]


def get_op_processes():
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
        "disturbance": 3}


class CBM:
    """The CBM model.

    Args:
        compute_functions (libcbm.wrapper.libcbm_wrapper.LibCBMWrapper): an
            instance of LibCBMWrapper.
        model_functions (libcbm.wrapper.cbm.cbm_wrapper.CBMWrapper): an
            instance of CBMWrapper
        pool_codes (list): list of pool code names (non localizable)
        flux_indicator_codes (list): list of flux indicator code names (non
            localizable)
    """
    def __init__(self, compute_functions, model_functions, pool_codes,
                 flux_indicator_codes):

        self.compute_functions = compute_functions
        self.model_functions = model_functions

        self.op_names = get_op_names()

        self.op_processes = get_op_processes()

        self.pool_codes = pool_codes
        self.flux_indicator_codes = flux_indicator_codes

    def spinup(self, cbm_vars, reporting_func=None):
        """Run the CBM-CFS3 spinup function on an array of stands,
        initializing the specified variables.

        See :py:mod:`libcbm.model.cbm.cbm_variables` for initialization
        routines for the cbm_vars object.

        Args:
            cbm_vars (object): spinup vars with the following fields:

                * pools: CBM pools of dimension n_stands by n_pools.
                  Initialized with spinup carbon values by this function.
                  Column order is important. See:
                  :py:func:`libcbm.model.cbm.cbm_variables.initialize_pools`
                  for a compatible definition
                * flux: CBM flux values of dimension n_stands by
                  n_flux_indicators. Set with the flux indicator values for
                  pool flows that occur for each spinup timestep. See:
                  :py:func:`libcbm.model.cbm.cbm_variables.initialize_flux`
                  for a compatible definition.  If set to None, flux for each
                  spinup step will not be computed. Defaults to None.
                * parameters: spinup parameters. See:
                  :py:func:`libcbm.model.cbm.cbm_variables.initialize_spinup_parameters`
                  for a compatible definition
                * state: spinup working variables. See:
                  :py:func:`libcbm.model.cbm.cbm_variables.initialize_spinup_variables`
                * inventory: cbm inventory data. Not be modified by this
                  function.
                * classifiers: matrix of numeric classifier value ids
                  associated with inventory. Not modified by this function.

            reporting_func (function): a function which accepts the spinup
                iteration spinup variables for reporting results by spinup
                iteration. The function returns None.

        Returns:
            object: cbm_vars
        """
        if cbm_vars.flux is not None and reporting_func is None:
            # can use reporting func without flux, but it does not make any
            # sense to compute flux and not use reporting func since the result
            # will not be visible
            raise ValueError("flux specified without reporting_func")

        n_stands = cbm_vars.pools.shape[0]

        ops = {
            x: self.compute_functions.allocate_op(n_stands)
            for x in self.op_names}

        self.model_functions.get_turnover_ops(
            ops["snag_turnover"], ops["biomass_turnover"], cbm_vars.inventory)

        self.model_functions.get_decay_ops(
            ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"],
            cbm_vars.inventory, cbm_vars.parameters,
            historical_mean_annual_temp=True)

        op_schedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "overmature_decline",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing",
            "disturbance"
            ]

        iteration = 0

        while True:

            n_finished = self.model_functions.advance_spinup_state(
                cbm_vars.inventory, cbm_vars.state, cbm_vars.parameters)

            if n_finished == n_stands:
                break

            self.model_functions.get_merch_volume_growth_ops(
                ops["growth"], ops["overmature_decline"], cbm_vars.classifiers,
                cbm_vars.inventory, cbm_vars.pools, cbm_vars.state)

            self.model_functions.get_disturbance_ops(
                ops["disturbance"], cbm_vars.inventory, cbm_vars.state)

            if cbm_vars.flux is None:
                self.compute_functions.compute_pools(
                    [ops[x] for x in op_schedule], cbm_vars.pools,
                    cbm_vars.state.enabled)
            else:
                # zero the memory (simply using flux *= 0.0 caused a copy
                # with a change in contiguity in some cases!)
                if isinstance(cbm_vars.flux, pd.DataFrame):
                    cbm_vars.flux.values[:] = 0
                else:
                    cbm_vars.flux[:] = 0
                self.compute_functions.compute_flux(
                    [ops[x] for x in op_schedule],
                    [self.op_processes[x] for x in op_schedule],
                    cbm_vars.pools, cbm_vars.flux, cbm_vars.state.enabled
                )
            if reporting_func:
                reporting_func(iteration, SimpleNamespace(
                    pools=cbm_vars.pools,
                    flux=cbm_vars.flux,
                    state=pd.DataFrame(data={
                        k: v for k, v
                        in vars(cbm_vars.state).items()
                        if v is not None}),
                    classifiers=cbm_vars.classifiers,
                    parameters=pd.DataFrame(data={
                        k: v for k, v
                        in vars(cbm_vars.parameters).items()
                        if v is not None}),
                    inventory=cbm_vars.inventory
                ))

            self.model_functions.end_spinup_step(
                cbm_vars.pools, cbm_vars.state)
            iteration = iteration + 1

        for op_name in self.op_names:
            self.compute_functions.free_op(ops[op_name])
        return cbm_vars

    def init(self, cbm_vars):
        """Set the initial state of CBM variables after spinup and prior
        to starting CBM simulation stepping

        See:
        :py:mod:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
        for definition of the cbm_vars object

        Args:
            cbm_vars (object): cbm vars

        Returns:
            object: cbm_vars
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
        cbm_vars.state.land_class = cbm_vars.inventory.land_class

        self.model_functions.initialize_land_state(
            cbm_vars.inventory, cbm_vars.pools, cbm_vars.state)
        return cbm_vars

    def step_start(self, cbm_vars):
        """Advance stand state and initialize variables prior to disturbance
        and annual process steps.  Should be called one time at the start of
        each CBM time step.

        See:
        :py:mod:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
        for definition of the cbm_vars object

        Args:
            cbm_vars (object): cbm vars

        Returns:
            object: cbm_vars
        """
        # zero the memory (simply using flux *= 0.0 caused a copy
        # with a change in contiguity in some cases!)
        if isinstance(cbm_vars.flux, pd.DataFrame):
            cbm_vars.flux.values[:] = 0
        else:
            cbm_vars.flux[:] = 0

        self.model_functions.advance_stand_state(
            cbm_vars.classifiers, cbm_vars.inventory, cbm_vars.state,
            cbm_vars.parameters)
        return cbm_vars

    def compute_disturbance_production(self, cbm_vars, disturbance_type,
                                       eligible=None, density=True):
        """Computes a series of disturbance production values based on the
        current pools in cbm_vars, and disturbance matrices associated with
        cbm_vars.parameters.disturbance type by default, and the specified
        disturbance_type scalar, or array otherwise.  Does not change values
        in cbm_vars.

        Args:
            cbm_vars (object): object containing current simulation state
            disturbance_type (numpy.ndarray, int): The integer code or array
                specifying the disturbance type(s).
            eligible (numpy.ndarray, optional): Bit values where True
                specifies the index is eligible for the disturbance, and
                false the opposite. In the returned result False indices
                will be set with 0's.  Specifying None is equivant to an
                full array of True values. Defaults to None.
            density (bool, optional): if set to True the return value is
                expressed in units of tonnes Carbon/hectare, and if False
                the return value is expressed in units of tonnes Carbon.
                Defaults to True.

        Returns:
            pandas.DataFrame: dataframe describing the C production associated
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
        # The set of inventory here is assumed to be the eligible for
        # disturbance filtered subset of records
        n_stands = cbm_vars.inventory.shape[0]

        # allocate space for computing the Carbon flows
        disturbance_op = self.compute_functions.allocate_op(n_stands)

        if np.ndim(disturbance_type) == 0:
            # set the disturbance type for all records
            disturbance_type = SimpleNamespace(
                disturbance_type=np.full(
                    n_stands, disturbance_type, dtype=np.int32))
        else:
            disturbance_type = SimpleNamespace(
                disturbance_type=disturbance_type)
        self.model_functions.get_disturbance_ops(
            disturbance_op, cbm_vars.inventory, disturbance_type)

        # zero the flux indicators
        cbm_vars.flux.values[:] = 0

        cbm_vars = cbm_variables.prepare(cbm_vars)

        pools_copy = np.array(
            cbm_vars.pools, copy=True, order="C", dtype=np.float64)

        # compute the flux based on the specified disturbance type
        self.compute_functions.compute_flux(
            [disturbance_op], [disturbance_op_process_id],
            pools_copy, cbm_vars.flux,
            enabled=(
                eligible.astype(np.int32)
                if eligible is not None else None))

        self.compute_functions.free_op(disturbance_op)
        # computes C harvested by applying the disturbance matrix to the
        # specified carbon pools
        df = pd.DataFrame(data={
            "DisturbanceSoftProduction":
                cbm_vars.flux["DisturbanceSoftProduction"],
            "DisturbanceHardProduction":
                cbm_vars.flux["DisturbanceHardProduction"],
            "DisturbanceDOMProduction":
                cbm_vars.flux["DisturbanceDOMProduction"],
            "Total":
                cbm_vars.flux["DisturbanceSoftProduction"] +
                cbm_vars.flux["DisturbanceHardProduction"] +
                cbm_vars.flux["DisturbanceDOMProduction"]})
        if density:
            return df
        else:
            return df.multiply(cbm_vars.inventory.area, axis=0)

    def step_disturbance(self, cbm_vars):
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

        See:
        :py:mod:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
        for definition of the cbm_vars object

        Args:
            cbm_vars (object): cbm_vars object

        Returns:
            object: cbm_vars
        """
        n_stands = cbm_vars.pools.shape[0]
        disturbance_op = self.compute_functions.allocate_op(n_stands)
        self.model_functions.get_disturbance_ops(
            disturbance_op, cbm_vars.inventory, cbm_vars.parameters)

        self.compute_functions.compute_flux(
            [disturbance_op], [self.op_processes["disturbance"]],
            cbm_vars.pools, cbm_vars.flux, enabled=None)
        # enabled = none on line above is due to a possible bug in CBM3. This
        # is very much an edge case:
        # stands can be disturbed despite having all other C-dynamics processes
        # disabled (which happens in peatland)
        self.compute_functions.free_op(disturbance_op)
        return cbm_vars

    def step_annual_process(self, cbm_vars):
        """Compute CBM annual process dynamics: growth, turnover and decay.
        This updates the cbm_vars.pools value and cbm_vars.flux values with
        computed annual process dynamics

        See:
        :py:mod:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
        for definition of the cbm_vars object

        Args:
            cbm_vars (object): cbm_vars object

        Returns:
            object: cbm_vars
        """
        n_stands = cbm_vars.pools.shape[0]

        ops = {
            x: self.compute_functions.allocate_op(n_stands)
            for x in self.op_names}

        self.model_functions.get_merch_volume_growth_ops(
            ops["growth"], ops["overmature_decline"], cbm_vars.classifiers,
            cbm_vars.inventory, cbm_vars.pools, cbm_vars.state)

        self.model_functions.get_turnover_ops(
            ops["snag_turnover"], ops["biomass_turnover"],
            cbm_vars.inventory)

        self.model_functions.get_decay_ops(
            ops["dom_decay"], ops["slow_decay"], ops["slow_mixing"],
            cbm_vars.inventory, cbm_vars.parameters)

        annual_process_op_schedule = [
            "growth",
            "snag_turnover",
            "biomass_turnover",
            "overmature_decline",
            "growth",
            "dom_decay",
            "slow_decay",
            "slow_mixing"
            ]

        self.compute_functions.compute_flux(
            [ops[x] for x in annual_process_op_schedule],
            [self.op_processes[x] for x in annual_process_op_schedule],
            cbm_vars.pools, cbm_vars.flux, cbm_vars.state.enabled)
        for op_name in self.op_names:
            self.compute_functions.free_op(ops[op_name])
        return cbm_vars

    def step_end(self, cbm_vars):
        """Apply end of timestep state changes.  Updates values in
        cbm_vars.state.

        See:
        :py:mod:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
        for definition of the cbm_vars object

        Args:
            cbm_vars (object): cbm_vars object

        Returns:
            object: cbm_vars
        """
        self.model_functions.end_step(cbm_vars.state)
        return cbm_vars

    def step(self, cbm_vars):
        """Run all default cbm step methods.  It is assumed that any
        records in the specified cbm_vars that require the spinup routine
        have been passed into the :py:func:`init` and :py:func:`spinup`
        and methods in this class.


        See:
        :py:mod:`libcbm.model.cbm.cbm_variables.initialize_simulation_variables`
        for definition of the cbm_vars object

        Args:
            cbm_vars (object): cbm_vars object

        Returns:
            object: cbm_vars
        """
        cbm_vars = self.step_start(cbm_vars)
        cbm_vars = self.step_disturbance(cbm_vars)
        cbm_vars = self.step_annual_process(cbm_vars)
        cbm_vars = self.step_end(cbm_vars)
        return cbm_vars

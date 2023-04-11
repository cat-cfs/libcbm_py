from __future__ import annotations
from typing import Union
from typing import Dict
from typing import Iterator
from contextlib import contextmanager
import pandas as pd
import numpy as np
from libcbm import resources
from libcbm.model.model_definition import model
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.model_definition.output_processor import ModelOutputProcessor
from libcbm.model.cbm_exn import cbm_exn_spinup
from libcbm.model.cbm_exn import cbm_exn_step
from libcbm.model.cbm_exn.cbm_exn_parameters import parameters_factory
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.cbm_exn.cbm_exn_matrix_ops import MatrixOps
from libcbm.wrapper.libcbm_operation import Operation

cbm_vars_type = Union[ModelVariables, Dict[str, pd.DataFrame]]


class SpinupReporter:
    """Tracks step-by-step results during spinup for debugging purposes."""

    def __init__(self, pandas_interface: bool):
        """initialize a SpinupReporter

        Args:
            pandas_interface (bool): if true ModelVariables members are assumed
                to be pandas dataframes, and otherwise the internal DataFrame
                interfaces
        """
        self._output_processor = ModelOutputProcessor()
        self._pandas_interface = pandas_interface

    def append_spinup_output(
        self, timestep: int, spinup_vars: ModelVariables
    ) -> None:
        """Append a set of results to the spinup reporter

        Args:
            timestep (int): the step number
            spinup_vars (ModelVariables): the spinup variables and state
        """
        self._output_processor.append_results(timestep, spinup_vars)

    def get_output(self) -> cbm_vars_type:
        """Gets the accumulated spinup results"""
        return self._output_processor.get_results()


class CBMEXNModel:
    """The CBMEXNModel: CBM-CFS3 with external net Aboveground Carbon biomass
    increments
    """

    def __init__(
        self,
        cbm_model: CBMModel,
        parameters: CBMEXNParameters,
        pandas_interface=True,
        spinup_reporter: SpinupReporter = None,
    ):
        """initialize the CBMEXNModel

        Args:
            cbm_model (CBMModel): abstraction of CBM model
            parameters (CBMEXNParameters): cbm_constant parameter
            pandas_interface (bool, optional): If set to true then all members
                of CBMVariables are assumed to be of type pd.DataFrame, if
                false then the internally defined libcbm DataFrame is used.
                Defaults to True.
            spinup_reporter (SpinupReporter, optional): If specified, spinup
                results are tracked for debugging purposes. Defaults to None.
        """
        self._cbm_model = cbm_model
        self._pandas_interface = pandas_interface
        self._spinup_reporter = spinup_reporter
        self._parameters = parameters
        self._matrix_ops = MatrixOps(cbm_model, self._parameters)

    @property
    def pool_names(self) -> list[str]:
        """Get the list of pool names, also used as the column headers for
        the cbm_vars pools dataframe

        Returns:
            list[str]: the pool names
        """
        return self._cbm_model.pool_names

    @property
    def flux_names(self) -> list[str]:
        """Get the list of flux indicator names, these form the columns for
        the flux dataframe.

        Returns:
            list[str]: the flux indicator names
        """
        return self._cbm_model.flux_names

    @property
    def matrix_ops(self) -> MatrixOps:
        """MatrixOps instance which computes and caches C matrix flows

        Returns:
            MatrixOps: instance of MatrixOps
        """
        return self._matrix_ops

    @property
    def parameters(self) -> CBMEXNParameters:
        """CBM constant parameters

        Returns:
            CBMEXNParameters: CBM constant parameters
        """
        return self._parameters

    def step(self, cbm_vars: cbm_vars_type) -> cbm_vars_type:
        """Perform one timestep of the CBMEXNModel

        Args:
            cbm_vars (cbm_vars_type): the simulation state
                and variables

        Returns:
            cbm_vars_type: modified state and variables.
        """
        if self._pandas_interface:
            return cbm_exn_step.step(
                self, ModelVariables.from_pandas(cbm_vars)
            ).to_pandas()
        else:
            return cbm_exn_step.step(self, cbm_vars)

    def spinup(self, spinup_input: cbm_vars_type) -> cbm_vars_type:
        """initializes Carbon pools along the row axis of the specified
        spinup input using the CBM-CFS3 approach for spinup.

        Args:
            spinup_input (cbm_vars_type): spinup variables and parameters

        Returns:
            cbm_vars_type: initlaized CBM variables and state, prepared
                for CBM stepping.
        """
        reporting_func = (
            self._spinup_reporter.append_spinup_output
            if self._spinup_reporter
            else None
        )
        if self._pandas_interface:
            return cbm_exn_spinup.spinup(
                self,
                ModelVariables.from_pandas(spinup_input),
                reporting_func=reporting_func,
                include_flux=reporting_func is not None,
            ).to_pandas()
        else:
            return cbm_exn_spinup.spinup(
                self,
                spinup_input,
                reporting_func=reporting_func,
                include_flux=reporting_func is not None,
            )

    def get_spinup_output(self) -> cbm_vars_type:
        """If spinup debugging was enabled during construction of this class,
        a time-step by time-step account of the spinup routine for all stands
        is returned.

        Raises:
            ValueError: this class instance was not initialized to track spinup
                results.

        Returns:
            cbm_vars_type: the timestep by timestep spinup results
        """
        if not self._spinup_reporter:
            raise ValueError("spinup reporter not initialized")
        else:
            return self._spinup_reporter.get_output()

    def create_operation(
        self,
        matrices: list,
        fmt: str,
        matrix_index: np.ndarray,
        process_id: int,
    ) -> Operation:
        """Creates an matrix based C flow operation.  An operation defines 1
        or more matrix to apply to stand's Carbon pools.  The matrices have a
        1:m relationship to stands.

        For information on format see:
        :py:func:`libcbm.model.model_definition.model_handle.ModelHandle.create_operation`

        Args:
            matrices (list): a list of matrices, whose format is described by
                the `fmt` parameter
            fmt (str): the matrix format of the `matrices` parameter
            matrix_index (np.ndarray): an integer array along the stand axis
                whose value is the index of the matrix to apply to that stand
                index.
            process_id (int): process id is used to define which flux
                indicators this operation applies to

        Returns:
            Operation: the initialized operation.
        """
        return self._cbm_model.create_operation(
            matrices, fmt, matrix_index, process_id
        )

    def compute(
        self,
        cbm_vars: ModelVariables,
        operations: list[Operation],
    ):
        """Apply several sequential operations to the pools, and flux stored
        in the specified `cbm_vars`.

        Args:
            cbm_vars (ModelVariables): Collection of CBM simulation variables.
                This function modifies the `pools` and `flux` dataframes stored
                within cbm_vars.
            operations (list[Operation]): The list of matrix operations to
                apply
        """
        self._cbm_model.compute(cbm_vars, operations)


@contextmanager
def initialize(
    parameters: dict = None,
    config_path: str = None,
    pandas_interface: bool = True,
    include_spinup_debug: bool = False,
) -> Iterator[CBMEXNModel]:
    """Initialize CBMEXNModel

    Args:
        parameters (dict, optional): a dictionary of named parameters for
            running the cbm_exn module.  During initialization any required
            parameters that are not defined in this dictionary will be drawn
            from the specified config_path.
        config_path (str, optional): path to directory containing cbm_exn
            parameters in json and csv formats.  If unspecified, packaged
            defaults are used.
            See: :py:func:`libcbm.resources.get_cbm_exn_parameters_dir`.
        pandas_interface (bool, optional): if set to true then all members
            of CBMVariables are assumed to be of type pd.DataFrame, if
            false then the internally defined libcbm DataFrame is used.
            Defaults to True.
        include_spinup_debug (bool, optional): If set to true, the
            `get_spinup_output` of the returned class instance can be used to
            inspect timestep-by-timestep spinup output.  This will cause slow
            spinup performance. Defaults to False.

    Yields:
        Iterator[CBMEXNModel]: instance of CBMEXNModel
    """

    if not config_path:
        config_path = resources.get_cbm_exn_parameters_dir()

    params = parameters_factory(config_path, parameters)
    with model.initialize(
        pool_config=params.pool_configuration(),
        flux_config=params.flux_configuration(),
    ) as cbm_model:
        spinup_reporter = (
            SpinupReporter(pandas_interface) if include_spinup_debug else None
        )

        yield CBMEXNModel(
            cbm_model,
            params,
            pandas_interface=pandas_interface,
            spinup_reporter=spinup_reporter,
        )

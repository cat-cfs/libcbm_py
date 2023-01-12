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
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.model_definition.output_processor import ModelOutputProcessor
from libcbm.model.cbm_exn import cbm_exn_spinup
from libcbm.model.cbm_exn import cbm_exn_step
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.cbm_exn.cbm_exn_matrix_ops import MatrixOps
from libcbm.wrapper.libcbm_operation import Operation

cbm_vars_type = Union[CBMVariables, Dict[str, pd.DataFrame]]


class SpinupReporter:
    def __init__(self, pandas_interface: bool):
        self._output_processor = ModelOutputProcessor()
        self._pandas_interface = pandas_interface

    def append_spinup_output(
        self, timestep: int, spinup_vars: CBMVariables
    ) -> None:
        self._output_processor.append_results(timestep, spinup_vars)

    def get_output(self) -> cbm_vars_type:
        return self._output_processor.get_results()


class CBMEXNModel:
    def __init__(
        self,
        cbm_model: CBMModel,
        parameters: CBMEXNParameters,
        pandas_interface=True,
        spinup_reporter: SpinupReporter = None,
    ):
        self._cbm_model = cbm_model
        self._pandas_interface = pandas_interface
        self._spinup_reporter = spinup_reporter
        self._parameters = parameters
        self._matrix_ops = MatrixOps(cbm_model, self._parameters)

    @property
    def pool_names(self) -> list[str]:
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
        return self._matrix_ops

    @property
    def parameters(self) -> CBMEXNParameters:
        return self._parameters

    def step(self, cbm_vars: cbm_vars_type) -> cbm_vars_type:
        if self._pandas_interface:
            return cbm_exn_step.step(
                self, CBMVariables.from_pandas(cbm_vars)
            ).to_pandas()
        else:
            return cbm_exn_step.step(self, cbm_vars)

    def spinup(self, spinup_input: cbm_vars_type) -> cbm_vars_type:
        reporting_func = (
            self._spinup_reporter.append_spinup_output
            if self._spinup_reporter
            else None
        )
        if self._pandas_interface:
            return cbm_exn_spinup.spinup(
                self,
                CBMVariables.from_pandas(spinup_input),
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
        return self._cbm_model.create_operation(
            matrices, fmt, matrix_index, process_id
        )

    def compute(
        self,
        cbm_vars: CBMVariables,
        operations: list[Operation],
    ):
        self._cbm_model.compute(cbm_vars, operations)


@contextmanager
def initialize(
    config_path: str, pandas_interface=True, include_spinup_debug=False
) -> Iterator[CBMEXNModel]:

    if not config_path:
        config_path = resources.get_cbm_exn_parameters_dir()
    parameters = CBMEXNParameters(config_path)
    with model.initialize(
        pool_config=parameters.pool_configuration(),
        flux_config=parameters.flux_configuration(),
    ) as cbm_model:

        spinup_reporter = (
            SpinupReporter(pandas_interface) if include_spinup_debug else None
        )

        yield CBMEXNModel(
            cbm_model,
            parameters,
            pandas_interface=pandas_interface,
            spinup_reporter=spinup_reporter,
        )

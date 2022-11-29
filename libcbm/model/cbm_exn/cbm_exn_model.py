from typing import Callable
from typing import Union
from contextlib import contextmanager
import pandas as pd
from libcbm.model.model_definition import model
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.cbm_variables import CBMVariables
from libcbm.model.model_definition.output_processor import ModelOutputProcessor
from libcbm.model.cbm_exn import cbm_exn_spinup
from libcbm.model.cbm_exn import cbm_exn_step
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters
from libcbm.model.cbm_exn.cbm_exn_matrix_ops import MatrixOps
from libcbm.wrapper.libcbm_operation import Operation

cbm_vars_type = Union[CBMVariables, dict[str, pd.DataFrame]]


class SpinupReporter:
    def __init__(self, pandas_interface: bool):
        self._output_processor = ModelOutputProcessor()
        self._pandas_interface = pandas_interface

    def append_spinup_output(
        self, timestep: int, spinup_vars: CBMVariables
    ) -> None:
        self._output_processor.append_results(
            timestep, spinup_vars.get_collection()
        )

    def get_output(self) -> cbm_vars_type:
        return self._output_processor.get_results()


def spinup_debug_method(
    cbm_model: CBMModel, spinup_input: CBMVariables
) -> CBMVariables:
    return cbm_exn_spinup.spinup(cbm_model, spinup_input, None, False)


def get_spinup_method(
    spinup_reporter: SpinupReporter = None,
) -> Callable[[CBMModel, cbm_vars_type], cbm_vars_type]:
    if not spinup_reporter:
        return lambda cbm_model, spinup_input: cbm_exn_spinup.spinup(
            cbm_model, spinup_input, None, False
        )
    else:
        return lambda cbm_model, spinup_input: cbm_exn_spinup.spinup(
            cbm_model,
            spinup_input,
            spinup_reporter.append_spinup_output,
            False,
        )


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
    def matrix_ops(self) -> MatrixOps:
        return self._matrix_ops

    @property
    def parameters(self) -> CBMEXNParameters:
        return self._parameters

    def step(self, cbm_vars: cbm_vars_type) -> cbm_vars_type:
        if self._pandas_interface:
            return self._cbm_model.step(
                CBMVariables.from_pandas(cbm_vars)
            ).to_pandas()
        else:
            return self._cbm_model.step(cbm_vars)

    def spinup(self, spinup_input: cbm_vars_type) -> cbm_vars_type:
        if self._pandas_interface:
            return self._cbm_model.step(
                CBMVariables.from_pandas(spinup_input)
            ).to_pandas()
        else:
            return self._cbm_model.spinup(spinup_input)

    def get_spinup_output(self) -> cbm_vars_type:
        if not self._spinup_reporter:
            raise ValueError("spinup reporter not initialized")
        else:
            return self._spinup_reporter.get_output()

    def create_operation(
        self, matrices: list, fmt: str, process_id: int
    ) -> Operation:
        return self._cbm_model.create_operation(matrices, fmt, process_id)

    def compute(
        self,
        cbm_vars: CBMVariables,
        operations: list[Operation],
    ):
        self._cbm_model.compute(cbm_vars, operations)


@contextmanager
def initialize(
    config_path: str, pandas_interface=True, include_spinup_debug=False
) -> CBMEXNModel:

    parameters = CBMEXNParameters(config_path)
    with model.initialize(
        pool_config=parameters.pool_configuration(),
        flux_config=parameters.flux_configuration(),
        spinup_func=get_spinup_method(include_spinup_debug),
        step_func=cbm_exn_step.step,
    ) as cbm_model:

        spinup_reporter = (
            SpinupReporter(pandas_interface) if include_spinup_debug else None
        )

        yield CBMEXNModel(
            cbm_model,
            parameters,
            pandas_interface=True,
            spinup_reporter=spinup_reporter,
        )

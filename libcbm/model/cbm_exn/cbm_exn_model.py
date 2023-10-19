from __future__ import annotations
from typing import Union
from typing import Dict
from typing import Iterator
from contextlib import contextmanager
import pandas as pd
from libcbm import resources
from libcbm.model.model_definition import model
from libcbm.model.model_definition.model import CBMModel
from libcbm.model.model_definition.model_matrix_ops import ModelMatrixOps
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.model_definition.output_processor import ModelOutputProcessor
from libcbm.model.cbm_exn import cbm_exn_spinup
from libcbm.model.cbm_exn import cbm_exn_step
from libcbm.model.cbm_exn.cbm_exn_parameters import parameters_factory
from libcbm.model.cbm_exn.cbm_exn_parameters import CBMEXNParameters


cbm_vars_type = Union[ModelVariables, Dict[str, pd.DataFrame]]


class SpinupReporter:
    """Tracks step-by-step results during spinup for debugging purposes."""

    def __init__(self):
        """initialize a SpinupReporter

        Args:
            pandas_interface (bool): if true ModelVariables members are assumed
                to be pandas dataframes, and otherwise the internal DataFrame
                interfaces
        """
        self._output_processor = ModelOutputProcessor()

    def append_spinup_output(
        self, timestep: int, spinup_vars: ModelVariables
    ) -> None:
        """Append a set of results to the spinup reporter

        Args:
            timestep (int): the step number
            spinup_vars (ModelVariables): the spinup variables and state
        """
        self._output_processor.append_results(timestep, spinup_vars)

    def get_output(self) -> ModelVariables:
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
        spinup_reporter: Union[SpinupReporter, None] = None,
    ):
        """initialize the CBMEXNModel

        Args:
            cbm_model (CBMModel): abstraction of CBM model
            parameters (CBMEXNParameters): cbm_constant parameter
            spinup_reporter (SpinupReporter, optional): If specified, spinup
                results are tracked for debugging purposes. Defaults to None.
        """
        self._cbm_model = cbm_model
        self._spinup_reporter = spinup_reporter
        self._parameters = parameters

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
    def matrix_ops(self) -> ModelMatrixOps:
        """ModelMatrixOps instance which computes and caches C matrix flows

        Returns:
            ModelMatrixOps: instance of ModelMatrixOps
        """
        return self._cbm_model.matrix_ops

    @property
    def parameters(self) -> CBMEXNParameters:
        """CBM constant parameters

        Returns:
            CBMEXNParameters: CBM constant parameters
        """
        return self._parameters

    def step(
        self,
        cbm_vars: cbm_vars_type,
        ops: Union[list[dict], None] = None,
        disturbance_op_sequence: Union[list[str], None] = None,
        step_op_sequence: Union[list[str], None] = None,
    ) -> cbm_vars_type:
        """Perform one timestep of the CBMEXNModel

        Args:
            cbm_vars (cbm_vars_type): the simulation state
                and variables

        Returns:
            cbm_vars_type: modified state and variables.
        """
        return_pandas_dict = False
        if isinstance(cbm_vars, dict):
            return_pandas_dict = True
            _cbm_vars = ModelVariables.from_pandas(cbm_vars)
        else:
            _cbm_vars = cbm_vars

        result = cbm_exn_step.step(
            self,
            _cbm_vars,
            ops,
            step_op_sequence,
            disturbance_op_sequence,
        )

        if return_pandas_dict:
            return result.to_pandas()
        else:
            return result

    def spinup(
        self,
        spinup_input: cbm_vars_type,
        ops: Union[list[dict], None] = None,
        op_sequence: Union[list[str], None] = None,
    ) -> cbm_vars_type:
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
        return_pandas_dict = False
        if isinstance(spinup_input, dict):
            return_pandas_dict = True
            _spinup_input = ModelVariables.from_pandas(spinup_input)
        else:
            _spinup_input = spinup_input
        spinup_vars = cbm_exn_spinup.prepare_spinup_vars(
            _spinup_input,
            self.parameters,
        )
        result = cbm_exn_spinup.spinup(
            self,
            spinup_vars,
            reporting_func=reporting_func,
            ops=ops,
            op_sequence=op_sequence,
        )

        if return_pandas_dict:
            return result.to_pandas()
        else:
            return result

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

    def compute(
        self,
        cbm_vars: ModelVariables,
        op_names: list[str],
    ):
        """Apply several sequential operations to the pools, and flux stored
        in the specified `cbm_vars`.

        Args:
            cbm_vars (ModelVariables): Collection of CBM simulation variables.
                This function modifies the `pools` and `flux` dataframes stored
                within cbm_vars.
            op_names (list[str]): The list of matrix operations names to
                apply
        """

        self._cbm_model.compute(
            cbm_vars,
            self._cbm_model.matrix_ops.get_operations(op_names, cbm_vars),
        )


@contextmanager
def initialize(
    parameters: Union[dict, None] = None,
    config_path: Union[str, None] = None,
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
        spinup_reporter = SpinupReporter() if include_spinup_debug else None

        m = CBMEXNModel(
            cbm_model,
            params,
            spinup_reporter=spinup_reporter,
        )
        yield m
        m.matrix_ops.dispose()

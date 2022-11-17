from typing import Callable
from typing import Iterator
from contextlib import contextmanager
from libcbm.model.cbm_exn.cbm_variables import CBMVariables
from libcbm.model.cbm_exn.cbm_variables import SpinupInput
from libcbm.model.cbm_exn import cbm_exn_spinup
from libcbm.model.cbm_exn import cbm_exn_step
from libcbm.model import model_definition


class CBMEXNModel:
    def __init__(
        self,
        model_handle: model_definition.ModelHandle,
        pool_config: list[str],
        flux_config: list[dict],
        model_parameters: dict,
        spinup_func: Callable[
            ["CBMEXNModel", SpinupInput], CBMVariables
        ] = None,
        step_func: Callable[
            ["CBMEXNModel", CBMVariables], CBMVariables
        ] = None,
    ):
        self._model_handle = model_handle
        self._pool_config = pool_config
        self._flux_config = flux_config
        self._spinup_func = (
            cbm_exn_spinup.spinup if not spinup_func else spinup_func
        )
        self._step_func = cbm_exn_step.step if not step_func else step_func
        self._parameters = model_parameters

    @property
    def parameters(self) -> dict:
        return self._parameters

    def spinup(self, spinup_input: SpinupInput) -> CBMVariables:
        return self._spinup_func(spinup_input)

    def step(self, cbm_vars: CBMVariables) -> CBMVariables:
        return self._step_func(self, cbm_vars)

    def create_operation(self, matrices: list, fmt: str):
        return self._model_handle.create_operation(matrices, fmt)


@contextmanager
def initialize(
    pool_config: list[str],
    flux_config: list[dict],
    spinup_func: Callable[[CBMEXNModel, SpinupInput], CBMVariables] = None,
    step_func: Callable[[CBMEXNModel, CBMVariables], CBMVariables] = None,
) -> Iterator[CBMEXNModel]:
    """_summary_

    Args:
        pool_config (list[str]): _description_
        flux_config (list[dict]): _description_
        spinup_func (Callable[[CBMEXNModel, SpinupInput], CBMVariables], optional): _description_. Defaults to None.
        step_func (Callable[[CBMEXNModel, CBMVariables], CBMVariables], optional): _description_. Defaults to None.

    Yields:
        Iterator[CBMEXNModel]: _description_
    """
    pools = None
    flux = None
    with model_definition.create_model(pools, flux) as model_handle:
        yield CBMEXNModel(
            model_handle, pool_config, flux_config, spinup_func, step_func
        )

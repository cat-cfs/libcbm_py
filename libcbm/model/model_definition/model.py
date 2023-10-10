from __future__ import annotations
from typing import Iterator
from contextlib import contextmanager
import numpy as np
from libcbm.model.model_definition.model_variables import ModelVariables
from libcbm.model.model_definition import model_handle
from libcbm.model.model_definition.model_handle import ModelHandle
from libcbm.model.model_definition.model_matrix_ops import ModelMatrixOps
from libcbm.wrapper.libcbm_operation import Operation


class CBMModel:
    """
    An abstraction of a Carbon budget model
    """

    def __init__(
        self,
        model_handle: ModelHandle,
        pool_config: list[str],
        flux_config: list[dict],
        op_processes: dict[str, int],
    ):
        invalid_pool_names = []
        for p in pool_config:
            if not p.isidentifier():
                invalid_pool_names.append(p)
        if invalid_pool_names:
            raise ValueError(
                f"pools names are not valid identifiers {invalid_pool_names}"
            )
        self._model_handle = model_handle
        self._pool_config = pool_config
        self._flux_config = flux_config
        self._model_matrix_ops = ModelMatrixOps(
            self._model_handle, pool_config, op_processes
        )

    @property
    def pool_names(self) -> list[str]:
        return self._pool_config.copy()

    @property
    def flux_names(self) -> list[str]:
        return [f["name"] for f in self._flux_config]

    @property
    def matrix_ops(self) -> ModelMatrixOps:
        return self._model_matrix_ops

    def compute(
        self,
        cbm_vars: ModelVariables,
        operations: list[Operation],
    ):
        """Compute a batch of C dynamics

        Args:
            cbm_vars (ModelVariables): cbm variables and state
            operations (list[Operation]): a list of Operation objects as
                allocated by `create_operation`
            op_process_ids (list[int]): list of integers
        """

        self._model_handle.compute(
            cbm_vars["pools"],
            cbm_vars["flux"] if "flux" in cbm_vars else None,
            cbm_vars["state"]["enabled"],
            operations,
        )


@contextmanager
def initialize(
    pool_config: list[str],
    flux_config: list[dict],
) -> Iterator[CBMModel]:
    """Initialize a CBMModel for spinup or stepping

    Args:
        pool_config (list[str]): list of string pool identifiers.
        flux_config (list[dict]): list of flux indicator dictionary
            structures.

    Example Pools::

        ["Input", "Merchantable", "OtherC"]

    Example Flux indicators::

        [
            {
                "name": "NPP",
                "process": "Growth",
                "source_pools": [
                    "Input",
                ],
                "sink_pools": [
                    "Merchantable",
                    "Foliage",
                    "Other",
                    "FineRoot",
                    "CoarseRoot"
                ]
            },
            {
                "name": "DOMEmissions",
                "process": "Decay",
                "source_pools": [
                    "AboveGroundVeryFast",
                    "BelowGroundVeryFast",
                    "AboveGroundFast",
                    "BelowGroundFast",
                    "MediumSoil",
                    "AboveGroundSlow",
                    "BelowGroundSlow",
                    "StemSnag",
                    "BranchSnag",
                ],
                "sink_pools": [
                    "CO2"
                ]
            }
        ]

    Yields:
        Iterator[CBMModel]: instance of CBMModel
    """
    pools = {p: i for i, p in enumerate(pool_config)}
    flux = [
        {
            "id": f_idx + 1,
            "index": f_idx,
            "process_id": int(f["process"]),
            "source_pools": [pools[x] for x in f["source_pools"]],
            "sink_pools": [pools[x] for x in f["sink_pools"]],
        }
        for f_idx, f in enumerate(flux_config)
    ]
    with model_handle.create_model_handle(pools, flux) as _model_handle:
        yield CBMModel(
            _model_handle,
            pool_config,
            flux_config,
        )

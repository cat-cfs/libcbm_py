from __future__ import annotations
import json
import numpy as np
from typing import Iterator
from contextlib import contextmanager
from libcbm.wrapper import libcbm_operation
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm import resources
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.series import Series


class ModelHandle:
    """
    Class to facilitate optimizied pool flux operations.

    This is essentially a layer for simplifying the python
    interface to libcbm matrix processing
    """

    def __init__(
        self,
        wrapper: LibCBMWrapper,
        pools: dict[str, int],
        flux_indicators: list[dict],
    ):
        """Initialize ModelHandle

        Args:
            wrapper (LibCBMWrapper): low level function wrapper
            pools (dict[str, int]): the collection of named pools
            flux_indicators (list[dict]): flux indicator configuration
        """
        self.wrapper = wrapper
        self.pools = pools
        self.flux_indicators = flux_indicators

    def _matrix_rc(
        self,
        value: list,
        process_id: int,
        matrix_index: np.ndarray,
        init_value: int,
    ) -> libcbm_operation.Operation:
        return libcbm_operation.Operation(
            self.wrapper,
            libcbm_operation.OperationFormat.RepeatingCoordinates,
            value,
            process_id,
            matrix_index,
            init_value,
        )

    def _matrix_list(
        self,
        value: list,
        process_id: int,
        matrix_index: np.ndarray,
        init_value: int,
    ) -> libcbm_operation.Operation:
        return libcbm_operation.Operation(
            self.wrapper,
            libcbm_operation.OperationFormat.MatrixList,
            value,
            process_id,
            matrix_index,
            init_value,
        )

    def create_operation(
        self,
        matrices: list,
        fmt: str,
        process_id: int,
        matrix_index: np.ndarray,
        init_value: int = 1,
    ) -> libcbm_operation.Operation:
        """Create a libcbm Operation

        `repeating_coordinates` description:

            can be used to repeat the same matrix
            coordinates across multiple stands, with varying matrix values.

            The lists of are of type [str, str, np.ndaray]

            The length of each array is the number of stands

            example `repeating_coordinates`::

                [
                    [pool_a, pool_b, [flow_ab_0, flow_ab_1, ... flow_ab_N]],
                    [pool_c, pool_a, [flow_ca_0, flow_ca_1, ... flow_ca_N]],
                    ...
                ]

        `matrix_list` description:

            Used for cases when coordinates vary for matrices.  This is a list
            of matrices in sparse coordinate format (COO)

            example `matrix_list`::

                [
                    [
                        [pool_a, pool_b, flow_ab],
                        [pool_a, pool_c, flow_ac],
                        ...
                    ],
                    [
                        [pool_b, pool_b, flow_bb],
                        [pool_a, pool_c, flow_ac],
                        ...
                    ],
                    ...
                ]

        Args:
            matrices (list): a list of matrix values.  The required format is
                dependant on the `fmt` parameter.
            fmt (str): matrix value format.  Can be either of:
                "repeating_coordinates" or "matrix_list"
            process_id (int): flux tracking category id.  Fluxes associated
                with this Operation will fall under this category.
            matrix_index (np.ndarray): an array whose length is the same as
                the number of stands being simulated.  Each value in the array
                is the index to one of the matrices defined in the
                `matrix_list` parameter.
            init_value (int, optional): The default value set on the diagonal
                of each matrix. Diagonal values specified in the `matrix_list`
                parameters will overwrite this default. Defaults to 1.

        Raises:
            ValueError: an unknown value for `fmt` was specified.

        Returns:
            libcbm_operation.Operation: initialized Operation object
        """
        if fmt == "repeating_coordinates":
            pool_id_mat = [
                [self.pools[row[0]], self.pools[row[1]], row[2]]
                for row in matrices
            ]
            return self._matrix_rc(
                pool_id_mat, process_id, matrix_index, init_value
            )
        elif fmt == "matrix_list":
            mat_list = []
            for mat in matrices:
                mat_len = len(mat)
                np_mat = np.zeros(shape=(mat_len, 3))
                for i_entry, entry in enumerate(mat):
                    np_mat[i_entry, 0] = self.pools[entry[0]]
                    np_mat[i_entry, 1] = self.pools[entry[1]]
                    np_mat[i_entry, 2] = entry[2]
                mat_list.append(np_mat)
            return self._matrix_list(
                mat_list, process_id, matrix_index, init_value
            )
        else:
            raise ValueError("unknown format")

    def compute(
        self,
        pools: DataFrame,
        flux: DataFrame,
        enabled: Series,
        operations: list[libcbm_operation.Operation],
    ) -> None:
        """compute a batch of Operations

        Args:
            pools (DataFrame): the pools dataframe, which is updated
                by this function
            flux (DataFrame): the flux dataframe, which is assigned
                specific pool flows
            enabled (Series): a boolean series indicating that particular
                stands are subject to the batch of operations (when 1) or
                not (when 0).  If zero, the corresponding pool and flux
                values will not be modified.
            operations (list[libcbm_operation.Operation]): the list of
                Operations.
        """
        libcbm_operation.compute(
            dll=self.wrapper,
            pools=pools,
            operations=operations,
            op_processes=[o.op_process_id for o in operations],
            flux=flux,
            enabled=enabled,
        )


@contextmanager
def create_model_handle(
    pools: dict[str, int], flux_indicators: list[dict]
) -> Iterator[ModelHandle]:
    """initialize a :py:class:`ModelHandle` object.

    Args:
        pools (dict[str, int]): pool definition
        flux_indicators (list[dict]): flux indicator configuration

    Yields:
        Iterator[ModelHandle]: the initialized Modelhandle
    """
    libcbm_config = {
        "pools": [
            {"name": p, "id": p_idx, "index": p_idx}
            for p, p_idx in pools.items()
        ],
        "flux_indicators": [
            {
                "id": f_idx + 1,
                "index": f_idx,
                "process_id": f["process_id"],
                "source_pools": [int(x) for x in f["source_pools"]],
                "sink_pools": [int(x) for x in f["sink_pools"]],
            }
            for f_idx, f in enumerate(flux_indicators)
        ],
    }

    with LibCBMHandle(
        resources.get_libcbm_bin_path(), json.dumps(libcbm_config)
    ) as handle:
        yield ModelHandle(LibCBMWrapper(handle), pools, flux_indicators)

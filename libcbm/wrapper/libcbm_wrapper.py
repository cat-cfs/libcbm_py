# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes
import numpy as np
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix_Int
from libcbm.wrapper import libcbm_wrapper_functions
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm.storage.dataframe import DataFrame
from libcbm.storage.dataframe import Series


class LibCBMWrapper:
    """Exposes low level ctypes wrapper to regular python, for the core
    libcbm functions.

        Args (LibCBMHandle): handle
            for the underlying dll/so compiled library
    """

    def __init__(self, handle: LibCBMHandle):
        self.handle = handle

    def allocate_op(self, size: int) -> int:
        """Allocates storage for matrices, returning an id for the
        allocated block.

        Args:
            size (int): The number of elements in the allocated matrix block
                index, which corresponds to the number of stands that can
                be processed with this matrix block

        Raises:
            AssertionError: raised if the Initialize method was not called
                prior to this method.
            RuntimeError: if an error is detected in libCBM, it will be
                re-raised with an appropriate error message.

        Returns:
            int: the id for an allocated block of matrices
        """
        op_id = self.handle.call("LibCBM_Allocate_Op", size)
        return op_id

    def free_op(self, op_id: int):
        """Deallocates a matrix block that was allocated by the allocate_op
        method.

        Args:
            op_id (int): The id for an allocated block of matrices.

        """
        self.handle.call("LibCBM_Free_Op", op_id)

    def set_op(
        self,
        op_id: int,
        matrices: list[np.ndarray],
        matrix_index: list[int],
        init: int = 0,
    ):
        """Assigns values to an allocated block of matrices.

            Example::

                n_stands = 3
                op_id = allocate_op(n_stands)
                matrix_0 = np.array([
                    [0, 1, 0.5],
                    [0, 0, 1.0],
                    [1, 1, 1.0]
                ])
                matrix_1 = np.array([
                    [1, 0, 0.5],
                    [0, 0, 1.0],
                    [1, 1, 0.5]
                ])
                matrices = [matrix_0, matrix_1]
                matrix_index = [0,1,0]
                SetOp(op_id, matrices, matrix_index)

            In this example, a pair of matrices are passed. Here the matrices
            are of dimension N by N where N is defined by the call to
            :py:func:`Initialize`.

            matrix_0:

            ===  ===  ===  ===  ===
             p   p0   p1   ...  pN
            ===  ===  ===  ===  ===
            p0   1.0  0.5  0.0  0.0
            p1   0.0  1.0  0.0  0.0
            ...  0.0  0.0  0.0  0.0
            pN   0.0  0.0  0.0  0.0
            ===  ===  ===  ===  ===

            matrix_1:

            ===  ===  ===  ===  ===
             p   p0   p1   ...  pN
            ===  ===  ===  ===  ===
            p0   1.0  0.0  0.0  0.0
            p1   0.5  0.5  0.0  0.0
            ...  0.0  0.0  0.0  0.0
            pN   0.0  0.0  0.0  0.0
            ===  ===  ===  ===  ===

            The matrices are indexed according to the following table:

            ===========  ============
            Stand_index  Matrix_index
            ===========  ============
                 0           0
                 1           1
                 2           0
            ===========  ============

            related functions: :py:func:`allocate_op`,
                :py:func:`compute_pools`, :py:func:`compute_flux`

        Args:
            op_id (int): The id for an allocated block of matrices
            matrices (list): a list of n by 3 ndarray matrices which are
                coordinate format triplet values (row,column,value).  All
                defined row/column combinations are set with the value, and
                all other matrix cells are assumed to be 0.
            matrix_index (ndarray): an array of length n stands where the
                value is an index to a matrix in the specified list of matrices
                provided to this function.
            init (int): if set to 0 matrices are initialized with zeros, and
                if 1 the matrix diagonals are initialized to 1 (identity) prior
                to assigning matrix values.  Other values will result in an
                error.

        """
        matrices_p = libcbm_wrapper_functions.get_matrix_list_pointer(matrices)
        self.handle.call(
            "LibCBM_SetOp",
            op_id,
            matrices_p,
            len(matrices),
            matrix_index,
            matrix_index.shape[0],
            init,
        )

    def set_op_repeating(
        self,
        op_id: int,
        coordinates: np.ndarray,
        values: np.ndarray,
        matrix_index: np.ndarray,
        init: int = 0,
    ):
        """Assigns the specified values associated with repeating coordinates
        to an allocated block of matrices.

        Args:
            op_id (int): The id for an allocated block of matrices
            coordinates (numpy.ndarray): matrix of integer coordinates
                corresponding to each column of the values.
                Shape (n_coordinate, 2)
            values (numpy.ndarray): matrix of float values for each matrix to
                assign. Shape (n_matrices, n_coordinate).
            matrix_index (ndarray): an array of length n stands where the
                value is an index to a row in the specifies values matrix
            init (int): if set to 0 matrices are initialized with zeros, and
                if 1 the matrix diagonals are initialized to 1 (identity) prior
                to assigning matrix values.  Other values will result in an
                error.
        """
        self.handle.call(
            "LibCBM_SetOp2",
            op_id,
            LibCBM_Matrix_Int(coordinates),
            LibCBM_Matrix(values),
            matrix_index,
            matrix_index.shape[0],
            init,
        )

    def compute_pools(
        self, ops: np.ndarray, pools: DataFrame, enabled: Series = None
    ):
        """Computes flows between pool values for all stands.

        Each value in the ops parameter is an id to a matrix block, and is also
        conceptually a list of matrixes of length n stands.

        Performs the following computation::

            for op in ops:
                for s in len(n_stands):
                    M = op.get_matrix(s)
                    pools[s,:] = np.matmul(pools[s,:], M)

        Where get_matrix is pseudocode for an internal function returning the
        matrix for the op, stand index combination.

        Args:
            ops (ndarray): list of matrix block ids as allocated by the
                :py:func:`allocate_op` function.
            pools (DataFrame): matrix of shape
                n_stands by n_pools. The values in this matrix are updated by
                this function.
            enabled (Series): optional int vector of length
                n_stands. If specified, enables or disables flows for each
                stand, based on the value at each stand index. A value of 0
                indicates a disabled stand index, and any other value is an
                enabled stand index. If None, all flows are assumed to be
                enabled. Defaults to None.

        """
        n_ops = len(ops)
        nd_pools = pools.to_c_contiguous_numpy_array()
        pool_mat = LibCBM_Matrix(nd_pools)
        ops_p = ctypes.cast(
            (ctypes.c_size_t * n_ops)(*ops), ctypes.POINTER(ctypes.c_size_t)
        )
        self.handle.call(
            "LibCBM_ComputePools",
            ops_p,
            n_ops,
            pool_mat,
            None if enabled is None else enabled.to_numpy(),
        )

    def compute_flux(
        self,
        ops: np.ndarray,
        op_processes: np.ndarray,
        pools: DataFrame,
        flux: DataFrame,
        enabled: Series = None,
    ):
        """Computes and tracks flows between pool values for all stands.

        Performs the same operation as compute_pools, except that the fluxes
        are tracked in the specified flux parameter, according to the
        flux_indicators configuration passed to the LibCBM initialize method.

        Args:
            ops (ndarray): list of matrix block ids as allocated by the
                allocate_op function.
            op_processes (ndarray): list of integers of length n_ops.
                Ids referencing flux indicator process_id definition in the
                Initialize method.
            pools (numpy.ndarray or pandas.DataFrame): matrix of shape
                n_stands by n_pools. The values in this matrix are updated by
                this function.
            flux (ndarray or pandas.DataFrame): matrix of shape n_stands
                by n_flux_indicators. The values in this matrix are updated
                by this function according to the definition of flux
                indicators in the configuration and the flows that occur in
                the specified operations.
            enabled (ndarray, optional): optional int vector of length
                n_stands. If specified, enables or disables flows for each
                stand, based on the value at each stand index. A value of 0
                indicates a disabled stand index, and any other value is an
                enabled stand index. If None, all flows are assumed to be
                enabled. Defaults to None.

        Raises:
            ValueError: raised when parameters passed to this function are not
                valid.
        """
        if not self.handle:
            raise AssertionError("dll not initialized")

        n_ops = len(ops)
        if len(op_processes) != n_ops:
            raise ValueError("ops and op_processes must be of equal length")
        nd_pools = pools.to_c_contiguous_numpy_array()
        pools_mat = LibCBM_Matrix(nd_pools)

        nd_flux = flux.to_c_contiguous_numpy_array()
        flux_mat = LibCBM_Matrix(nd_flux)

        ops_p = ctypes.cast(
            (ctypes.c_size_t * n_ops)(*ops), ctypes.POINTER(ctypes.c_size_t)
        )
        op_process_p = ctypes.cast(
            (ctypes.c_size_t * n_ops)(*op_processes),
            ctypes.POINTER(ctypes.c_size_t),
        )

        self.handle.call(
            "LibCBM_ComputeFlux",
            ops_p,
            op_process_p,
            n_ops,
            pools_mat,
            flux_mat,
            None if enabled is None else enabled.to_numpy(),
        )

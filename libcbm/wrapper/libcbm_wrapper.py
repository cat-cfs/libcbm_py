import ctypes
from libcbm.wrapper.libcbm_matrix import LibCBM_Matrix
from libcbm import data_helpers


class LibCBMWrapper():
    """Exposes low level ctypes wrapper to regular python, for the core
    libcbm functions.

        Args (:py:class:`libcbm.wrapper.libcbm_handle.LibCBMHandle`): handle
            for the underlying dll/so compiled library
    """
    def __init__(self, handle):
        self.handle = handle

    def allocate_op(self, n):
        """Allocates storage for matrices, returning an id for the
        allocated block.

        Args:
            n (int): The number of elements in the allocated matrix block
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
        op_id = self.handle.call("LibCBM_Allocate_Op", n)
        return op_id

    def free_op(self, op_id):
        """Deallocates a matrix block that was allocated by the allocate_op method.

        Args:
            op_id (int): The id for an allocated block of matrices.

        """
        self.handle.call("LibCBM_Free_Op", op_id)

    def set_op(self, op_id, matrices, matrix_index):
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

            related functions: :py:func:`allocate_op`, :py:func:`compute_pools`,
                :py:func:`compute_flux`

        Args:
            op_id (int): The id for an allocated block of matrices
            matrices (list): a list of n by 3 ndarray matrices which are
                coordinate format triplet values (row,column,value).  All
                defined row/column combinations are set with the value, and
                all other matrix cells are assumed to be 0.
            matrix_index (ndarray): an array of length n stands where the
                value is an index to a matrix in the specified list of matrices
                provided to this function.

        """
        matrices_array = (LibCBM_Matrix * len(matrices))()
        for i, x in enumerate(matrices):
            matrices_array[i] = LibCBM_Matrix(x)
        matrices_p = ctypes.cast(matrices_array, ctypes.POINTER(LibCBM_Matrix))
        self.handle.call(
            "LibCBM_SetOp", op_id, matrices_p, len(matrices), matrix_index,
            matrix_index.shape[0])

    def compute_pools(self, ops, pools, enabled=None):
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
            pools (numpy.ndarray or pandas.DataFrame): matrix of shape
                n_stands by n_pools. The values in this matrix are updated by
                this function.
            enabled ([type], optional): optional int vector of length
                n_stands. If specified, enables or disables flows for each
                stand, based on the value at each stand index. A value of 0
                indicates a disabled stand index, and any other value is an
                enabled stand index. If None, all flows are assumed to be
                enabled. Defaults to None.

        """
        n_ops = len(ops)
        p = data_helpers.get_ndarray(pools)
        poolMat = LibCBM_Matrix(p)
        ops_p = ctypes.cast(
            (ctypes.c_size_t*n_ops)(*ops), ctypes.POINTER(ctypes.c_size_t))

        self.handle.call(
            "LibCBM_ComputePools", ops_p, n_ops, poolMat,
            data_helpers.get_nullable_ndarray(enabled, type=ctypes.c_int))

    def compute_flux(self, ops, op_processes, pools, flux, enabled=None):
        """Computes and tracks flows between pool values for all stands.

        Performs the same operation as ComputePools, except that the fluxes
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
        p = data_helpers.get_ndarray(pools)
        poolMat = LibCBM_Matrix(p)

        f = data_helpers.get_ndarray(flux)
        fluxMat = LibCBM_Matrix(f)

        ops_p = ctypes.cast(
            (ctypes.c_size_t*n_ops)(*ops), ctypes.POINTER(ctypes.c_size_t))
        op_process_p = ctypes.cast(
            (ctypes.c_size_t*n_ops)(*op_processes),
            ctypes.POINTER(ctypes.c_size_t))
        enabled = data_helpers.get_nullable_ndarray(enabled, type=ctypes.c_int)

        self.handle.call(
            "LibCBM_ComputeFlux", ops_p, op_process_p, n_ops, poolMat,
            fluxMat, enabled)

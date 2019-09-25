import json
import numpy as np
import scipy.sparse

from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm import resources


def load_dll(config):

    dll = LibCBMWrapper(
        LibCBMHandle(resources.get_libcbm_bin_path(), json.dumps(config)))
    return dll


def create_pools(names):
    """creates pool configuration based on the specific list of names

    Args:
        names (list): a list of pool names

    Returns:
        list: a configuration for the libcbm dll/so
    """
    return [
        {'name': x, 'id': i+1, 'index': i}
        for i, x in enumerate(names)]


def create_pools_by_name(pools):
    return {x["name"]: x for x in pools}


def to_coordinate(matrix):
    """convert the specified matrix to a matrix of coordinate triples.
    This is needed since libcbm deals with sparse matrices.

    Args:
        matrix (numpy.ndarray): [description]

    Returns:
        numpy.ndarray: a n by 3 matrix where n is the number of non-zero values
            in the input. Has columns for row, col, data based on the input
            data.

    Example

        Input::

            np.array([[1,2],
                      [3,4]])

        Result::

            [[0,0,1],
             [0,1,2],
             [1,0,3],
             [1,1,4]]

    """
    coo = scipy.sparse.coo_matrix(matrix)
    return np.column_stack((coo.row, coo.col, coo.data))


def ComputePools(pools, ops, op_indices):
    """Runs the ComputePools libCBM function based on the specified numpy pool
    matrix, and the specified matrix ops.

    Args:
        pools (numpy.ndarray): a matrix of pool values of dimension n_stands by
            n_pools
        ops (list): list of list of numpy matrices, the major dimension is
            n_ops, and the minor dimension may be jagged.  Each matrix is of
            dimension n_pools by n_pools.
        op_indices (numpy.ndarray): An n_stands by n_ops matrix, where each
            column is a vector of indices to the jagged minor dimension of the
            ops parameter.

    Returns:
        numpy.ndarray: the result of the ComputePools libcbm operation
    """
    pools = pools.copy()
    pooldef = create_pools([str(x) for x in range(pools.shape[1])])
    dll = load_dll({
        "pools": pooldef,
        "flux_indicators": []
    })
    op_ids = []
    for i, op in enumerate(ops):
        op_id = dll.AllocateOp(pools.shape[0])
        op_ids.append(op_id)
        # The set op function accepts a matrix of coordinate triples.
        # In LibCBM matrices are stored in a sparse format, so 0 values can be
        # omitted from the parameter.
        dll.SetOp(op_id, [to_coordinate(x) for x in op],
                  np.ascontiguousarray(op_indices[:, i]))

    dll.ComputePools(op_ids, pools)

    return pools

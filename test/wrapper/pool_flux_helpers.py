import json
import numpy as np
import scipy.sparse

from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
from libcbm import resources
from libcbm.storage import dataframe


def load_dll(config):

    dll = LibCBMWrapper(
        LibCBMHandle(resources.get_libcbm_bin_path(), json.dumps(config))
    )
    return dll


def create_pools(names):
    """creates pool configuration based on the specific list of names

    Args:
        names (list): a list of pool names

    Returns:
        list: a configuration for the libcbm dll/so
    """
    return [{"name": x, "id": i + 1, "index": i} for i, x in enumerate(names)]


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


def compute_pools(pools, ops, op_indices):
    """Runs the compute_pools libCBM function based on the specified numpy pool
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
        numpy.ndarray: the result of the compute_pools libcbm operation
    """
    pools = pools.copy()
    pooldef = create_pools([str(x) for x in range(pools.shape[1])])
    dll = load_dll({"pools": pooldef, "flux_indicators": []})
    op_ids = []
    for i, op in enumerate(ops):
        op_id = dll.allocate_op(pools.shape[0])
        op_ids.append(op_id)
        # The set op function accepts a matrix of coordinate triples.
        # In LibCBM matrices are stored in a sparse format, so 0 values can be
        # omitted from the parameter.
        dll.set_op(
            op_id,
            [to_coordinate(x) for x in op],
            np.ascontiguousarray(op_indices[:, i]),
        )
    pools_df = dataframe.from_numpy(
        {str(x): pools[:, x] for x in range(pools.shape[1])}
    )
    dll.compute_pools(op_ids, pools_df)

    return pools_df.to_c_contiguous_numpy_array()


def create_flux_indicator(pools_by_name, process_id, sources, sinks):
    """helper method to create configuration for dll"""

    return {
        "id": None,
        "index": None,
        "process_id": process_id,
        "source_pools": [pools_by_name[x]["id"] for x in sources],
        "sink_pools": [pools_by_name[x]["id"] for x in sinks],
    }


def append_flux_indicator(collection, flux_indicator):
    """helper method to create configuration for dll"""
    flux_indicator["index"] = len(collection)
    flux_indicator["id"] = len(collection) + 1
    collection.append(flux_indicator)


def compute_flux(
    pools: np.ndarray,
    poolnames: list,
    mats: list[np.ndarray],
    op_indices: np.ndarray,
    op_processes: list,
    flux_indicators: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the libcbm compute_flux method for testing purposes

    Args:
        pools (numpy.ndarray): a n_stands, by n_pools matrix of pool values
        poolnames (list): string labels for each of the columns in pools
        mats (list): a nested list of flow matrices (numpy.ndarrays)
        op_indices (numpy.ndarray): An n_stands by n_ops matrix, where each
            column is a vector of indices to the jagged minor dimension of the
            ops parameter.
        op_processes (list): a list of integers used to filter flux indicators
            by the process defined in flux_indicator config.
        flux_indicators (list): a list of dictionaries which define flux
            indicator configuration

    Returns:
        tuple: 1. the pool result (numpy.ndarray) and 2. the flux result
            (numpy.ndarray) of the compute_flux libcbm method.
    """
    pools = pools.copy()
    flux = np.zeros((pools.shape[0], len(flux_indicators)))
    pooldef = create_pools([poolnames[x] for x in range(pools.shape[1])])
    pools_by_name = create_pools_by_name(pooldef)
    fi_collection = []
    for flux_indicator in flux_indicators:
        flux_indicator_config = create_flux_indicator(
            pools_by_name,
            flux_indicator["process_id"],
            flux_indicator["sources"],
            flux_indicator["sinks"],
        )
        append_flux_indicator(fi_collection, flux_indicator_config)
    dll = load_dll({"pools": pooldef, "flux_indicators": fi_collection})
    op_ids = []
    for i, matrix in enumerate(mats):
        op_id = dll.allocate_op(pools.shape[0])
        op_ids.append(op_id)
        dll.set_op(
            op_id,
            [to_coordinate(x) for x in matrix],
            np.ascontiguousarray(op_indices[:, i]),
        )

    pools_df = dataframe.from_numpy(
        {name: pools[:, idx] for idx, name in enumerate(poolnames)}
    )

    flux_df = dataframe.from_numpy(
        {f"flux{idx}": flux[:, idx] for idx, _ in enumerate(flux_indicators)}
    )
    dll.compute_flux(op_ids, op_processes, pools_df, flux_df)
    return (
        pools_df.to_c_contiguous_numpy_array(),
        flux_df.to_c_contiguous_numpy_array(),
    )

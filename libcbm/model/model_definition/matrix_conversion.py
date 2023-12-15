import pandas as pd
import numpy as np
from scipy import sparse


def filter_pools(
    pool_idx: dict[str, int],
    op_data: pd.DataFrame,
) -> pd.DataFrame:
    """Retain only those columns from a matrix-format
    dataframe where the pools present in the specified
    pool_idx parameter are present.

    Example::

        result = matrix_operations.filter_pools(
            {
                "A": 1,
                "B": 2,

            },
            op_data=pd.DataFrame(
                columns=[
                    "A.A",
                    "A.B",
                    "A.C",
                    "B.A",
                    "B.B",
                    "B.C",
                    "C.A",
                    "C.B",
                    "C.C",
                ],
                data=[[1]*9]*5
            )
        )

        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame(
                columns=["A.A", "A.B", "B.A", "B.B"],
                data=[[1]*4]*5
            )
        )


    Args:
        pool_idx (dict[str, int]): dictionary of pool names, pool index
        op_data (pd.DataFrame): dataframe where each column represents a
            src.sink coordinate in a sparse matrix and each row represents
            a 2d matrix.

    Returns:
        pd.DataFrame: a copy of the original dataframe with filtered columns
    """
    included_col_idx = list()
    for i_col, col in enumerate(op_data.columns):
        source, sink = col.split(".")
        if source in pool_idx and sink in pool_idx:
            included_col_idx.append(i_col)
    return op_data.iloc[:, included_col_idx].copy()


def to_coo_matrix(
    pool_idx: dict[str, int],
    op_data: pd.DataFrame,
) -> sparse.coo_matrix:
    """create a scipy.sparse.coomatrix like `scipy.sparse.block_diag`
    using a matrix-formatted pandas dataframe

    Example::

        result = matrix_operations.to_coo_matrix(
            {
                "A": 0,
                "B": 1,
            },
            op_data=pd.DataFrame(
                columns=[
                    "A.A",
                    "A.B",
                    "B.A",
                    "B.B",
                ],
                data=[[1] * 4] * 3,
            ),
        )

        assert (
            result.toarray() == [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
            ]
        ).all()

    Args:
        pool_idx (dict[str, int]): the named-enumerated collection of pools
        op_data (pd.DataFrame): dataframe where each column represents a
            src.sink coordinate in a sparse matrix and each row represents
            a 2d matrix.

    Returns:
        sparse.coo_matrix: the block_diag-like result
    """
    n_pools = len(pool_idx)
    n_rows = len(op_data.index)
    diag_coords = set(zip(range(n_pools), range(n_pools)))
    nonzero_coords = {}
    for col in op_data.columns:
        source, sink = col.split(".")
        row_idx = pool_idx[source]
        col_idx = pool_idx[sink]
        nonzero_coords[(row_idx, col_idx)] = op_data[col].to_numpy()

    all_coords = diag_coords.union(nonzero_coords.keys())

    n_coords = len(all_coords)

    out_rows = np.zeros(n_coords * n_rows, dtype="int")
    out_cols = np.zeros(n_coords * n_rows, dtype="int")
    out_values = np.zeros(n_coords * n_rows, dtype="float")

    for idx, (row_idx, col_idx) in enumerate(all_coords):
        if (row_idx, col_idx) in nonzero_coords:
            value = nonzero_coords[(row_idx, col_idx)]
        else:
            value = 1.0
        out_rows[idx::n_coords] = row_idx + np.arange(
            0, n_pools * n_rows, n_pools
        )
        out_cols[idx::n_coords] = col_idx + np.arange(
            0, n_pools * n_rows, n_pools
        )
        out_values[idx::n_coords] = value
    return sparse.coo_matrix((out_values, (out_rows, out_cols)))

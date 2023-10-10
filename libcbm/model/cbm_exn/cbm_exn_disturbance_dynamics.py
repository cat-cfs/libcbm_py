import numpy as np
import pandas as pd


def disturbance(
    pools: list[str],
    disturbance_matrices: pd.DataFrame,
    dm_associations: pd.DataFrame,
) -> pd.DataFrame:
    matrix_data_by_dmid: dict[int, list[list]] = {}
    dmid = disturbance_matrices["disturbance_matrix_id"].to_numpy()
    source = disturbance_matrices["source_pool"].to_list()
    sink = disturbance_matrices["sink_pool"].to_list()
    proportion = disturbance_matrices["proportion"].to_numpy()
    for i in range(dmid.shape[0]):
        dmid_i = dmid[i]
        if dmid_i not in matrix_data_by_dmid:
            matrix_data_by_dmid[dmid_i] = []
        matrix_data_by_dmid[dmid_i].append([source[i], sink[i], proportion[i]])
    dmids = list(matrix_data_by_dmid.keys())
    for dmid in dmids:
        pool_set = set(pools)
        for row in matrix_data_by_dmid[dmid]:
            pool_set.discard(row[0])
        for pool in pool_set:
            matrix_data_by_dmid[dmid].append([pool, pool, 1.0])

    flow_cols = {"dmid": np.full(len(matrix_data_by_dmid), 0, "int64")}
    for i_dmid, (dmid, matrix_list) in enumerate(matrix_data_by_dmid.items()):
        flow_cols["dmid"][i_dmid] = dmid
        for row in matrix_list:
            flow_colname = f"{row[0]}.{row[1]}"
            if flow_colname not in flow_cols:
                flow_cols[flow_colname] = np.full(
                    len(matrix_data_by_dmid), 0, "float"
                )
            flow_cols[flow_colname][i_dmid] = row[2]
    dm_flows = pd.DataFrame(flow_cols)
    dm_flows = dm_flows.reindex(sorted(dm_flows.columns), axis=1)
    dm_associations = dm_associations.rename(
        columns={
            "spatial_unit_id": "[state.spatial_unit_id]",
            "disturbance_type_id": "[parameters.disturbance_type_id]",
            "sw_hw": "[state.sw_hw]",
        }
    )
    output = dm_associations.merge(
        dm_flows, left_on="disturbance_matrix_id", right_on="dmid", how="left"
    )
    output = output.drop(columns=["disturbance_matrix_id", "dmid"])
    return output
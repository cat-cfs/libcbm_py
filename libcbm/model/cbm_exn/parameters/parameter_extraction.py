import os
import json
import pandas as pd
import sqlite3
import argparse


def query(db_path, query):
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, con)
    return df


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4)


def _flux_indicator_config(output_dir: str, db_path: str):
    cbm3_pools = query(db_path, "select * from pool")
    cbm3_flux_indicator = query(db_path, "select * from flux_indicator")
    cbm3_flux_indicator_sink = query(
        db_path, "select * from flux_indicator_sink"
    )
    cbm3_flux_indicator_source = query(
        db_path, "select * from flux_indicator_source"
    )
    fi_sink_merged = cbm3_flux_indicator_sink.merge(
        cbm3_pools, left_on="pool_id", right_on="id"
    )
    fi_source_merged = cbm3_flux_indicator_source.merge(
        cbm3_pools, left_on="pool_id", right_on="id"
    )
    cbm3_flux_indicators = []
    for i, fi_row in cbm3_flux_indicator.iterrows():
        fi_id = fi_row["id"]
        sink_rows = fi_sink_merged[
            fi_sink_merged["flux_indicator_id"] == fi_id
        ]
        source_rows = fi_source_merged[
            fi_source_merged["flux_indicator_id"] == fi_id
        ]
        cbm3_flux_indicators.append(
            {
                "name": str(fi_row["name"]),
                "process": fi_row["flux_process_id"],
                "source_pools": [str(c) for c in source_rows["code"]],
                "sink_pools": [str(c) for c in sink_rows["code"]],
            }
        )
    cbm_exn_flux_indicator = cbm3_flux_indicators.copy()

    for item in cbm_exn_flux_indicator:
        updated_sources = []
        for p in item["source_pools"]:
            updated_sources.append(
                p.replace("Hardwood", "").replace("Softwood", "")
            )

        item["source_pools"] = list(dict.fromkeys(updated_sources))

        updated_sinks = []
        for p in item["sink_pools"]:
            updated_sinks.append(
                p.replace("Hardwood", "").replace("Softwood", "")
            )

        item["sink_pools"] = list(dict.fromkeys(updated_sinks))

        cbm_exn_flux_indicator_filtered = []
        for item in cbm_exn_flux_indicator:
            if item["name"] == "DisturbanceSoftProduction":
                item["name"] = "DisturbanceProduction"
                cbm_exn_flux_indicator_filtered.append(item)
            elif item["name"] == "DecaySWStemSnagToAir":
                item["name"] = "DecayStemSnagToAir"
                cbm_exn_flux_indicator_filtered.append(item)
            elif item["name"] == "DecaySWBranchSnagToAir":
                item["name"] = "DecayBranchSnagToAir"
                cbm_exn_flux_indicator_filtered.append(item)
            elif item["name"] == "DisturbanceSWStemSnagToAir":
                item["name"] = "DisturbanceStemSnagToAir"
                cbm_exn_flux_indicator_filtered.append(item)
            elif item["name"] == "DisturbanceSWBranchSnagToAir":
                item["name"] = "DisturbanceBranchSnagToAir"
                cbm_exn_flux_indicator_filtered.append(item)
            elif item["name"] in [
                "DisturbanceHardProduction",
                "DecayHWStemSnagToAir",
                "DecayHWBranchSnagToAir" "DisturbanceHWStemSnagToAir",
                "DisturbanceHWBranchSnagToAir",
                "DisturbanceHWStemSnagToAir",
                "DisturbanceHWBranchSnagToAir",
            ]:
                continue
            else:
                cbm_exn_flux_indicator_filtered.append(item)
    write_json(
        cbm_exn_flux_indicator_filtered, os.path.join(output_dir, "flux.json")
    )


def _disturbance_matrices(db_path: str, output_dir: str):
    cbm3_pools = query(db_path, "select * from pool")
    cbm3_dm_values = query(db_path, "select * from disturbance_matrix_value")
    merged_dm_values = cbm3_dm_values.merge(
        cbm3_pools.rename(columns={"id": "sink_pool_id", "code": "sink_pool"}),
        left_on="sink_pool_id",
        right_on="sink_pool_id",
        how="left",
    )
    merged_dm_values = merged_dm_values.merge(
        cbm3_pools.rename(
            columns={"id": "source_pool_id", "code": "source_pool"}
        ),
        left_on="source_pool_id",
        right_on="source_pool_id",
        how="left",
    )

    merged_dm_values["has_sw"] = merged_dm_values.source_pool_id.isin(
        [1, 2, 3, 4, 5, 18, 19]
    ) | merged_dm_values.sink_pool_id.isin([1, 2, 3, 4, 5, 18, 19])

    merged_dm_values["has_hw"] = merged_dm_values.source_pool_id.isin(
        [6, 7, 8, 9, 10, 20, 21]
    ) | merged_dm_values.sink_pool_id.isin([6, 7, 8, 9, 10, 20, 21])
    if (merged_dm_values["has_hw"] & merged_dm_values["has_sw"]).any():
        raise ValueError()
    merged_dm_values_hw = merged_dm_values.loc[
        ~merged_dm_values.has_sw | merged_dm_values.has_hw
    ]
    merged_dm_values_sw = merged_dm_values.loc[
        merged_dm_values.has_sw | ~merged_dm_values.has_hw
    ]

    output_sw_dms = merged_dm_values_sw[
        ["disturbance_matrix_id", "source_pool", "sink_pool", "proportion"]
    ].copy()
    output_hw_dms = merged_dm_values_hw[
        ["disturbance_matrix_id", "source_pool", "sink_pool", "proportion"]
    ].copy()
    output_hw_dms["disturbance_matrix_id"] = (
        output_hw_dms["disturbance_matrix_id"]
        + output_hw_dms["disturbance_matrix_id"].max()
    )
    output_dms = pd.concat([output_sw_dms, output_hw_dms])

    output_dms["source_pool"] = output_dms.source_pool.str.replace(
        "Hardwood", "", regex=False
    ).str.replace("Softwood", "", regex=False)
    output_dms["sink_pool"] = output_dms.sink_pool.str.replace(
        "Hardwood", "", regex=False
    ).str.replace("Softwood", "", regex=False)
    output_dms.to_csv(
        os.path.join(output_dir, "disturbance_matrix_value.csv"), index=False
    )


def _dm_association(db_path: str, output_dir: str):
    cbm3_dm_association = query(
        db_path, "select * from disturbance_matrix_association"
    )
    dm_association_sw = cbm3_dm_association.copy()
    dm_association_sw["sw_hw"] = "sw"

    dm_association_hw = cbm3_dm_association.copy()
    dm_association_hw["sw_hw"] = "hw"
    dm_association_hw["disturbance_matrix_id"] = (
        dm_association_hw["disturbance_matrix_id"]
        + dm_association_hw["disturbance_matrix_id"].max()
    )
    output_dm_association = pd.concat([dm_association_sw, dm_association_hw])[
        [
            "spatial_unit_id",
            "disturbance_type_id",
            "sw_hw",
            "disturbance_matrix_id",
        ]
    ]
    output_dm_association.to_csv(
        os.path.join(output_dir, "disturbance_matrix_association.csv"),
        index=False,
    )


def _write_pools(output_dir: str):
    cbm_exn_pools = [
        "Input",
        "Merch",
        "Foliage",
        "Other",
        "CoarseRoots",
        "FineRoots",
        "AboveGroundVeryFast",
        "BelowGroundVeryFast",
        "AboveGroundFast",
        "BelowGroundFast",
        "MediumSoil",
        "AboveGroundSlow",
        "BelowGroundSlow" "StemSnag",
        "BranchSnag",
        "CO2",
        "CH4",
        "CO",
        "NO2",
        "Products",
    ]
    write_json(cbm_exn_pools, os.path.join(output_dir, "pools.json"))


def extract(db_path: str, output_dir: str):
    pass


if __name__ == "__main__":
    pass

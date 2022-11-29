import os
import json
import pandas as pd
import sqlite3
from argparse import ArgumentParser


def query(db_path, query):
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, con)
    return df


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4)


def _flux_indicator_config(db_path: str, output_dir: str):
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


def _slow_mixing_rate(db_path: str, output_dir: str):
    query(db_path, "select * from slow_mixing_rate").to_csv(
        os.path.join(output_dir, "slow_mixing_rate.csv"), index=False
    )


def _decay_parameters(db_path: str, output_dir: str):
    qry = """
    select * from pool
    inner join dom_pool on
    dom_pool.pool_id = pool.id
    inner join decay_parameter on
    decay_parameter.dom_pool_id = dom_pool.id
    """
    decay_rates = query(db_path, qry)

    decay_rates = decay_rates[
        [
            "code",
            "base_decay_rate",
            "reference_temp",
            "q10",
            "prop_to_atmosphere",
            "max_rate",
        ]
    ]
    decay_rates = decay_rates.rename(columns={"code": "pool"})
    decay_rates["pool"] = decay_rates["pool"].str.replace(
        "Hardwood", "", regex=False
    )
    decay_rates["pool"] = decay_rates["pool"].str.replace(
        "Softwood", "", regex=False
    )
    decay_rates.drop_duplicates().to_csv(
        os.path.join(output_dir, "decay_parameters.csv"), index=False
    )


def _root_parameters(db_path: str, output_dir: str):
    root_parameter = query(db_path, "select * from root_parameter")
    root_parameter["biomass_to_carbon_rate"] = 0.5
    root_parameter.to_csv(
        os.path.join(output_dir, "root_parameters.csv"), index=False
    )


def _species(db_path: str, output_dir: str, locale_code: str):
    qry_txt = """
        select
        species.id as species_id,
        species_tr.name as species_name,
        species.genus_id as genus_id,
        genus_tr.name as genus_name,
        species.forest_type_id as forest_type_id,
        forest_type_tr.name as forest_type_name
        from species
        inner join species_tr on species_tr.species_id = species.id
        inner join genus on species.genus_id = genus.id
        inner join genus_tr on genus.id = genus_tr.genus_id
        inner join forest_type on species.forest_type_id = forest_type.id
        inner join forest_type_tr on forest_type.id
         = forest_type_tr.forest_type_id
        inner join locale on species_tr.locale_id = locale.id
        where locale.code = ?
            and genus_tr.locale_id = species_tr.locale_id
            and forest_type_tr.locale_id = species_tr.locale_id
        """
    query(db_path, qry_txt, [locale_code]).to_csv(
        os.path.join(output_dir, "species.csv"), index=False
    )


def _turnover_parameters(db_path: str, output_dir: str):
    qry_text = """
    select spatial_unit.id as spatial_unit_id,
    turnover_parameter.* from turnover_parameter
    inner join eco_boundary on
    eco_boundary.turnover_parameter_id=turnover_parameter.id
    inner join spatial_unit on spatial_unit.eco_boundary_id = eco_boundary.id
    """
    turnover_parameter = query(db_path, qry_text)
    cbm_exn_turnover = pd.concat(
        [
            pd.DataFrame(
                {
                    "spatial_unit_id": turnover_parameter["spatial_unit_id"],
                    "sw_hw": "sw",
                    "FoliageFallRate": turnover_parameter["sw_foliage"],
                    "StemAnnualTurnoverRate": turnover_parameter[
                        "stem_turnover"
                    ],
                    "BranchTurnoverRate": turnover_parameter["sw_branch"],
                    "CoarseRootTurnProp": turnover_parameter["coarse_root"],
                    "FineRootTurnProp": turnover_parameter["fine_root"],
                    "OtherToBranchSnagSplit": turnover_parameter[
                        "branch_snag_split"
                    ],
                    "CoarseRootAGSplit": turnover_parameter["coarse_ag_split"],
                    "FineRootAGSplit": turnover_parameter["fine_ag_split"],
                    "StemSnag": turnover_parameter["sw_stem_snag"],
                    "BranchSnag": turnover_parameter["sw_branch_snag"],
                }
            ),
            pd.DataFrame(
                {
                    "spatial_unit_id": turnover_parameter["spatial_unit_id"],
                    "sw_hw": "hw",
                    "FoliageFallRate": turnover_parameter["hw_foliage"],
                    "StemAnnualTurnoverRate": turnover_parameter[
                        "stem_turnover"
                    ],
                    "BranchTurnoverRate": turnover_parameter["hw_branch"],
                    "CoarseRootTurnProp": turnover_parameter["coarse_root"],
                    "FineRootTurnProp": turnover_parameter["fine_root"],
                    "OtherToBranchSnagSplit": turnover_parameter[
                        "branch_snag_split"
                    ],
                    "CoarseRootAGSplit": turnover_parameter["coarse_ag_split"],
                    "FineRootAGSplit": turnover_parameter["fine_ag_split"],
                    "StemSnag": turnover_parameter["hw_stem_snag"],
                    "BranchSnag": turnover_parameter["hw_branch_snag"],
                }
            ),
        ]
    )
    cbm_exn_turnover.to_csv(
        os.path.join(output_dir, "turnover_parameters.csv"), index=False
    )


def extract(db_path: str, output_dir: str, locale_code: str):
    _write_pools(output_dir)
    _flux_indicator_config(db_path, output_dir)
    _disturbance_matrices(db_path, output_dir)
    _dm_association(db_path, output_dir)
    _slow_mixing_rate(db_path, output_dir)
    _decay_parameters(db_path, output_dir)
    _root_parameters(db_path, output_dir)
    _species(db_path, output_dir, locale_code)
    _turnover_parameters(db_path, output_dir)


def main():
    parser = ArgumentParser(
        description=(
            "extract parameters for cbm_exn from a cbm_defaults database"
        )
    )

    parser.add_argument(
        "db_path",
        help="path to a cbm_defaults sqlite database",
        type=os.path.abspath,
    )

    parser.add_argument(
        "output_dir",
        help="output directory where parameters will be written",
        type=os.path.abspath,
    )
    parser.add_argument(
        "locale_code",
        help=(
            "locale code eg. 'en-CA' or 'fr-CA'.  Determines the language of "
            "parameter metadata (species names, disturbance type names) drawn "
            "from the cbm_default database"
        ),
        type=os.path.abspath,
    )
    parser.parse_args
    extract()


if __name__ == "__main__":
    main()

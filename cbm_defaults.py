#loads cbm defaults into configuration dictionary format
import sqlite3

queries = {
    "decay_parameters" : """
        select
        pool.id as Pool,
        decay_parameter.base_decay_rate as OrganicMatterDecayRate,
        decay_parameter.reference_temp as ReferenceTemp,
        decay_parameter.q10 as Q10,
        decay_parameter.prop_to_atmosphere as PropToAtmosphere,
        decay_parameter.max_rate as MaxDecayRate
        from decay_parameter
        inner join dom_pool on dom_pool.id = decay_parameter.dom_pool_id
        inner join pool on pool.id = dom_pool.pool_id;
        """,

    "slow_mixing_rate": "select rate from slow_mixing_rate;",

    "mean_annual_temp": """
        select spatial_unit.id as spatial_unit_id, spatial_unit.mean_annual_temperature
        from spatial_unit;
    """,

    "turnover_parameters": """
        select eco_boundary.id as EcoBoundaryId,
        turnover_parameter.sw_foliage as SoftwoodFoliageFallRate,
        turnover_parameter.hw_foliage as HardwoodFoliageFallRate,
        turnover_parameter.stem_turnover as StemAnnualTurnoverRate,
        turnover_parameter.sw_branch as SoftwoodBranchTurnoverRate,
        turnover_parameter.hw_branch as HardwoodBranchTurnoverRate,
        turnover_parameter.coarse_ag_split as CoarseRootAGSplit,
        turnover_parameter.coarse_root as CoarseRootTurnProp,
        turnover_parameter.fine_ag_split as FineRootAGSplit,
        turnover_parameter.fine_root as FineRootTurnProp,
        turnover_parameter.branch_snag_split as OtherToBranchSnagSplit,
        turnover_parameter.branch_snag as BranchSnagTurnoverRate,
        turnover_parameter.stem_snag as StemSnagTurnoverRate
        from eco_boundary
        inner join turnover_parameter on
        eco_boundary.turnover_parameter_id = turnover_parameter.id; 
    """,

    "disturbance_matrix_values": "select * from disturbance_matrix_value",

    "disturbance_matrix_associations": "select * from disturbance_matrix_association;",
    
    "root_parameter": """
        select root_parameter.*,
        biomass_to_carbon_rate.rate as biomass_to_carbon_rate
        from root_parameter, biomass_to_carbon_rate
    """,

    "growth_multipliers": """
        select
        growth_multiplier_series.disturbance_type_id,
        growth_multiplier_value.forest_type_id,
        growth_multiplier_value.time_step,
        growth_multiplier_value.value
        from growth_multiplier_series
        inner join growth_multiplier_value on
        growth_multiplier_value.growth_multiplier_series_id =
        growth_multiplier_series.id
    """,

    "land_classes": """
        select land_class.id, land_class.is_forest,
        land_class.transitional_period, land_class.transition_id
        from land_class;
    """,

    "land_class_transitions": """
        select disturbance_type.id as disturbance_type_id,
        disturbance_type.transition_land_class_id
        from disturbance_type
        where disturbance_type.transition_land_class_id is not null
    """,

    "spatial_units": "select spatial_unit.id as spatial_unit_id, spatial_unit.eco_boundary_id from spatial_unit;",

    "random_return_interval": """
        select eco_boundary.id as eco_boundary_id,
        random_return_interval.a_Nu,
        random_return_interval.b_Nu,
        random_return_interval.a_Lambda,
        random_return_interval.b_Lambda
        from eco_boundary inner join
        random_return_interval on
        eco_boundary.random_return_interval_id =
        random_return_interval.id;
    """,

    "flux_indicator_process": """
        select flux_indicator.id as flux_indicator_id, flux_process.id as flux_process_id from flux_indicator
        inner join flux_process on flux_process.id = flux_indicator.flux_process_id
    """,

    "flux_indicator_source": """
        select flux_indicator.id, flux_indicator_source.pool_id from flux_indicator
        inner join flux_indicator_source on flux_indicator_source.flux_indicator_id = flux_indicator.id
    """,

    "flux_indicator_sink": """
        select flux_indicator.id, flux_indicator_sink.pool_id from flux_indicator
        inner join flux_indicator_sink on flux_indicator_sink.flux_indicator_id = flux_indicator.id
    """

}

def load_cbm_parameters(sqlitePath):
    result = {}
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        for table, query in queries.items():
            cursor.execute(query)
            data = [[col for col in row] for row in cursor]
            if table in result:
                raise AssertionError("duplicate table name detected {}".format(table))
            result[table] = {
                "column_map": { v[0]: i for i,v in enumerate(cursor.description) },
                "data": data
            }

    return result

def load_cbm_pools(sqlitePath):
    result = []
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        index = 0
        for row in cursor.execute("select code, id from pool order by id"):
            result.append({"name": row[0], "id": row[1], "index": index})
            index += 1
        return result

def load_flux_indicators(sqlitePath):
    result = []
    flux_indicator_source_sql = """
        select flux_indicator_source.pool_id from flux_indicator
        inner join flux_indicator_source on flux_indicator_source.flux_indicator_id = flux_indicator.id 
        where flux_indicator.id = ?"""
    flux_indicator_sink_sql = """
        select flux_indicator_sink.pool_id from flux_indicator
        inner join flux_indicator_sink on flux_indicator_sink.flux_indicator_id = flux_indicator.id 
        where flux_indicator.id = ?"""
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        index = 0;
        flux_indicator_rows = list(cursor.execute("select id, flux_process_id from flux_indicator"))
        for row in flux_indicator_rows:
            flux_indicator = {
                "id": row[0],
                "index": index,
                "process_id": row[1],
                "source_pools": [],
                "sink_pools": []
            }
            for source_pool_row in cursor.execute(flux_indicator_source_sql, (row[0],)):
                flux_indicator["source_pools"].append(int(source_pool_row[0]))
            for sink_pool_row in cursor.execute(flux_indicator_sink_sql, (row[0],)):
                flux_indicator["sink_pools"].append(int(sink_pool_row[0]))
            result.append(flux_indicator)
            index += 1
        return result


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
        select land_class.id, land_class.is_forest, land_class.is_simulated
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
    """,

    "spinup_parameter": """
        select
        spatial_unit.id as spatial_unit_id,
        spinup_parameter.return_interval,
        spinup_parameter.historic_mean_temperature,
        spinup_parameter.min_rotations,
        spinup_parameter.max_rotations,
        spinup_parameter.return_interval
        from spinup_parameter inner join spatial_unit on
        spatial_unit.spinup_parameter_id == spinup_parameter.id
    """,

    "afforestation_pre_type": """
        select
        afforestation_pre_type.id,
        afforestation_initial_pool.spatial_unit_id,
        afforestation_initial_pool.pool_id,
        afforestation_initial_pool.value
        from afforestation_pre_type 
        inner join afforestation_initial_pool on 
        afforestation_initial_pool.afforestation_pre_type_id = afforestation_pre_type.id
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


def load_species_reference(path, locale_code="en-CA"):
    query = """
        select species_tr.name, species.id, species.forest_type_id
        from species 
        inner join species_tr on species_tr.species_id = species.id
        inner join locale on species_tr.locale_id = locale.id
        where locale.code = ?"""
    result = {}
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query, (locale_code,)):
            result[row[0]] = {"species_id": int(row[1]), "forest_type_id": int(row[2])}
    return result


def get_spatial_unit_ids_by_admin_eco_name(sqlitePath, locale_code="en-CA"):
    query = """
        select spatial_unit.id, admin_boundary_tr.name as admin_boundary_name, eco_boundary_tr.name as eco_boundary_name from spatial_unit
        inner join eco_boundary on eco_boundary.id = spatial_unit.eco_boundary_id
        inner join admin_boundary on admin_boundary.id = spatial_unit.admin_boundary_id
        inner join eco_boundary_tr on eco_boundary_tr.eco_boundary_id = eco_boundary.id
        inner join admin_boundary_tr on admin_boundary_tr.admin_boundary_id = admin_boundary.id
        inner join locale on admin_boundary_tr.locale_id = locale.id
        where admin_boundary_tr.locale_id = eco_boundary_tr.locale_id and locale.code = ?
        order by spatial_unit.id
        """
    result = {}
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query, (locale_code,)):
            result[(row[1],row[2])] = row[0]
    return result


def get_disturbance_type_ids_by_name(sqlitePath, locale_code="en-CA"):
    query = """
        select  disturbance_type.id, disturbance_type_tr.name
        from disturbance_type 
        inner join disturbance_type_tr on disturbance_type_tr.disturbance_type_id == disturbance_type.id
        inner join locale on disturbance_type_tr.locale_id = locale.id
        where locale.code = ?
        """
    result = {}
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query, (locale_code,)):
            result[row[1]] = row[0]
    return result


def get_flux_indicator_names_by_id(sqlitePath):
    query = """select flux_indicator.id, flux_indicator.name from flux_indicator
        """
    result = {}
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query):
            result[row[0]] = row[1]
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


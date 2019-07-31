# methods for finding name/id associations in a CBM defaults
# database

import sqlite3
import libcbm.configuration.cbm_defaults_queries as queries


def load_data(sqlite_path, query, locale_code="en-CA"):
    result = []
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        for row in cursor.execute(query, (locale_code,)):
            result.append(row)
    return result


def load_species_reference(sqlite_path, locale_code="en-CA"):
    """Loads CBM species data into a list of tuples

    Arguments:
        sqlite_path {str} -- sqlitePath {str} -- Path to a CBM parameters
        database as formatted like: https://github.com/cat-cfs/cbm_defaults

    Keyword Arguments:
        locale_code {str} -- language tag for the returned names
            (default: {"en-CA"})

    Returns:
        list -- [description]
    """
    query = queries.get_query("species_name_ref.sql")
    return load_data(sqlite_path, query, locale_code)


def get_spatial_unit_reference(sqlitePath, locale_code="en-CA"):
    query = """
        select spatial_unit.id, admin_boundary_tr.name as admin_boundary_name,
        eco_boundary_tr.name as eco_boundary_name from spatial_unit
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
            result[(row[1], row[2])] = row[0]
    return result


def get_land_class_disturbance_reference(sqlitePath, locale_code="en-CA"):
    query = """
        select
        disturbance_type.id as disturbance_type_id,
        disturbance_type_tr.name as disturbance_type_name,
        land_class.id as land_class_id,
        land_class.code as land_class_code,
        land_class_tr.description as land_class_description
        from disturbance_type
        inner join land_class on disturbance_type.transition_land_class_id = land_class.id
        inner join land_class_tr on land_class_tr.land_class_id = land_class.id
        inner join disturbance_type_tr on disturbance_type_tr.disturbance_type_id == disturbance_type.id
        inner join locale dt_loc on disturbance_type_tr.locale_id = dt_loc.id
        inner join locale lc_loc on land_class_tr.locale_id = lc_loc.id
        where dt_loc.code = ? and lc_loc.code = ?
    """
    result = []
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query, (locale_code,locale_code)):
            result.append({
            "disturbance_type_id": row[0],
            "disturbance_type_name": row[1],
            "land_class_id": row[2],
            "land_class_code": row[3],
            "land_class_description": row[4]
            })
    return result

def get_land_class_reference(sqlitePath, locale_code="en-CA"):
    query = """
        select
        land_class.id as land_class_id, land_class.code, land_class_tr.description
        from land_class
        inner join land_class_tr on land_class_tr.land_class_id = land_class.id
        inner join locale on land_class_tr.locale_id = locale.id
        where locale.code = ?
    """
    result = []
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query, (locale_code,)):
            result.append({
            "land_class_id": row[0],
            "land_class_code": row[1],
            "land_class_description": row[2]
            })
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

def get_afforestation_types_by_name(sqlitePath, locale_code="en-CA"):
    query = """
        select afforestation_pre_type.id, afforestation_pre_type_tr.name
        from afforestation_pre_type inner join afforestation_pre_type_tr
        on afforestation_pre_type_tr.afforestation_pre_type_id = afforestation_pre_type.id
        inner join locale on afforestation_pre_type_tr.locale_id = locale.id
        where locale.code = ? and afforestation_pre_type.id>0
        """
    result = {}
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query, (locale_code,)):
            result[row[1]] = row[0]
    return result


def get_flux_indicator_names(sqlitePath):
    query = """select flux_indicator.id, flux_indicator.name from flux_indicator
        """
    result = []
    with sqlite3.connect(sqlitePath) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query):
            result.append({"id": row[0], "name": row[1]})
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



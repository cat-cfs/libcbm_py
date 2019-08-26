# methods for finding name/id associations in a CBM defaults
# database

import sqlite3
import pandas as pd
import libcbm.model.cbm.cbm_defaults_queries as queries


class CBMDefaultsReference:

    def __init__(self, sqlite_path, locale_code="en-CA"):
        """Creates a reference to the localized name and id relationships
        stored in a cbm_defaults database.

        Args:
            sqlite_path (str): path to a cbm_defaults sqlite database.
            locale_code (str, optional): locale code as defined in the locale
                table of the cbm_defaults database. Defaults to "en-CA".
        """

        # queries for species name/species id associations
        self.species_reference_query = queries.get_query("species_ref.sql")

        # queries for disturbance type name/disturbance type id associations
        self.disturbance_reference_query = queries.get_query(
            "disturbance_type_ref.sql")

        # queries for spatial unit id, admin boundary name, eco boundary name
        # associations
        self.spatial_unit_reference_query = queries.get_query(
            "spatial_units_name_ref.sql")

        # queries information on disturbance types which have an effect on
        # UNFCCC land class
        self.land_class_disturbance_query = queries.get_query(
            "land_class_disturbance_ref.sql")

        # queries for land class name,id,code,descriptions
        self.land_class_query = queries.get_query("land_class_ref.sql")

        # queries for afforestation pre-type/name associations
        self.afforestation_pre_type_query = queries.get_query(
            "afforestation_pre_type_ref.sql")

        # queries for names of flux indicators id/name associations
        self.flux_indicator_query = queries.get_query("flux_indicator_ref.sql")

        self.pools_query = queries.get_query("pools.sql")

        locale_param = (locale_code,)
        self.species_ref = self.load_data(
            sqlite_path, self.species_reference_query, locale_param)
        self.species_by_name = {x["species_name"]: x for x in self.species_ref}

        self.disturbance_type_ref = self.load_data(
            sqlite_path, self.disturbance_reference_query, locale_param)
        self.disturbance_type_by_name = {
            x["disturbance_type_name"]: x for x in self.disturbance_type_ref}

        self.spatial_unit_ref = self.load_data(
            sqlite_path, self.spatial_unit_reference_query, locale_param)
        self.spatial_unit_by_admin_eco_names = {
            (x["admin_boundary_name"], x["eco_boundary_name"]): x
            for x in self.spatial_unit_ref}

        self.afforestation_pre_type_ref = self.load_data(
            sqlite_path, self.afforestation_pre_type_query, locale_param)
        self.afforestation_pre_type_by_name = {
            x["afforestation_pre_type_name"]: x
            for x in self.afforestation_pre_type_ref}

        self.land_class_ref = self.load_data(
            sqlite_path, self.land_class_query, locale_param)
        self.land_class_by_code = {
            x["code"]: x for x in self.land_class_ref}

        self.pools_ref = self.load_data(sqlite_path, self.pools_query)

        self.flux_indicator_ref = self.load_data(
            sqlite_path, self.flux_indicator_query)

        self.land_class_disturbance_ref = self.load_data(
            sqlite_path, self.land_class_disturbance_query, locale_param)
        self.land_classes_by_dist_type = {
            x["disturbance_type_name"]: x
            for x in self.land_class_disturbance_ref}

    def load_data(self, sqlite_path, query, query_params=None):
        """loads the specified query into a list of dictionary formatted query

        Args:
            sqlite_path (str): path to a SQLite database
            query (str): sqlite query
            query_params (Tuple, optional): tuple of query parameters to pass
                to the sqlite3 execute method. Defaults to None.

        Returns:
            list: a list of sqlite3.Row objects containing the query results
        """
        with sqlite3.connect(sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if query_params:
                return cursor.execute(query, query_params).fetchall()
            else:
                return cursor.execute(query).fetchall()

    def get_species_id(self, species_name):
        """Get the species id associated with the specified species name.

        Args:
            species_name (str): a species name

        Returns:
            int: the id associated with the specified name
        """
        return self.species_by_name[species_name]["species_id"]

    def get_species(self):
        """Get all name and id information about every CBM species.

        Result is returned as a list of rows with keys:

            - species_id
            - species_name
            - genus_id
            - genus_name
            - forest_type_id
            - forest_type_name

        Returns:
            list: list of rows with species information
        """
        return self.species_ref

    def get_disturbance_type_id(self, disturbance_type_name):
        """Get the disturbance type id associated with the specified
        disturbance type name

        Args:
            disturbance_type_name (str): a disturbance type name

        Returns:
            int: disturbance type id
        """
        match = self.disturbance_type_by_name[disturbance_type_name]
        return match["disturbance_type_id"]

    def get_disturbance_types(self):
        """Get all name and id information about every CBM disturbance type
        as a list of rows with keys:

            - disturbance_type_id
            - disturbance_type_name

        Returns:
            list -- list of rows with disturbance type information
        """
        return self.disturbance_type_ref

    def get_spatial_unit_id(self, admin_boundary_name, eco_boundary_name):
        """Get the spatial unit id associated with the specified
        admin-boundary-name, eco-boundary-name combination

        Args:
            admin_boundary_name (str): an admin boundary name
            eco_boundary_name (str): an eco boundary name

        Returns:
            int: the spatial unit id
        """
        return self.spatial_unit_by_admin_eco_names[
            (admin_boundary_name, eco_boundary_name)]["spatial_unit_id"]

    def get_spatial_units(self):
        """Get name and id information for the spatial units defined in the
        underlying cbm_defaults database.  Returns a list of rows
        with the following keys:

            - spatial_unit_id
            - admin_boundary_name
            - eco_boundary_name

        Returns:
            list: rows containing spatial unit data
        """
        return self.spatial_unit_ref

    def get_afforestation_pre_type_id(self, afforestation_pre_type_name):
        """Get the afforestation pre-type id associated with the specified
        afforestation pre-type name

        Args:
            afforestation_pre_type_name (str): an afforestation pre-type name

        Returns:
            int: the associated afforestation_pre_type_id
        """
        return self.afforestation_pre_type_by_name[
            afforestation_pre_type_name]["afforestation_pre_type_id"]

    def get_afforestation_pre_types(self):
        """Get name and id information for the afforestation pre-types
        defined in the underlying cbm_defaults database.  Returns a list
        of dictionaries with the following keys:

            - afforestation_pre_type_id
            - afforestation_pre_type_name

        Returns:
            list: list of rows with afforestation pre type data
        """
        return self.afforestation_pre_type_ref

    def get_land_class_id(self, land_class_code):
        """Get the land class id associated with the specified CBM land class
        code (where a code might be for example: UNFCCC_FL_R_FL)

        Args:
            land_class_code (str): a CBM formatted UNFCCC land class code

        Returns:
            int: the land class id associated with the code
        """
        return self.land_class_by_code[land_class_code]["land_class_id"]

    def get_land_class_disturbance_ref(self):
        """Get name and id information for the cbm disturbance types that
        cause a change to UNFCCC land class, along with the post-disturbance
        land class.  Non-land class altering disturbance types are not
        included in the result. Returns a list of dictionaries with the
        following keys:

            - disturbance_type_id
            - disturbance_type_name
            - land_class_id
            - land_class_code
            - land_class_description

        Returns:
            list: a list of rows with disturbance type/landclass data
        """
        return self.land_class_disturbance_ref

    def get_land_class_by_disturbance_type(self, disturbance_type_name):
        """For the specified disturbance_type_name get UNFCCC land class
        info for the land class caused by the disturbance type.  If no
        UNFCCC land class is associated, return None.

        Args:
            disturbance_type_name (str): a disturbance type name

        Returns:
            dict, or None: if a match is found for the specified name, a
            dictionary containing values for:

                - land_class_id
                - land_class_code
                - land_class_description

            is returned
        """
        if disturbance_type_name not in self.land_classes_by_dist_type:
            return None
        match = self.land_classes_by_dist_type[disturbance_type_name]
        return {
            "land_class_id": match["land_class_id"],
            "land_class_code": match["land_class_code"],
            "land_class_description": match["land_class_description"]
        }

    def get_pools(self):
        """Get the ordered list of human readable pool codes defined in cbm_defaults

        Returns:
            list: list of string codes for cbm pools
        """
        return [x["code"] for x in self.pools_ref]

    def get_flux_indicators(self):
        """Get the ordered list of human readable flux indicator codes defined
            in cbm_defaults

        Returns:
            list: list of string names of flux indicators
        """
        return [x["name"] for x in self.flux_indicator_ref]

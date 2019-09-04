

def classifier_value(value, description=""):
    """Composes a classifier value structure used for configuring libcbm.

    Args:
        value (str): the classifier value string identifier
        description (str, optional): An optional classifier value description.
            Defaults to "".

    Returns:
        dict: a dictionary formatted for libcbm.

            Example return value::

                {
                    "value": "c1_v1",
                    "description": "c1_v1"
                }
    """
    return {
        "value": value,
        "description": description
    }


def classifier(name, values):
    """Composes a classifier structure used for configuring libcbm.

    Args:
        name (str): the name of the classifier
        values (dict): The classifier values that compose this classifier.
            Use classifier values of the format returned by:
            :py:func:`classifier_value`

    Returns:
        dict: A dictionary with the following keys:

            - "classifier": The classifier
            - "classifier_values": The list of classifier value dictionaries
                associated with the classifier

            Example return value::

                {
                    "classifier": {"name": "c1"},
                    "classifier_values": [
                        {
                            "value": "c1_v1",
                            "description": "c1_v1"
                        },
                        ...
                    ]
                }
    """
    return {
        "classifier": {"name": name},
        "classifier_values": values
    }


def classifier_config(classifiers):
    """Compose the classifier value scheme used for configuring libcbm.
    This method will generate classifier ids and classifier value ids
    based on the order of classifiers and classifier values.

    Args:
        classifiers (list): A list of dictionaries describing classifiers.
            For example::

                [
                    {
                        "classifier": {
                            "name" "c1"
                        },
                        "classifier_values":[
                            {"name": "c1_v1", "description": "c1_v1"},
                            {"name": "c1_v2", "description": "c1_v2"},
                            ...
                            {"name": "c1_vn", "description": "c1_vn"}
                        ]
                    },
                    {
                        "classifier": {
                            "name" "c2"
                        },
                        "classifier_values":[
                            {"name": "c2_v1", "description": "c2_v1"},
                            {"name": "c2_v2", "description": "c2_v2"},
                            ...
                            {"name": "c2_vk", "description": "c1_vk"}
                        ]
                    },
                    ...,
                    {
                        "classifier": {
                            "name" "cj"
                        },
                        "classifier_values":[
                            {"name": "cj_v1", "description": "cj_v1"},
                            {"name": "cj_v2", "description": "cj_v2"},
                            ...
                            {"name": "cj_vi", "description": "cj_vi"}
                        ]
                    }
                ]

            where:
            - j is the number of classifiers,
            - n,k,i are the number of classifier values for each
            corresponding classifier

            The dictionaries composing this argument can be created by the
            :py:func:`classifier_value`, and :py:func:`classifier` functions

    Returns:
        dict: A dictionary with the following keys

            - "classifiers": the list of classifiers
            - "classifier_values": the list of classifier values

            This is the format required by the libcbm CBM model component.

            For example: (following the same pattern given in the input)::

                {
                "classifiers": [
                    {"id": 1, "name": "c1"},
                    {"id": 2, "name": "c2"},
                    ...,
                    {"id": j, "name": "cj"}
                ],
                "classifier_values": [
                    {"id": 1, "classifier_id": 1, "name": "c1_v1", ...},
                    {"id": 2, "classifier_id": 1, "name": "c1_v2", ...},
                    ...,
                    {"id": n, "classifier_id": 1, "name": "c1_vn", ...},
                    {"id": n+1, "classifier_id": 2, "name": "c2_v1", ...},
                    {"id": n+2, "classifier_id": 2, "name": "c2_v2", ...},
                    ...,
                    {"id": n+k, "classifier_id": 2, "name": "c2_vk", ...},
                    {"id": n+k+1, "classifier_id": j, "name": "cj_v1", ...},
                    {"id": n+k+2, "classifier_id": j, "name": "cj_v2", ...},
                    ...,
                    {"id": n+k+i, "classifier_id": j, "name": "cj_vi", ...}
                ]
                }
    """
    result = {
        "classifiers":       [],
        "classifier_values": [],
    }
    for i, c in enumerate(classifiers):
        classifier = c["classifier"]
        values = c["classifier_values"]
        classifier["id"] = i + 1
        result["classifiers"].append(classifier)
        for cv in values:
            cv["id"] = len(result["classifier_values"]) + 1
            cv["classifier_id"] = classifier["id"]
            result["classifier_values"].append(cv)
    return result


def merch_volume_curve(classifier_set, merch_volumes):
    """Formats merchantable volume growth curve data for libcbm CBM model
    consumption.

    Args:
        classifier_set (list): a list of classifier value names ordered by
            classifiers. For example::

                ["c1_v1", "c2_v1", ... "ck_vx"]

        merch_volumes (list): a list of dictionaries with keys:

            - species_id,
            - age_volume_pairs

            for example::

                [
                    {
                        "species_id": 1,
                        "age_volume_pairs": [(a1,v1),(a2,v2),...,(an,vn)]
                    },
                    {
                        "species_id": 6,
                        "age_volume_pairs": [(a1,v1),(a2,v2),...,(an,vn)]
                    },
                ]


    Returns:
        dict: a dictionary with keys `classifier_set` and `components`.

            For example::

                {
                "classifier_set": {
                    "type": "name",
                    "values": ["c1_v1", "c2_v1", ... "ck_vx"]
                },
                "components": [
                    {
                        "species_id": 1,
                        "age_volume_pairs": [(a1,v1),(a2,v2),...,(an,vn)]
                    },
                    {
                        "species_id": 6,
                        "age_volume_pairs": [(a1,v1),(a2,v2),...,(an,vn)]
                    },
                ]
                }

    """
    result = {
        "classifier_set": {
            "type": "name", "values": [x for x in classifier_set]},
    }
    components = []
    for m in merch_volumes:
        components.append({
            "species_id": m["species_id"],
            "age_volume_pairs": m["age_volume_pairs"]
        })
    result["components"] = components
    return result


def merch_volume_to_biomass_config(db_path, merch_volume_curves):
    """Formats merchantable volume growth curve data for libcbm CBM model
    consumption.

    Args:
        db_path (str): path to a cbm_defaults database
        merch_volume_curves (list): a list of dictionaries in the same format
            as the return value of :py:func:`merch_volume_curve`

    Returns:
        dict: A dictionary containing configuration for merchantable volume
              growth in the CBM model

              For example::

                {
                    "db_path": "cbm_defaults.db"
                    "merch_volume_curves": [
                        {
                            "classifier_set": {
                                "type": "name",
                                "values": ["c1_v1", "c2_v1", ... "ck_vx"]
                            },
                            "components": [
                                {
                                    "species_id": 1,
                                    "age_volume_pairs": [(a1,v1),
                                                         (a2,v2),
                                                         ...,
                                                         (an,vn)]
                                },
                                {
                                    "species_id": 6,
                                    "age_volume_pairs": [(a1,v1),
                                                         (a2,v2),
                                                         ...,
                                                         (an,vn)]
                                },
                            ]
                        },
                        {
                            "classifier_set": {
                                "type": "name",
                                "values": ["c1_v5", "c2_v1", ... "ck_vx"]
                            },
                            "components": [
                                {
                                    "species_id": 2,
                                    "age_volume_pairs": [(a1,v1),
                                                         (a2,v2),
                                                         ...,
                                                         (an,vn)]
                                },
                                {
                                    "species_id": 65,
                                    "age_volume_pairs": [(a1,v1),
                                                         (a2,v2),
                                                         ...,
                                                         (an,vn)]
                                },
                            ]
                        }
                    ]
                }
    """
    return {"db_path": db_path, "merch_volume_curves": merch_volume_curves}

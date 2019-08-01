

def classifier_value(value, description=""):
    return {
        "id": None,
        "classifier_id": None,
        "value": value,
        "description": description}


def classifier(name, values):
    return {"id": None, "name": name}, values


def classifier_config(classifiers):
    """
    Returns a dictionary like this one:

        {'classifiers':       [{'id': 1, 'name': 'growth_curve'}],
         'classifier_values': [{
             'id': 1, 'classifier_id': 1, 'value': '1', 'description': ''}]}

    """
    result = {
        "classifiers":       [],
        "classifier_values": [],
    }
    for i, c in enumerate(classifiers):
        classifier = c[0]
        values = c[1]
        classifier["id"] = i + 1
        result["classifiers"].append(classifier)
        for j, cv in enumerate(values):
            cv["id"] = j+1
            cv["classifier_id"] = classifier["id"]
            result["classifier_values"].append(cv)
    return result


def merch_volume_curve(classifier_set, merch_volumes):
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
    return {"db_path": db_path, "merch_volume_curves": merch_volume_curves}

import json

def loadjson(path):
    with open(path) as f:
        data = json.load(f)
    return data

def create_layer_classifiers(layer_json_paths):
    result = {} 
    result["classifiers"]=[]
    result["classifier_values"]=[]
    result["raster_id_xref"]={}
    classifier_value_id = 1
    id_xref = []
    for i, path in enumerate(layer_json_paths):
        classifier_id = i + 1
        data = loadjson(path)
        classifier_name = data["attributes"][0]
        result["classifiers"].append({
            "id": classifier_id,
            "name": classifier_name})
        result["raster_id_xref"][classifier_name] = {}
        for raster_code, values in data["attribute_table"].items():
            raster_code = int(raster_code)
            result["raster_id_xref"][classifier_name][raster_code] = classifier_value_id
            result["classifier_values"].append({
                "id": classifier_value_id, 
                "classifier_id": classifier_id,
                "value": values[0],
                "description": ""})
            classifier_value_id += 1
    return result




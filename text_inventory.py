import csv
import numpy as np

def load(path, classifiers, classifier_values):
    n_classifiers = len(classifiers)
    classifiers_data_header = [x["name"] for x in classifiers]
    data_header = ["index", "age","spatial_unit","historic_disturbance_type","last_pass_disturbance_type","delay"]

    classifier_id_lookups = {}
    for c in classifiers:
        classifier_id = c["id"]
        classifier_id_lookup= {}

        for cv in classifier_values:
            if cv["classifier_id"] == c["id"]:
                classifier_id_lookup[cv["value"]] = cv["id"]

        classifier_id_lookups[c["name"]] = classifier_id_lookup


    classifiers_array = []

    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for x in reader:
            id_list = []
            for c in classifiers:
                cv_id = classifier_id_lookups[c["name"]][x[c["name"]]]
                id_list.append(cv_id)
            classifiers_array.append(id_list)


    usecols = [0] #index col
    usecols.extend(range(n_classifiers+1,n_classifiers+len(data_header)))
    inventory = np.genfromtxt(
        path,
        delimiter=',',
        usecols=tuple(usecols),
        skip_header=True,
        dtype=np.int32)


    if len(inventory.shape)==1:
        inventory = inventory.reshape((1,inventory.shape[0]))

    nstands = inventory.shape[0]
    result = {
        "nstands": inventory.shape[0],
        "classifiers": np.array(classifiers_array, dtype=np.int32)
    }
    
    for h in data_header:
        result[h] = inventory[:,data_header.index(h)]

    return result

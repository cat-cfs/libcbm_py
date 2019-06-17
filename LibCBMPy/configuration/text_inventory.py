#parses text formatted (csv) inventory into numpy arrays
import csv, os
import tempfile
import numpy as np

def create_classifier_lookup_table(classifiers, classifier_values):
    classifier_id_lookups = {}
    for c in classifiers:
        classifier_id = c["id"]
        classifier_id_lookup= {}

        for cv in classifier_values:
            if cv["classifier_id"] == c["id"]:
                classifier_id_lookup[cv["value"]] = cv["id"]

        classifier_id_lookups[c["name"]] = classifier_id_lookup

    return classifier_id_lookups

def load(path, classifiers, classifier_values, disturbance_type_ids, spatial_unit_ids):
    n_classifiers = len(classifiers)


    classifier_id_lookups = create_classifier_lookup_table(classifiers, classifier_values)
    with open(path) as csvfile, tempfile.NamedTemporaryFile('w', delete=False) as temp:
        reader = csv.DictReader(csvfile)
        writer = csv.writer(temp)
        for row in reader:
            index = int(row["index"])
            classifier_value_ids = []
            for classifier in classifiers:
                classifier_name = classifier["name"]
                classifier_id_lookup = classifier_id_lookups[classifier_name]
                classifier_value_ids.append(classifier_id_lookup[row[classifier_name]])
            age = row["age"]
            admin_boundary= row["admin_boundary"]
            eco_boundary=row["eco_boundary"]
            spatial_unit_id = spatial_unit_ids[(admin_boundary, eco_boundary)]
            historic_disturbance_type_id = disturbance_type_ids[row["historic_disturbance_type"]]
            last_pass_disturbance_type_id = disturbance_type_ids[row["last_pass_disturbance_type"]]
            delay = int(row["delay"])
            outrow = [index]
            outrow.extend(classifier_value_ids)
            outrow.extend([age, spatial_unit_id, historic_disturbance_type_id, last_pass_disturbance_type_id,delay ])
            writer.writerow(outrow)
        
    inventory = np.genfromtxt(temp.name,dtype=np.int32, delimiter=',')
    os.remove(temp.name)

    if len(inventory.shape)==1:
        inventory = inventory.reshape((1,inventory.shape[0]))

    nstands = inventory.shape[0]
    result = {
        "nstands": inventory.shape[0],
        "index": np.ascontiguousarray(inventory[:,0]),
        "classifiers": np.ascontiguousarray(inventory[:,1:(1+n_classifiers)]),
        "age": np.ascontiguousarray(inventory[:,n_classifiers+1]),
        "spatial_unit": np.ascontiguousarray(inventory[:,n_classifiers+2]),
        "historic_disturbance_type": np.ascontiguousarray(inventory[:,n_classifiers+3]),
        "last_pass_disturbance_type": np.ascontiguousarray(inventory[:,n_classifiers+4]),
        "delay": np.ascontiguousarray(inventory[:,n_classifiers+5])
    }

    return result

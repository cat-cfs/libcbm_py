#loads text formatted events into dictionary (step, numpy array)
import csv

def load_text_events(path, disturbance_type_ids):

    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for x in reader:
            index = x["index"]
            step = x["step"]
            dist_type_id = disturbance_type_ids[x["disturbance_type"]]
            transition_rule_id = x["transition_rule_id"]


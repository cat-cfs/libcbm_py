#loads text formatted events into dictionary (step, numpy array)
import csv
import numpy as np

def load_text_events(path, nstands, disturbance_type_ids):

    disturbance_type_output = {}
    transition_rules_output = {}
    rows_by_step = {}
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for x in reader:
            index = int(x["index"])
            step = int(x["step"])
            dist_type_id = disturbance_type_ids[x["disturbance_type"]]
            transition_rule_id = int(x["transition_rule_id"])
            row = (index,dist_type_id,transition_rule_id)
            if step in rows_by_step:
                rows_by_step[step].append(row)
            else:
                rows_by_step[step] = [row]
        for step, rows in rows_by_step.items():
            disturbance_type_output[step] = np.zeros(nstands, dtype = np.int32)
            transition_rules_output[step] = np.zeros(nstands, dtype = np.int32)
            for row in rows:
                disturbance_type_output[step][row[0]] = row[1]
                transition_rules_output[step][row[0]] = row[2]
    return {
        "disturbance": disturbance_type_output,
        "transition": transition_rules_output
        }




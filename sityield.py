#parses SIT_yield-like formatted files

import csv, sqlite3

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


def get_grouped_components(filtered_group, age_class_size, num_yields, species_ref):
    if len(filtered_group) == 0:
        return None
    sortedComponents = sorted(
        filtered_group,
        key = lambda x: sum(x["volumes"]),
        reverse=True)
    
    leadingSpecies = sortedComponents[0]["species"]
    ages = list(range(0,num_yields*age_class_size, age_class_size))
    volumes = [0]*num_yields
    for y in range(0,num_yields):
        for i in sortedComponents:
            volumes[y] += i["volumes"][y]
    return {"species_id":species_ref[leadingSpecies]["species_id"],
            "age_volume_pairs": list(zip(ages,volumes)) }

def read_sit_yield(path, cbm_defaults_path, num_classifiers, age_class_size,
                locale_code="en-CA", header=True, delimiter=',' ):
    num_yields = None
    species_ref = load_species_reference(cbm_defaults_path, locale_code)
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter = delimiter)
        if header:
            next(reader, None)
        parsed_rows = []
        for row in reader:
            parsed_row = {
                "classifiers": row[0:num_classifiers],
                "species": row[num_classifiers],
                "volumes": [float(x) for x in row[num_classifiers+1:]]
            }
            if num_yields  is None:
                num_yields = len(parsed_row["volumes"])
            elif num_yields != len(parsed_row["volumes"]):
                raise ValueError("inconsistent number of yield values")
            if not parsed_row["species"] in species_ref:
                raise ValueError("specified species '{}' not found in cbm_defaults"
                                 .format(parsed_row["species"]))
            parsed_rows.append(parsed_row)

        grouped_by_classifiers = {}
        for p in parsed_rows:
             key = tuple(p["classifiers"])
             if key in grouped_by_classifiers:
                 grouped_by_classifiers[key].append(p)
             else:
                 grouped_by_classifiers[key] = [p]
        result = []
        for cset,group in grouped_by_classifiers.items():
            softwood = get_grouped_components([x for x in group
                                              if species_ref[x["species"]]["forest_type_id"] == 1],
                                              age_class_size,
                                              num_yields,
                                              species_ref)

            hardwood = get_grouped_components([x for x in group 
                                               if species_ref[x["species"]]["forest_type_id"] != 1],# not equal softwood is cbm3 behaviour
                                              age_class_size,
                                              num_yields,
                                              species_ref)

            growth_curve = {"classifier_set": {"type": "name", "values": list(cset)}}
            if softwood:
                growth_curve["softwood_component"] = softwood
            if hardwood:
                growth_curve["hardwood_component"] = hardwood
            if not softwood and not hardwood:
                raise ValueError("curve has neither hardwood or softwood")
            result.append(growth_curve)

        return result


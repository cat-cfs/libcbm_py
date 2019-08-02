import math
import numpy as np
from libcbm.configuration.cbm_defaults_reference import CBMDefaultsReference


def get_classifier_value_name(id):
    """encodes a test case id as a classifier value name

    Arguments:
        id {int} -- test case id

    Returns:
        str -- encoded classifier value name
    """
    return str(id)


def get_random_sigmoid_func():
    """randomly parameterizes and returns a sigmoid function
    of x

    Returns:
        func -- a sigmoid unction
    """
    x_0 = np.random.rand(1)[0] * 100
    L = np.random.rand(1)[0] * 400
    k = 0.1

    def sigmoid(x):
        return L/(1+math.exp(-k*(x-x_0)))
    return sigmoid


def get_step_func():
    """randomly parameterizes and returns a step function
    of x

    Returns:
        func -- a step function
    """
    y = np.random.rand(1)[0] * 500
    minX = np.random.randint(low=1, high=200)

    def step(x):
        if x == 0:
            return 0
        if x >= minX:
            return y
        else:
            return 0
    return step


def get_ramp_func():
    """randomly parameterizes and returns a ramp function
    of x

    Returns:
        func -- a ramp function
    """
    rate = np.random.rand(1)[0] * 5

    def ramp(x):
        return x*rate
    return ramp


def get_expCurve_func():
    """randomly parameterizes and returns a exponential curve as a function
    of x

    Returns:
        func -- an exp curve function
    """
    yMax = np.random.rand(1)[0] * 500

    def expCurve(x):
        return yMax - math.exp(-x) * yMax
    return expCurve


def choose_random_yield_func(func_factories=[
            get_random_sigmoid_func,
            get_step_func,
            get_ramp_func,
            get_expCurve_func]):
    return np.random.choice(func_factories, 1)[0]()


def generate_scenarios(random_seed, num_cases, db_path, n_steps,
                       max_disturbances, max_components, n_growth_digits,
                       age_interval, growth_curve_len, growth_only=False):
    """create a list of test cases for comparing CBM-CFS3 versus libCBM.

    Arguments:
        random_seed {int} -- a random seed for all random functions within
            the test generator
        num_cases {int} -- the number of random cases (ie. number of stands
            to generate)
        db_path {str} -- path to a cbm_defaults database
        n_steps {int} -- the number of timesteps to simulate
        max_disturbances {int} -- the maximum number of disturbances that
            can occur on randomly generated stand scenarios
        max_components {int} -- the maximum number of growth curve component
            that can occur on a randomly generated stand
        n_growth_digits {[type]} -- included since CBM3 has a rounding to the
            second decimal place for growth curves
        age_interval {int} -- the number of years between growth curve values
        growth_curve_len {int} -- the number of growth curve values

    Keyword Arguments:
        growth_only {bool} -- if set to True, stands will only be forested
            initially (default: {False})

    Returns:
        list -- a list of parameters which can be run as stand level
        simulations by both libCBM and by CBM-CFS3
    """
    np.random.seed(random_seed)
    ref = CBMDefaultsReference(db_path, "en-CA")
    species_ref = ref.get_species()

    fire_type = "Wildfire"
    harvest_type = "Clearcut harvesting with salvage"
    deforestation_type = "Deforestation"
    afforestation_type = "Afforestation"

    # exclude species names that are too long for the CBM-CFS3 project
    # database schema, and forest_types that are not hardwood or softwood
    species = [
        k["species_name"] for k in species_ref
        if len(k["species_name"]) < 50 and k["forest_type_id"] in [1, 3]
        ]

    spatial_units = ref.get_spatial_units()

    random_spatial_units = np.random.choice([
        ",".join([x["admin_boundary_name"], x["eco_boundary_name"]])
        for x in spatial_units], num_cases)

    disturbance_types = ref.get_disturbance_types()

    # the following disturbance type ids don't have full coverage for all
    # spatial units, so if they are included it is possible a random draw can
    # produce an invalid combination of dist type/spu
    disturbance_types = [
        x for x in disturbance_types
        if x["disturbance_type_id"] not in
        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    # the 4 spruce beetle types have a strange unicode issue in the name
    disturbance_types = [
        x for x in disturbance_types
        if "Spruce beetle" not in x["disturbance_type_name"]]

    disturbance_type_names = [
        x["disturbance_type_name"] for x in disturbance_types]

    afforestation_pre_types = ref.get_afforestation_pre_types()
    afforestation_pre_type_names = [
        x["afforestation_pre_type_name"] for x in afforestation_pre_types]

    post_deforestation_land_class = ref.get_land_class_by_disturbance_type(
        "Deforestation")
    post_deforestation_land_class_code = \
        post_deforestation_land_class["land_class_code"]

    cases = []
    for i in range(num_cases):
        num_components = np.random.randint(1, max_components) \
            if max_components > 1 else 1
        random_species = np.random.choice(species, num_components)
        spu = random_spatial_units[i].split(',')
        components = []
        for c in range(num_components):
            growth_func = choose_random_yield_func()
            components.append({
                "species": random_species[c],
                "age_volume_pairs": [
                    (x, round(growth_func(x), n_growth_digits))
                    for x in range(0, growth_curve_len, age_interval)]
            })

        disturbance_events = []
        if max_disturbances > 0:
            num_disturbances = np.random.randint(0, max_disturbances)
            random_dist_types = np.random.choice(
                disturbance_type_names, num_disturbances)
            if num_disturbances > 0:
                event_interval = n_steps // num_disturbances
                for d in range(num_disturbances):
                    min_timestep = event_interval * d + 1
                    max_timestep = event_interval * (d + 1) + 1
                    disturbance_events.append({
                        "disturbance_type": random_dist_types[d],
                        "time_step": np.random.randint(
                            min_timestep, max_timestep)
                    })

        creation_disturbance = np.random.choice([
            fire_type, harvest_type, deforestation_type, afforestation_type
            ], 1)[0]

        age = 0
        delay = 0
        afforestation_pre_type = None
        last_pass_disturbance = fire_type
        historic_disturbance = fire_type
        unfccc_land_class = "UNFCCC_FL_R_FL"
        if not growth_only:
            if creation_disturbance in [fire_type, harvest_type]:
                age = np.random.randint(0, 350)
                last_pass_disturbance = creation_disturbance
            if creation_disturbance == deforestation_type:
                delay = np.random.randint(0, 20)  # UNFCCC rules
                last_pass_disturbance = creation_disturbance
                unfccc_land_class = post_deforestation_land_class_code
            if creation_disturbance == afforestation_type:
                # since there are constant pools in the afforestation case,
                # spinup, and therefore historic/last pass disturbance types
                # do not apply
                unfccc_land_class = "UNFCCC_CL_R_CL"
                afforestation_pre_type = np.random.choice(
                    list(afforestation_pre_type_names), 1)[0]
                if len(disturbance_events) > 0:
                    # Since we are trying to model the afforestation case,
                    # override the randomly selected first disturbance with
                    # afforestation.
                    disturbance_events[0]["disturbance_type"] = \
                        afforestation_type
                else:
                    disturbance_events.append({
                        "disturbance_type": afforestation_type,
                        "time_step": np.random.randint(1, n_steps)
                    })

        cases.append({
            "id": i + 1,
            "age": age,
            "area": 1.0,
            "delay": delay,
            "afforestation_pre_type": afforestation_pre_type,
            "unfccc_land_class": unfccc_land_class,
            "admin_boundary": spu[0],
            "eco_boundary": spu[1],
            "historic_disturbance": historic_disturbance,
            "last_pass_disturbance": last_pass_disturbance,
            "components": components,
            "events": disturbance_events})

    return cases

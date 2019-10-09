from libcbm.model.cbm.rule_based import rule_target
from libcbm.input.sit import sit_disturbance_event_parser


def is_production_sort(sit_event_row):
    production_sorts = ["MERCHCSORT_TOTAL", "MERCHCSORT_SW", "MERCHCSORT_HW"]
    if sit_event_row["sort_type"] in production_sorts:
        # sorted by the production
        return True

def is_production_based(sit_event_row):
    """Returns true if the specified disturbance event requires computation
    of production via disturbance matrix

    Args:
        sit_event_row (dict): a row dict from parsed SIT disturbance events
    """
    if is_production_sort():
        return True
    if sit_event_row["target"] == \
        sit_disturbance_event_parser.get_target_types()["M"]:
        # the production is the target variable
        return True
    return False


def get_production_sort_value(sort_type, pools, production):
    if sort_type == "MERCHCSORT_TOTAL":
        return production.Total
    elif sort_type == "MERCHCSORT_SW":
        return production.DisturbanceSoftProduction + \
               production.DisturbanceDOMProduction
    elif sort_type == "MERCHCSORT_HW":
        return production.DisturbanceHardProduction + \
               production.DisturbanceDOMProduction
    else:
        raise ValueError(
            f"specified sort_type '{sort_type}' is not a production sort")


def get_sort_value(sort_type, pools, state_variables):
    if sort_type == "SORT_BY_SW_AGE" or sort_type == "SORT_BY_HW_AGE":
        return state_variables.age
    elif sort_type == "TOTALSTEMSNAG":
        return pools.SoftwoodStemSnag + pools.HardwoodStemSnag
    elif sort_type == "SWSTEMSNAG":
        return pools.SoftwoodStemSnag
    elif sort_type == "HWSTEMSNAG":
        return pools.HardwoodStemSnag
    else:
        raise ValueError(
            f"specified sort_type '{sort_type}' is not sort value")


def create_sit_event_target(sit_event_row, cbm, cbm_defaults_ref, pools, inventory, on_unrealized):

    sort = sit_event_row["sort_type"]
    target = sit_event_row["target"]
    disturbance_type_name = sit_event_row["disturbance_type"]
    disturbance_type_id = cbm_defaults_ref.get_disturbance_type_id(
        disturbance_type_name)
    area_target_type = sit_disturbance_event_parser.get_target_types()["A"]
    non_sorted = ["SVOID", "PROPORTION_OF_EVERY_RECORD"]
    if is_production_based(sit_event_row):
        production = rule_target.compute_disturbance_production(
            cbm, pools, inventory, disturbance_type_id,
            cbm_defaults_ref.get_flux_indicators())
    if target == area_target_type and sort not in non_sorted:
        if is_production_sort(sit_event_row):
            rule_target.sorted_area_target(
                area_target_value=sit_event_row["target"],
                sort_value=production,
                inventory=inventory,
                on_unrealized=on_unrealized)
        else:
            rule_target.sorted_area_target(
                area_target_value=sit_event_row["target"],
                sort_value=production,
                inventory=inventory,
                on_unrealized=on_unrealized)

   #if sit_event_row["sort_type"] == "PROPORTION_OF_EVERY_RECORD"

   #"MERCHCSORT_TOTAL",
   #    3: "SORT_BY_SW_AGE",
   #    5: "SVOID ",
   #    6: "RANDOMSORT",
   #    7: "TOTALSTEMSNAG",
   #    8: "SWSTEMSNAG",
   #    9: "HWSTEMSNAG",
   #    10: "MERCHCSORT_SW",
   #    11: "MERCHCSORT_HW",
   #    12: "SORT_BY_HW_AGE"}"
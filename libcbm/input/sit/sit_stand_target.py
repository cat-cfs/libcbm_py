from libcbm.model.cbm.rule_based import rule_target
from libcbm.input.sit import sit_disturbance_event_parser


def is_production_based(sit_event_row):
    """Returns true if the specified disturbance event requires computation
    of production via disturbance matrix

    Args:
        sit_event_row (dict): a row dict from parsed SIT disturbance events
    """
    production_sorts = ["MERCHCSORT_TOTAL", "MERCHCSORT_SW", "MERCHCSORT_HW"]
    if sit_event_row["sort_type"] in production_sorts:
        # sorted by the production
        return True
    if sit_event_row["target"] == sit_disturbance_event_parser.get_target_types("M"):
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


def create_sit_event_target(sit_event_row, inventory):

    sort = sit_event_row["sort_type"]
    target = sit_event_row["target"]
    area_target_type = sit_disturbance_event_parser.get_target_types("A")
    non_sorted = ["SVOID", "PROPORTION_OF_EVERY_RECORD"]
    if target == area_target_type and sort not in non_sorted:

        rule_target.sorted_area_target(
            area_target_value = sit_event_row["target"],
            sort_value,
            inventory = inventory,
            on_unrealized)
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
from libcbm.input.sit import sit_disturbance_event_parser


def _is_production_sort(sit_event_row):
    production_sorts = [
        "MERCHCSORT_TOTAL", "MERCHCSORT_SW", "MERCHCSORT_HW"]
    if sit_event_row["sort_type"] in production_sorts:
        # sorted by the production
        return True


def _is_production_based(sit_event_row):
    """Returns true if the specified disturbance event requires computation
    of production via disturbance matrix

    Args:
        sit_event_row (dict): a row dict from parsed SIT disturbance events
    """
    if sit_event_row["sort_type"] == "SVOID":
        # spatially explicit cannot be a merch production target
        return False
    if _is_production_sort(sit_event_row):
        return True
    if sit_event_row["target_type"] == \
       sit_disturbance_event_parser.get_target_types()["M"]:
        # the production is the target variable
        return True
    return False


def _get_production_sort_value(sort_type, production):
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


def _get_sort_value(sort_type, pools, state_variables, random_generator):
    if sort_type == "SORT_BY_SW_AGE" or sort_type == "SORT_BY_HW_AGE":
        return state_variables.age
    elif sort_type == "TOTALSTEMSNAG":
        return pools.SoftwoodStemSnag + pools.HardwoodStemSnag
    elif sort_type == "SWSTEMSNAG":
        return pools.SoftwoodStemSnag
    elif sort_type == "HWSTEMSNAG":
        return pools.HardwoodStemSnag
    elif sort_type == "RANDOMSORT":
        return random_generator(pools.shape[0])
    else:
        raise ValueError(
            f"specified sort_type '{sort_type}' is not sort value")


def create_sit_event_target_factory(rule_target, sit_event_row,
                                    disturbance_production_func, eligible,
                                    random_generator, on_unrealized):

    def factory(pools, inventory, state_variables):
        return create_sit_event_target(
            rule_target, sit_event_row, pools, inventory, state_variables,
            disturbance_production_func, eligible, random_generator,
            on_unrealized)
    return factory


def create_sit_event_target(rule_target, sit_event_row,
                            pools, inventory, state_variables,
                            disturbance_production_func, eligible,
                            random_generator, on_unrealized):

    sort = sit_event_row["sort_type"]
    target_type = sit_event_row["target_type"]
    target = sit_event_row["target"]
    target_types = sit_disturbance_event_parser.get_target_types()
    area_target_type = target_types["A"]
    merchantable_target_type = target_types["M"]
    proportional_target_type = target_types["P"]
    non_sorted = ["SVOID", "PROPORTION_OF_EVERY_RECORD"]
    if _is_production_based(sit_event_row):
        production = disturbance_production_func(
            pools, inventory, sit_event_row["disturbance_type_id"])
    rule_target_result = None
    if target_type == area_target_type and sort not in non_sorted:
        if _is_production_sort(sit_event_row):
            rule_target_result = rule_target.sorted_area_target(
                area_target_value=target,
                sort_value=_get_production_sort_value(sort, production),
                inventory=inventory,
                eligible=eligible,
                on_unrealized=on_unrealized)
        else:
            rule_target_result = rule_target.sorted_area_target(
                area_target_value=target,
                sort_value=_get_sort_value(
                    sort, pools, state_variables, random_generator),
                inventory=inventory,
                eligible=eligible,
                on_unrealized=on_unrealized)
    elif target_type == merchantable_target_type and sort not in non_sorted:
        if _is_production_sort(sit_event_row):
            rule_target_result = rule_target.sorted_merch_target(
                carbon_target=target,
                disturbance_production=production.Total,
                inventory=inventory,
                sort_value=_get_production_sort_value(sort, production),
                efficiency=sit_event_row["efficiency"],
                eligible=eligible,
                on_unrealized=on_unrealized)
        else:
            rule_target_result = rule_target.sorted_merch_target(
                carbon_target=target,
                disturbance_production=production.Total,
                inventory=inventory,
                sort_value=_get_sort_value(
                    sort, pools, state_variables, random_generator),
                efficiency=sit_event_row["efficiency"],
                eligible=eligible,
                on_unrealized=on_unrealized)
    elif target == proportional_target_type:
        if sort != "PROPORTION_OF_EVERY_RECORD" or sort != "SVOID":
            raise ValueError(
                f"specified sort: '{sort}', target: '{target}' combination "
                "not valid")
    elif sort == "PROPORTION_OF_EVERY_RECORD":
        if target_type == area_target_type:
            rule_target_result = rule_target.proportion_area_target(
                area_target_value=target,
                inventory=inventory,
                eligible=eligible,
                on_unrealized=on_unrealized)
        elif target_type == merchantable_target_type:
            rule_target_result = rule_target.proportion_merch_target(
                carbon_target=target,
                disturbance_production=production.Total,
                inventory=inventory,
                efficiency=sit_event_row["efficiency"],
                eligible=eligible,
                on_unrealized=on_unrealized)
    elif sort == "SVOID":
        rule_target_result = rule_target.spatially_indexed_target(
            identifier=sit_event_row["spatial_reference"],
            inventory=inventory)
    return rule_target_result

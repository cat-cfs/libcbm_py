import numpy as np
from libcbm.input.sit import sit_parser
from libcbm.input.sit import sit_format
from libcbm.input.sit import sit_classifier_parser


def parse(transition_rules, classifiers, classifier_values,
          classifier_aggregates, disturbance_types, age_classes):

    transition_rule_format = sit_format.get_transition_rules_format(
        classifiers.name, len(transition_rules.columns))

    transitions = sit_parser.unpack_table(
        transition_rules, transition_rule_format, "yield")

    # check that each value in transition_rules events classifier sets is
    # defined in classifier values, classifier aggregates or is a wildcard
    for row in classifiers.itertuples():
        source_classifiers = transitions[row.name].unique()

        # get the destination classifier
        tr_dest_fmt = sit_format.get_transition_rule_classifier_set_postfix()
        dest_classifiers = transitions[f"{row.name}{tr_dest_fmt}"]

        defined_classifiers = classifier_values[
            classifier_values["classifier_id"] == row.id]["name"].unique()

        aggregates = np.array(
            [x["name"] for x in
             classifier_aggregates if x["classifier_id"] == row.id])
        wildcard = np.array([sit_classifier_parser.get_wildcard_keyword()])
        valid_source_classifiers = np.concatenate(
            [defined_classifiers, aggregates, wildcard])

        diff_source = np.setdiff1d(
            source_classifiers, valid_source_classifiers)
        if len(diff_source) > 0:
            raise ValueError(
                "Undefined classifier values detected: "
                f"classifier: '{row.name}', values: {diff_source}")

        # aggregates may not appear in transition rule destination classifier
        # set (only the defined classifier values, or wildcards)
        valid_dest_classifiers = np.concatenate(
            [defined_classifiers, wildcard])
        diff_dest = np.setdiff1d(
            dest_classifiers, valid_dest_classifiers)
        if len(diff_dest) > 0:
            raise ValueError(
                "Undefined classifier values detected: "
                f"classifier: '{row.name}', values: {diff_dest}")

    parse_bool_func = sit_parser.get_parse_bool_func(
        "transitions", "using_age_class")
    transitions = substitute_using_age_class_rows(
        transitions, parse_bool_func, age_classes)

    # check that all age criteria are identical between SW and HW (since CBM
    # has only a stand age)
    differing_age_criteria = transitions.loc[
        (transitions.min_softwood_age != transitions.min_hardwood_age) |
        (transitions.max_softwood_age != transitions.max_hardwood_age)]
    if len(differing_age_criteria) > 0:
        raise ValueError(
            "Values of column min_softwood_age must equal values of column "
            "min_hardwood_age, and values of column max_softwood_age must "
            "equal values of column max_hardwood_age since CBM defines only "
            "a stand age and does not track hardwood and softwood age "
            "seperately.")

    # validate and subsitute disturbance type names versus the SIT disturbance
    # types
    undefined_disturbances = np.setdiff1d(
        transitions.disturbance_type.unique(),
        disturbance_types.id.unique()
    )
    if len(undefined_disturbances) > 0:
        raise ValueError(
            "Undefined disturbance type ids (as defined in sit "
            f"disturbance types) detected: {undefined_disturbances}"
        )
    disturbance_join = transitions.merge(
        disturbance_types, left_on="disturbance_type",
        right_on="id")
    transitions.disturbance_type = disturbance_join.name

    transitions = transitions.rename(
        columns={
            "min_softwood_age": "min_age",
            "max_softwood_age": "max_age"})

    transitions = transitions.drop(
        columns=["using_age_class", "min_hardwood_age", "max_hardwood_age"])

    return transitions


def substitute_using_age_class_rows(rows, parse_bool_func, age_classes):
    # if age classes are used substitute the age critera based on the age
    # class id, and raise an error if the id is not defined, and drop
    # using_age_class from output
    rows.using_age_class = \
        rows.using_age_class.map(parse_bool_func)
    non_using_age_class_rows = rows.loc[
        ~rows.using_age_class]
    using_age_class_rows = rows.loc[
        rows.using_age_class].copy()

    for age_class_criteria_col in [
          "min_softwood_age", "min_hardwood_age",
          "max_softwood_age", "max_hardwood_age"]:
        valid_age_classes = np.concatenate(
            [age_classes.name.unique(), np.array(["-1"])])
        undefined_age_classes = np.setdiff1d(
            using_age_class_rows[age_class_criteria_col].astype(np.str).unique(),
            valid_age_classes)
        if len(undefined_age_classes) > 0:
            raise ValueError(
                f"In column {age_class_criteria_col}, the following age class "
                f"identifiers: {undefined_age_classes} are not defined in SIT "
                "age classes.")

    age_class_start_year_map = {
        x.name: x.start_year for x in age_classes.itertuples()}
    age_class_end_year_map = {
        x.name: x.end_year for x in age_classes.itertuples()}
    using_age_class_rows.min_softwood_age = using_age_class_rows \
        .min_softwood_age.astype(np.str).map(age_class_start_year_map)
    using_age_class_rows.min_hardwood_age = using_age_class_rows \
        .min_hardwood_age.astype(np.str).map(age_class_start_year_map)
    using_age_class_rows.max_softwood_age = using_age_class_rows \
        .max_softwood_age.astype(np.str).map(age_class_end_year_map)
    using_age_class_rows.max_hardwood_age = using_age_class_rows \
        .max_hardwood_age.astype(np.str).map(age_class_end_year_map)

    # if the above mapping fails, it results in Nan values in the failed rows,
    # this replaces those with -1
    using_age_class_rows.min_softwood_age = \
        using_age_class_rows.min_softwood_age.fillna(-1)
    using_age_class_rows.min_hardwood_age = \
        using_age_class_rows.min_hardwood_age.fillna(-1)
    using_age_class_rows.max_softwood_age = \
        using_age_class_rows.max_softwood_age.fillna(-1)
    using_age_class_rows.max_hardwood_age = \
        using_age_class_rows.max_hardwood_age.fillna(-1)

    # return the final substituted rows
    result = non_using_age_class_rows.append(using_age_class_rows) \
        .reset_index(drop=True)
    return result

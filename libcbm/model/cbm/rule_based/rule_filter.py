import numpy as np
import numexpr as ne


def filter_classifiers(mask, classifier_set, classifier_values,
                       classifiers_config, classifier_aggregates):
    n_classifiers = len(classifiers_config["classifiers"])
    if n_classifiers != classifier_values.shape[1] or \
       n_classifiers != len(classifier_set):
        raise ValueError(
            "mismatch in number of classifiers: "
            f"classifier_set {len(classifier_set)}, "
            f"classifiers_config: {n_classifiers}, "
            f"classifier value columns {classifier_values.shape[1]}")


    return false

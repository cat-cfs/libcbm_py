from types import SimpleNamespace
from libcbm.input.sit import sit_classifier_parser


class ClassifierFilter():
    """Creates a classifier filter
    """
    def __init__(self, classifiers_config, classifier_aggregates):
        self.wildcard_keyword = sit_classifier_parser.get_wildcard_keyword()
        self.classifiers_config = classifiers_config
        self.n_classifiers = len(self.classifiers_config["classifiers"])
        self.classifier_aggregates = classifier_aggregates
        self.classifier_lookup = {
            x["id"]: x["name"] for x in self.classifiers_config["classifiers"]
        }
        self.classifier_value_lookup = {
            x["name"]: self._get_classifier_value_index(x["id"])
            for x in self.classifiers_config["classifiers"]}
        self.aggregate_value_lookup = {
            x["name"]: self._get_classifier_aggregate_index(x["id"])
            for x in self.classifiers_config["classifiers"]
        }

    def _get_classifier_aggregate_index(self, classifier_id):
        result = {}
        for aggregate in self.classifier_aggregates:
            if aggregate["classifier_id"] != classifier_id:
                continue
            classifier_name = self.classifier_lookup[classifier_id]
            result[aggregate["name"]] = [
                self.classifier_value_lookup[classifier_name][y]
                for y in aggregate["classifier_values"]]
        return result

    def _get_classifier_value_index(self, classifier_id):
        return {
            x["name"]: x["id"] for x
            in self.classifiers_config["classifier_values"]
            if x["classifier_id"] == classifier_id}

    def create_classifiers_filter(self, classifier_set, classifier_values):
        """Creates a filter based on the specified classifier set to select a
        subset of the values in classifier_values

        Args:
            classifier_set (list): a list of strings, these may be any of:

                - a defined classifier value
                - a classifier aggregate
                - or a wildcard "?"

            classifier_values (pandas.DataFrame): dataframe of classifier
                value ids by stand (row), by classifier (columns).  Column
                labels are the classifier names.

        Raises:
            ValueError: mismatch in the number of classifiers
            ValueError: a classifier value in the specified classifier
                set is not defined

        Returns:
            object: an object with properties:

                - expression (str): a boolean expression to filter the values
                    in local_dict. The variables are defined as the keys in
                    local_dict.
                - local_dict (dict): a dictionary containing named numpy
                    variables to filter.

        """

        if self.n_classifiers != classifier_values.shape[1] or \
           self.n_classifiers != len(classifier_set):
            raise ValueError(
                "mismatch in number of classifiers: "
                f"classifier_set {len(classifier_set)}, "
                f"classifiers_config: {self.n_classifiers}, "
                f"classifier value columns {classifier_values.shape[1]}")

        expression_tokens = []

        def get_classifier_variable(num):
            return f"c_{num}"

        for i_classifier, classifier in enumerate(
                self.classifiers_config["classifiers"]):
            classifier_set_value = classifier_set[i_classifier]
            classifier_name = classifier["name"]
            classifier_id_by_name = self.classifier_value_lookup[
                classifier_name]
            aggregates = self.aggregate_value_lookup[classifier_name]
            if classifier_set_value in classifier_id_by_name:
                expression_tokens.append(
                    " == ".join([
                        get_classifier_variable(i_classifier),
                        str(classifier_id_by_name[classifier_set_value])]))
            elif classifier_set_value in aggregates:
                aggregate_expression_tokens = []
                aggregate_values = aggregates[classifier_set_value]
                for classifier_value_id in aggregate_values:
                    aggregate_expression_tokens.append(
                        " == ".join([
                            get_classifier_variable(i_classifier),
                            str(classifier_value_id)]))
                expression_tokens.append(
                    "({})".format(" | ".join(aggregate_expression_tokens)))
            elif classifier_set_value != self.wildcard_keyword:
                raise ValueError(
                    f"undefined classifier set value {classifier_set_value}")

        result = SimpleNamespace()
        result.expression = ""
        result.local_dict = {}
        if not expression_tokens:
            # this can happen if the classifier set is all wildcards
            return result

        result.expression = " & ".join(expression_tokens)
        result.local_dict = {
            get_classifier_variable(i): classifier_values[x["name"]].to_numpy()
            for i, x in enumerate(self.classifiers_config["classifiers"])}
        return result

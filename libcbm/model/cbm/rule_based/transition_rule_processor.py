# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations
from typing import Iterable
from typing import Tuple
import numpy as np
from libcbm.storage import dataframe
from libcbm.storage.dataframe import DataFrame
from libcbm.storage import series
from libcbm.storage.series import Series
from libcbm.model.cbm.cbm_variables import CBMVariables
from libcbm.model.cbm.rule_based import rule_filter
from libcbm.model.cbm.rule_based.rule_filter import RuleFilter


class TransitionRuleProcessor(object):
    def __init__(
        self,
        classifiers_config: dict[str, list],
        wildcard: str,
        transition_classifier_postfix: str,
    ):
        self._wildcard = wildcard
        self._transition_classifier_postfix = transition_classifier_postfix
        self.classifiers_config = classifiers_config
        self.classifier_names = [
            x["name"] for x in self.classifiers_config["classifiers"]
        ]
        self.classifier_value_lookup = {
            x["name"]: self._get_classifier_value_index(x["id"])
            for x in self.classifiers_config["classifiers"]
        }

    def _get_classifier_value_index(self, classifier_id: int) -> dict:
        return {
            x["value"]: x["id"]
            for x in self.classifiers_config["classifier_values"]
            if x["classifier_id"] == classifier_id
        }

    def _get_transition_classifier_set(
        self, transition_rule: dict
    ) -> Iterable[Tuple[str, int]]:
        for classifier_name in self.classifier_names:
            transition_classifier_value = transition_rule[
                classifier_name + self._transition_classifier_postfix
            ]
            if transition_classifier_value == self._wildcard:
                continue
            transition_id = self.classifier_value_lookup[classifier_name][
                transition_classifier_value
            ]
            yield classifier_name, transition_id

    def apply_transition_rule(
        self,
        tr_group: DataFrame,
        rule_filters: list[RuleFilter],
        proportions: list[float],
        transition_mask: Series,
        cbm_vars: CBMVariables,
    ) -> Tuple[Series, CBMVariables]:
        """Apply the specified transition rule group to the simulation
        variables, updating classifier values, and returning the transition
        rule variables reset age, and regeneration delay.  For each member of
        the transition rule group a split of the simulation variables will
        occur with area being reduced according to the "percent" column in
        the member transition rules.

        Args:
            tr_group (DataFrame): the grouped transition rules, where
                each row is a member.
            rule_filters (list): list of trnasition rule filters that includes
                or excludes areas for this transition.
            proportions (list): list of proportions adding up to 100. This
                is row-index-aligned with the specified `tr_group` with the
                exception that an extra element at the end representing
                non-transitioned proportion can be present.
            transition_mask (Series): a boolean mask indicating when
                true that the correspoding index has already been transitioned.
                This is used to detect transition rule criteria collisions.
            cbm_vars (CBMVariables): CBM simulation variables and state

        Raises:
            ValueError: a transition rule criteria resulted in the selection of
                stands targeted by at least one other transition rule

        Returns:
            tuple:

                - transition_mask: the specified transition_mask parameter is
                    returned altered with the indices transitioned by this
                    function call.
                - cbm_vars: updated and potentially expanded cbm variables and
                    state

        """
        filtered = rule_filter.evaluate_filters(*rule_filters)

        # sets the transitioned array with the transition filter result
        eligible = dataframe.logical_and(
            dataframe.logical_not(transition_mask), filtered
        )

        transition_mask_output = dataframe.logical_or(
            transition_mask, eligible
        )

        if not eligible.any():
            return transition_mask_output, cbm_vars
        eligible_idx = dataframe.indices_nonzero(eligible)

        # storage for split records
        classifier_split = None
        inventory_split = None
        pool_split = None
        state_split = None
        parameters_split = None
        flux_split = None
        next_id = cbm_vars.inventory["inventory_id"].max() + 1
        for i_proportion, proportion in enumerate(proportions):
            if i_proportion == 0:
                continue
                # the first proportion corresponds to the source data
                # for all remaining splits, so it must be updated last so
                # those changes don't propagate into the splits.

            # For all proportions other than the first we need to make
            # a copy of each of the state variables to split off the
            # percentage for the current group member
            pools = cbm_vars.pools.take(eligible_idx)
            flux = cbm_vars.flux.take(eligible_idx)
            parameters = cbm_vars.parameters.take(eligible_idx)
            state = cbm_vars.state.take(eligible_idx)
            inventory = cbm_vars.inventory.take(eligible_idx)
            classifiers = cbm_vars.classifiers.take(eligible_idx)
            transition_mask_output = dataframe.concat_series(
                [
                    transition_mask_output,
                    dataframe.make_boolean_series(
                        init=True,
                        size=pools.n_rows,
                        backend_type=cbm_vars.pools.backend_type,
                    ),
                ],
                backend_type=transition_mask.backend_type,
            )

            # set the area for the split portion according to the current
            # group member proportion
            inventory["area"].assign(inventory["area"] * proportion)
            inventory["parent_inventory_id"].assign(inventory["inventory_id"])

            inventory["inventory_id"].assign(
                series.range(
                    "inventory_id",
                    next_id,
                    next_id + inventory.n_rows,
                    1,
                    "int",
                    cbm_vars.inventory.backend_type,
                )
            )
            next_id = next_id + inventory.n_rows
            # if the current proportion is the remainder of 100 minus the
            # group's percentage sums, then this split portion will not be
            # transitioned, meaning the classifier set is not changed, and
            # the reset age, regeneration delay parameters will not take
            # effect for this split portion
            is_transition_split = i_proportion < tr_group.n_rows
            if is_transition_split:
                transition_classifier_ids = (
                    self._get_transition_classifier_set(
                        transition_rule=tr_group.at(i_proportion)
                    )
                )
                # update the split classifiers with the transitioned value
                for classifier_name, value_id in transition_classifier_ids:
                    classifiers[classifier_name].assign(value_id)

                state["regeneration_delay"].assign(
                    int(tr_group["regeneration_delay"].at(i_proportion)),
                )

                parameters["reset_age"].assign(
                    int(tr_group["reset_age"].at(i_proportion))
                )

            classifier_split = dataframe.concat_data_frame(
                [classifier_split, classifiers]
            )
            inventory_split = dataframe.concat_data_frame(
                [inventory_split, inventory]
            )
            pool_split = dataframe.concat_data_frame([pool_split, pools])
            state_split = dataframe.concat_data_frame([state_split, state])
            parameters_split = dataframe.concat_data_frame(
                [parameters_split, parameters]
            )
            flux_split = dataframe.concat_data_frame([flux_split, flux])

        classifiers = cbm_vars.classifiers
        inventory = cbm_vars.inventory
        pools = cbm_vars.pools
        state = cbm_vars.state
        parameters = cbm_vars.parameters
        flux = cbm_vars.flux
        # for the first index in the tr_group use the existing matched records
        transition_classifier_ids = self._get_transition_classifier_set(
            transition_rule=tr_group.at(0)
        )

        for classifier_name, value_id in transition_classifier_ids:
            classifiers[classifier_name].assign(
                np.int32(value_id), eligible_idx
            )

        if proportions[0] < 1.0:
            inventory["area"].assign(
                cbm_vars.inventory["area"].take(eligible_idx) * proportions[0],
                eligible_idx,
            )

        next_id = next_id + eligible_idx.length
        state["regeneration_delay"].assign(
            np.int32(tr_group["regeneration_delay"].at(0)),
            eligible_idx,
        )

        parameters["reset_age"].assign(
            np.int32(tr_group["reset_age"].at(0)), eligible_idx
        )

        if len(proportions) > 1:
            classifiers = dataframe.concat_data_frame(
                [classifiers, classifier_split]
            )
            inventory = dataframe.concat_data_frame(
                [inventory, inventory_split]
            )
            pools = dataframe.concat_data_frame([pools, pool_split])
            state = dataframe.concat_data_frame([state, state_split])
            parameters = dataframe.concat_data_frame(
                [parameters, parameters_split]
            )
            flux = dataframe.concat_data_frame([flux, flux_split])

        return transition_mask_output, CBMVariables(
            pools, flux, classifiers, state, inventory, parameters
        )

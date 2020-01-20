# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd


class RuleBasedStats:
    """Storage and method for appending to timestep by timestep model
    statistics.
    """
    def __init__(self):
        self.stats = pd.DataFrame()

    def append_stats(self, timestep, stats):
        """adds the contents of the argument stats along with a new column
        containing the timestep to this instances' stats dataframe.

        Args:
            timestep (int): the timestep for the data in stats
            stats (pandas.DataFrame): the stats to append to this instance
        """
        stats = stats.copy()
        stats.insert(loc=0, column="timestep", value=timestep)
        self.stats = self.stats.append(stats, sort=True)

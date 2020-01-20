import pandas as pd


class RuleBasedStats:
    """Storage for timestep by timestep statistics.
    """
    def __init__(self):
        self.stats = pd.DataFrame()

    def append_stats(self, timestep, stats):
        """adds the contents of the argument stats along with a new column
        containing the timestep to this instances' stats dataframe.

        Args:
            timestep (int): the timestep for the data in stats
            stats (pandas.DataFrame): the stats
        """
        stats = stats.copy()
        stats.insert(loc=0, column="timestep", value=timestep)
        self.stats = self.stats.append(stats)

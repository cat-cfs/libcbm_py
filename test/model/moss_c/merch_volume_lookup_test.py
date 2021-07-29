import unittest
import pandas as pd
import numpy as np
from libcbm.model.moss_c.merch_vol_lookup import MerchVolumeLookup


class MerchVolumeLookupTest(unittest.TestCase):

    def test_get_merch_volume(self):
        mv_lookup = MerchVolumeLookup(
            pd.DataFrame(
                index=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                columns=["age", "volume"],
                data=[
                    [0, 0],
                    [1, 10],
                    [2, 20],
                    [3, 40],
                    [4, 60],
                    [1, 0],
                    [2, 10],
                    [3, 15],
                    [4, 20],
                    [5, 15]]
            ))

        output = mv_lookup.get_merch_vol(
            age=np.array(
                [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6]),
            merch_vol_id=np.array(
                [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
            ))
        self.assertTrue(
            (output == np.array([
                0, 10, 20, 40, 60, 60, 0, 10, 15, 20, 15, 15])).all()
        )

    def test_missing_merch_volume(self):
        mv_lookup = MerchVolumeLookup(
            pd.DataFrame(
                index=[1, 1, 1, 1],
                columns=["age", "volume"],
                data=[
                    [0, 0],
                    [1, 10],
                    # [2, 20],
                    [3, 40],
                    [4, 60]]
            ))

        with self.assertRaises(ValueError):
            output = mv_lookup.get_merch_vol(
                age=np.array([2]),
                merch_vol_id=np.array([1]))

    def test_error_on_negative_values(self):
        with self.assertRaises(ValueError):
            MerchVolumeLookup(pd.DataFrame(
                index=[1], columns=["age", "volume"], data=[[-10, 0]]))
        with self.assertRaises(ValueError):
            MerchVolumeLookup(pd.DataFrame(
                index=[1], columns=["age", "volume"], data=[[10, -10]]))

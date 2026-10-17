import unittest
import pandas as pd
from src.evaluate import score
class ScoreTests(unittest.TestCase):
    def test_lists_and_series_use_positional_pairing(self):
        for actual,predicted in [([1,0,1,0],[1,1,0,0]),(pd.Series([1,0,1,0],index=[4,5,6,7]),pd.Series([1,1,0,0],index=[0,1,2,3]))]:
            result=score(actual,predicted)
            self.assertEqual([result[k] for k in ('tp','fp','fn','tn')],[1,1,1,1])
    def test_invalid_shapes_and_nonbinary_values_are_rejected(self):
        for actual,predicted in [([1],[1,0]),([[1]],[[1]]),([None],[0]),([2],[1])]:
            with self.assertRaises(ValueError):score(actual,predicted)

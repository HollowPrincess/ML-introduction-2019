import unittest
import pandas as pd
import numpy as np

import sys
sys.path.append("..")

from code_source import data_prep

class TestDataPrep(unittest.TestCase):
    def test_normalize(self):
        df=pd.DataFrame([[0,1],[2,3],[4,5]], columns=['col1','col2'])
        normalized, mean, std=data_prep.normalize(df)
        ans_mean=[2,3]
        ans_std=[np.sqrt(np.sum((df['col1']-ans_mean[0])**2)/3), np.sqrt(np.sum((df['col2']-ans_mean[1])**2)/3)]
        self.assertEqual(list(mean.values), ans_mean)
        self.assertEqual(list(std.values), ans_std)
        
        
if __name__ == '__main__':
    unittest.main()
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
        ans_df=pd.DataFrame()
        ans_df['col1']=(df['col1']-ans_mean[0])/ans_std[0]
        ans_df['col2']=(df['col2']-ans_mean[1])/ans_std[1]
        
        self.assertListEqual(list(mean.values), ans_mean)
        self.assertListEqual(list(std.values), ans_std)
        self.assertTrue(ans_df.equals(normalized))
        
    def test_normalize_with_params(self):
        df=pd.DataFrame([[0,1],[2,3],[4,5]], columns=['col1','col2'])
        
        mean=[1,0]
        std=[2,3]        
        normalized=data_prep.normalize_with_params(df, mean, std)
        
        ans_df=pd.DataFrame()
        ans_df['col1']=(df['col1']-mean[0])/std[0]
        ans_df['col2']=(df['col2']-mean[1])/std[1]
        
        self.assertTrue(ans_df.equals(normalized))
        
        
if __name__ == '__main__':
    unittest.main()
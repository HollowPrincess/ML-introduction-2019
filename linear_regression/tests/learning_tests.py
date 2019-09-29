import unittest
import pandas as pd
import numpy as np

import sys
sys.path.append("..")

from code_source import learning as l

class TestPrediction(unittest.TestCase):
    def test_w_from_zero_all_ind(self):
        old_w=[0,0]
        df=pd.DataFrame([[0,1,1],[2,3,1],[4,5,1]], columns=['col1','col2', 'target'])
        pred=np.array([0,0,0])
        lam=3
        gamma=0        
        indices=np.array(df.iloc[0:3].index)
        
        new_w=l.get_para_w(old_w, df, pred, lam, gamma, indices)
        ans=np.array([12,18], dtype=float)
        
        self.assertTrue(np.equal(new_w,ans).all()==True)
        
    def test_w_from_zero_two_ind(self):
        old_w=[0,0]
        df=pd.DataFrame([[0,1,1],[2,3,1],[4,5,1]], columns=['col1','col2', 'target'])
        pred=np.array([0,0,0])
        lam=3
        gamma=0        
        indices=np.array(df.iloc[1:3].index)
        
        new_w=l.get_para_w(old_w, df, pred, lam, gamma, indices)
        ans=np.array([12,16], dtype=float)
        
        self.assertTrue(np.equal(new_w,ans).all()==True)

    def test_w0_from_zero_two_ind(self):
        old_w0=0
        df=pd.DataFrame([[0,1,1],[2,3,1],[4,5,1]], columns=['col1','col2', 'target'])
        pred=np.array([0,0,0])
        lam=3
        gamma=0        
        indices=np.array(df.iloc[1:3].index)
        
        new_w=l.get_para_w0(old_w0, df, pred, lam, gamma, indices)
        ans=4
        
        self.assertEqual(new_w,ans)
  
    def test_get_pred_after_zero(self):
        df=pd.DataFrame([[0,1,1],[2,3,1],[4,5,1]], columns=['col1','col2', 'target'])
        w=np.array([12,18], dtype=float)
        w0=4
        
        pred=l.get_prediction(w,w0,df.iloc[:,:-1])
        ans=np.array([22,82,142], dtype=float)
        self.assertTrue(np.equal(pred,ans).all()==True)
        
    def test_w_from_nonzero_two_ind(self):
        old_w=[12,18]
        df=pd.DataFrame([[0,1,1],[2,3,1],[4,5,1]], columns=['col1','col2', 'target'])
        pred=np.array([22,82,142], dtype=float)
        lam=6
        gamma=0        
        indices=np.array(df.iloc[1:3].index)
        
        new_w=l.get_para_w(old_w, df, pred, lam, gamma, indices)
        ans=np.array([-2892,-3774], dtype=float)
        
        self.assertTrue(np.equal(new_w,ans).all()==True)
        
    def test_w0_from_nonzero_two_ind(self):
        old_w0=4
        df=pd.DataFrame([[0,1,1],[2,3,1],[4,5,1]], columns=['col1','col2', 'target'])
        pred=np.array([22,82,142], dtype=float)
        lam=6
        gamma=0        
        indices=np.array(df.iloc[1:3].index)
        
        new_w=l.get_para_w0(old_w0, df, pred, lam, gamma, indices)
        ans=-884
        
        self.assertEqual(new_w,ans)
 
        
if __name__ == '__main__':
    unittest.main()
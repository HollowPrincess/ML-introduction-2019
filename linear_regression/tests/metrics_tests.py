import unittest
import numpy as np

import sys
sys.path.append("..")

from code_source import metrics

class TestMetrics(unittest.TestCase):
    def test_best_RMSE(self):
        pred=np.array([0,1,2])
        target=pred.copy()
        res=metrics.custom_RMSE(target,pred,0,0)
        ans=0
        self.assertEqual(res, ans)
        
    def test_best_R2(self):
        pred=np.array([0,1,2])
        target=pred.copy()
        res=metrics.custom_R2(target,pred)
        ans=1
        self.assertEqual(res, ans)
        
    def test_RMSE(self):
        pred=np.array([0,1,2,0])
        target=np.array([2,1,1,2])
        res=metrics.custom_RMSE(target,pred,0,0)
        ans=1.5
        self.assertEqual(res, ans)
        
    def test_R2(self):
        pred=np.array([1,1,1,1])
        target=np.array([0,2,2,0])
        
        res=metrics.custom_R2(target,pred)
        ans=0
        self.assertEqual(res, ans)
        
if __name__ == '__main__':
    unittest.main()
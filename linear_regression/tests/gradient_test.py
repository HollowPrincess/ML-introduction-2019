import unittest
import pandas as pd
import numpy as np

import sys
sys.path.append("..")

from code_source import learning as l

class TestGradientsDescend(unittest.TestCase):
    def test_grad(self):
        #with start: w1=1,w2=0,w0=0
        df=pd.DataFrame() #it is: y=2*x1+0*x2+3
        df['col1']=np.array(range(0,10))
        df['col2']=np.array([0]*df.shape[0])
        df['target']=np.array(df['col1']*2+3)
        lam=10
        gamma=0        
        terms_num=2
        max_iter=100
        
        res=l.gradient_descend(df, lam, gamma, terms_num, max_iter)
        ans=[2,0,3] #model params
        print(res)
        self.assertTrue(abs(res[0]-2)<=0.01)
        self.assertTrue(abs(res[1])<=0.01)
        self.assertTrue(abs(res[2]-3)<=0.01)       
 
        
if __name__ == '__main__':
    unittest.main()
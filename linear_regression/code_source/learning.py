import numpy as np
import math
from sklearn.utils import shuffle
from code_source import metrics

def get_para_w(old_w, df, pred, lam, gamma, indices):
    #indices=np.random.choice(df.shape[0]-1,terms_num,replace=False)
    old_w=old_w+2*lam*(
          (np.dot((df.loc[indices,'target']-pred[indices]).values,df.iloc[indices,:-1].values))/len(pred)
        )
    return old_w

def get_para_w0(old_w0, df, pred, lam, gamma, indices):
    return old_w0+2*lam*(
        (np.sum(df.loc[indices,'target']-pred[indices]))/len(pred))

def get_prediction(w,w0,x_df):
    return np.dot(x_df, w)+w0
    
def gradient_descend(df, lam, gamma, terms_num, max_iter):
    df=df.reset_index(drop=True)
    w=np.random.rand(df.shape[1]-1)#np.array([0]*(df.shape[1]-1), dtype=float) #one column is the target
    w0=0.0
    prediction=get_prediction(w,w0,df.iloc[:,:-1])
    minRMSE=metrics.custom_RMSE(df['target'], prediction, gamma, w)

    best_params=np.append(w,w0)
    best_pred=prediction

    curr_err=10
    err=1e-4
    
    iter_num=0
    while (curr_err>err) & (iter_num<max_iter) :    
        iter_num+=1
        df=shuffle(df)
        prediction=get_prediction(w,w0,df.iloc[:,:-1])
        curr_err=prediction.copy()
        
        for batch_counter in range(0, math.ceil(df.shape[0]/terms_num)):
            indices=np.array(df.iloc[batch_counter*terms_num:(batch_counter+1)*terms_num].index)            
            w=get_para_w(w, df, prediction, lam, gamma, indices)
            w0=get_para_w0(w0, df, prediction, lam, gamma, indices)            
            prediction=get_prediction(w,w0,df.iloc[:,:-1])

            if metrics.custom_RMSE(df['target'], prediction, gamma, w)<minRMSE:
                minRMSE=metrics.custom_RMSE(df['target'], prediction, gamma, w)
                best_params=np.append(w,w0)
                best_pred=prediction
        curr_err=np.linalg.norm(curr_err-prediction)
            
    print('Iterations number is:' +str(iter_num))
    if iter_num==max_iter:
        print('The maximum number of iterations was reached.')
    print('RMSE is:'+str(minRMSE))
    print('R2 is:'+str(metrics.custom_R2(df.iloc[:,-1], best_pred)))
    
    return best_params
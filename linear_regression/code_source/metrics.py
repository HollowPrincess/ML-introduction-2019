import numpy as np

def custom_RMSE(target, pred, gamma, w):
    return np.sqrt(
        np.sum((target-pred)**2)/len(pred)+gamma*np.linalg.norm(w)
    )

def custom_R2(target, pred):
    return 1-(np.sum((target-pred)**2))/(np.sum((target-np.mean(target))**2))
import numpy as np
from multitask_learning import tca, helper

X_s, X_t, y_s, y_t = helper.load_dsads_ra_la()

def intra_class_transfer(X_s, X_t):
    classes = set(y_t)
    X_s_trans_stack = []
    X_t_trans_stack = []
    kernels = []
    for c in classes:
        mytca = tca.TCA(dim=30)
        idx_s = np.where(y_s==c)
        idx_t = np.where(y_t==c)
        X_s_cl = X_s[idx_s]
        X_t_cl = X_t[idx_t]
        
        print('Dimensions to transfer:')
        print(X_s_cl.shape, X_t_cl.shape)
        
        X_s_trans, X_t_trans, _ = mytca.fit_transform(X_s_cl, X_t_cl)
        
        X_s_trans_stack.append(X_s_cl)
        X_t_trans_stack.append(X_t_cl)
        kernels.append(mytca.get_L())
    
    return np.vstack(X_s_trans_stack), np.vstack(X_t_trans_stack)

X_s_tra, X_t_tra, kernels = intra_class_transfer(X_s, X_t)
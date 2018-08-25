import numpy as np
from scipy import io
from sklearn.preprocessing import LabelEncoder, StandardScaler
from . import tca

def load_opp_dsads():
    opp_mat = io.loadmat('./cross_opp.mat')['data_opp'][:,0:80]
    label_opp = io.loadmat('./cross_opp.mat')['data_opp'][:,459]
    dsads_mat = io.loadmat('./cross_dsads.mat')['data_dsads'][:,0:80]
    label_dsads = io.loadmat('./cross_dsads.mat')['data_dsads'][:,405]
    
    opp_mat = StandardScaler().fit_transform(opp_mat)
    dsads_mat = StandardScaler().fit_transform(dsads_mat)
    
    lenc = LabelEncoder()
    label_opp = lenc.fit_transform(label_opp)
    label_dsads = lenc.fit_transform(label_dsads)
    
    X_s = opp_mat
    X_t = dsads_mat
    y_s = label_opp
    y_t = label_dsads
    return X_s, X_t, y_s, y_t

def load_opp_dsads_right_hand():
    opp_mat = io.loadmat('./cross_opp.mat')['data_opp'][:,162:242]
    label_opp = io.loadmat('./cross_opp.mat')['data_opp'][:,459]
    dsads_mat = io.loadmat('./cross_dsads.mat')['data_dsads'][:,81:161]
    label_dsads = io.loadmat('./cross_dsads.mat')['data_dsads'][:,405]
    
    opp_mat = StandardScaler().fit_transform(opp_mat)
    dsads_mat = StandardScaler().fit_transform(dsads_mat)
    
    lenc = LabelEncoder()
    label_opp = lenc.fit_transform(label_opp)
    label_dsads = lenc.fit_transform(label_dsads)
    
    X_s = opp_mat
    X_t = dsads_mat
    y_s = label_opp
    y_t = label_dsads
    return X_s, X_t, y_s, y_t

def load_opp_rua_lua():
    rua = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,81:161]
    lua = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,243:323]
    label = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,459]
    lenc = LabelEncoder()
    label = lenc.fit_transform(label)
    
    rua = StandardScaler().fit_transform(rua)
    lua = StandardScaler().fit_transform(lua)
    
    X_s = rua
    X_t = lua
    y_s = label
    y_t = label
    return X_s, X_t, y_s, y_t

def load_opp_rla_lla():
#    [163:243],[325:405]
    rua = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,162:242]
    lua = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,324:404]
    label = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,459]
    lenc = LabelEncoder()
    label = lenc.fit_transform(label)
    
    rua = StandardScaler().fit_transform(rua)
    lua = StandardScaler().fit_transform(lua)
    
    X_s = rua
    X_t = lua
    y_s = label
    y_t = label
    return X_s, X_t, y_s, y_t

def load_dsads_ra_la():
    rua = io.loadmat('./dsads.mat')['data_dsads'][:,81:161]
    lua = io.loadmat('./dsads.mat')['data_dsads'][:,162:242]
    label = io.loadmat('./dsads.mat')['data_dsads'][:,406]
    used_cols = [2,3,4,5,6,7,9,12,18]
    idx_used = np.where(np.isin(label, used_cols))
    rua = rua[idx_used]
    lua = lua[idx_used]
    label = label[idx_used]
    
    lenc = LabelEncoder()
    label = lenc.fit_transform(label)
    X_s = rua
    X_t = lua
    y_s = label
    y_t = label
    return X_s, X_t, y_s, y_t

def load_dsads_rl_ll():
    rua = io.loadmat('./dsads.mat')['data_dsads'][:,243:323]
    lua = io.loadmat('./dsads.mat')['data_dsads'][:,324:404]
    label = io.loadmat('./dsads.mat')['data_dsads'][:,406]
    used_cols = [2,3,4,5,6,7,9,12,18]
    idx_used = np.where(np.isin(label, used_cols))
    rua = rua[idx_used]
    lua = lua[idx_used]
    label = label[idx_used]
    lenc = LabelEncoder()
    label = lenc.fit_transform(label)
    X_s = rua
    X_t = lua
    y_s = label
    y_t = label
    return X_s, X_t, y_s, y_t




def intra_class_transfer(X_s, X_t, y_s, y_t):
    classes = set(y_t)
    X_s_trans_stack = []
    X_t_trans_stack = []
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
    
    return np.vstack(X_s_trans_stack), np.vstack(X_t_trans_stack)
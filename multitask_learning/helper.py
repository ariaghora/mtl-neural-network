import numpy as np
from scipy import io
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from . import tca

def tanh_scale(X):
    return 0.5*(np.tanh(0.01 * (X - np.mean(X, axis=0)) / np.std(X, axis=0)) + 1)

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

def load_dsads_pamap():
    dsads_mat = io.loadmat('./cross_dsads.mat')['data_dsads'][:,0:80]
    label_dsads = io.loadmat('./cross_dsads.mat')['data_dsads'][:,405]
    pamap_mat = io.loadmat('./cross_pamap.mat')['data_pamap'][:,81:161]
    label_pamap = io.loadmat('./cross_pamap.mat')['data_pamap'][:,243]
    
    dsads_mat = StandardScaler().fit_transform(dsads_mat)
    pamap_mat = StandardScaler().fit_transform(pamap_mat)
    
    lenc = LabelEncoder()
    label_dsads = lenc.fit_transform(label_dsads)
    label_pamap = lenc.fit_transform(label_pamap)
    
    X_s = dsads_mat
    X_t = pamap_mat
    y_s = label_dsads
    y_t = label_pamap
    return X_s, X_t, y_s, y_t

def load_pamap_opp():
    pamap_mat = io.loadmat('./cross_pamap.mat')['data_pamap'][:,81:161]
    label_pamap = io.loadmat('./cross_pamap.mat')['data_pamap'][:,243]
    opp_mat = io.loadmat('./cross_opp.mat')['data_opp'][:,0:80]
    label_opp = io.loadmat('./cross_opp.mat')['data_opp'][:,459]
    
    # pamap_mat = MinMaxScaler().fit_transform(pamap_mat)
    # opp_mat = MinMaxScaler().fit_transform(opp_mat)
    
    
    lenc = LabelEncoder()
    label_opp = lenc.fit_transform(label_opp)
    label_pamap = lenc.fit_transform(label_pamap)
    
    X_s = opp_mat
    X_t = pamap_mat
    y_s = label_opp
    y_t = label_pamap
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

minus = 0#27*2-1
def load_opp_dsads_right_hand_test():
    opp_mat = io.loadmat('./cross_opp.mat')['data_opp'][:,162:242-minus]
    label_opp = io.loadmat('./cross_opp.mat')['data_opp'][:,459]
    dsads_mat = io.loadmat('./cross_dsads.mat')['data_dsads'][:,81:161-minus]
    label_dsads = io.loadmat('./cross_dsads.mat')['data_dsads'][:,405]
    print(opp_mat.shape)
    
    opp_mat = StandardScaler().fit_transform(opp_mat)
    dsads_mat = StandardScaler().fit_transform(dsads_mat)
    
    lenc = LabelEncoder()
    label_opp = lenc.fit_transform(label_opp)
    label_dsads = lenc.fit_transform(label_dsads)
    
    X_s = opp_mat
    X_t = dsads_mat
    y_s = label_opp
    y_t = label_dsads
    X_t, X_test, y_t, y_test = train_test_split(X_t, y_t, test_size=0.2)
    return X_s, X_t, y_s, y_t, X_test, y_test

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

def load_opp_rla_lla_test():
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
    X_t, X_test, y_t, y_test = train_test_split(X_t, y_t, test_size=0.2)
    return X_s, X_t, y_s, y_t, X_test, y_test

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

def load_dsads_ra_la_test():
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
    rua = StandardScaler().fit_transform(rua)
    lua = StandardScaler().fit_transform(lua)
    X_s = rua
    X_t = lua
    y_s = label
    y_t = label
    X_t, X_test, y_t, y_test = train_test_split(X_t, y_t, test_size=0.2)
    return X_s, X_t, y_s, y_t, X_test, y_test

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

def load_dsads_ra_t():
    rua = io.loadmat('./dsads.mat')['data_dsads'][:,81:161]
    lua = io.loadmat('./dsads.mat')['data_dsads'][:,0:80]
    label = io.loadmat('./dsads.mat')['data_dsads'][:,406]
    used_cols = [2,3,4,5,6,7,9,12,18]
    idx_used = np.where(np.isin(label, used_cols))
    rua = rua[idx_used]
    lua = lua[idx_used]
    label = label[idx_used]
    
    lenc = LabelEncoder()
    label = lenc.fit_transform(label)
    rua = StandardScaler().fit_transform(rua)
    lua = StandardScaler().fit_transform(lua)
    X_s = rua
    X_t = lua
    y_s = label
    y_t = label
    return X_s, X_t, y_s, y_t

def load_pamap_h_c():
    rua = io.loadmat('./pamap.mat')['data_pamap'][:,0:80]
    lua = io.loadmat('./pamap.mat')['data_pamap'][:,81:161]
    label = io.loadmat('./pamap.mat')['data_pamap'][:,243]
    
    lenc = LabelEncoder()
    label = lenc.fit_transform(label)
    rua = StandardScaler().fit_transform(rua)
    lua = StandardScaler().fit_transform(lua)
    X_s = rua
    X_t = lua
    y_s = label
    y_t = label
    return X_s, X_t, y_s, y_t

def load_opp_rla_t():
    rua = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,162:242]
    lua = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,0:80]
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

def load_opp_rua_t():
    rua = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,81:161]
    lua = io.loadmat('./opp_loco.mat')['data_opp_loco'][:,0:80]
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

#%%
def load_c_a():
    c = io.loadmat('./datasets/zscore/Caltech10_zscore_SURF_L10.mat')
    a = io.loadmat('./datasets/zscore/amazon_zscore_SURF_L10.mat')
    
    X_s = c['Xt']
    y_s = c['Yt']
    X_t = a['Xt']
    y_t = a['Yt']
    
    X_s = MinMaxScaler().fit_transform(X_s)
    X_t = MinMaxScaler().fit_transform(X_t)
    y_s = y_s.reshape(1, len(y_s))[0]
    y_t = y_t.reshape(1, len(y_t))[0]
    
    lenc = LabelEncoder()
    y_s = lenc.fit_transform(y_s)
    y_t = lenc.fit_transform(y_t)
    
    return X_s, X_t, y_s, y_t

def load_c_w():
    c = io.loadmat('./datasets/zscore/Caltech10_zscore_SURF_L10.mat')
    w = io.loadmat('./datasets/zscore/webcam_zscore_SURF_L10.mat')
    
    X_s = c['Xt']
    y_s = c['Yt']
    X_t = w['Xt']
    y_t = w['Yt']
    
    X_s = MinMaxScaler().fit_transform(X_s)
    X_t = MinMaxScaler().fit_transform(X_t)
    y_s = y_s.reshape(1, len(y_s))[0]
    y_t = y_t.reshape(1, len(y_t))[0]
    
    lenc = LabelEncoder()
    y_s = lenc.fit_transform(y_s)
    y_t = lenc.fit_transform(y_t)
    return X_s, X_t, y_s, y_t
    
def load_c_d():
    c = io.loadmat('./datasets/zscore/Caltech10_zscore_SURF_L10.mat')
    d = io.loadmat('./datasets/zscore/dslr_zscore_SURF_L10.mat')
    
    X_s = c['Xt']
    y_s = c['Yt']
    X_t = d['Xs']
    y_t = d['Ys']
    
    X_s = MinMaxScaler().fit_transform(X_s)
    X_t = MinMaxScaler().fit_transform(X_t)
    y_s = y_s.reshape(1, len(y_s))[0]
    y_t = y_t.reshape(1, len(y_t))[0]
    
    lenc = LabelEncoder()
    y_s = lenc.fit_transform(y_s)
    y_t = lenc.fit_transform(y_t)
    
    return X_s, X_t, y_s, y_t

def load_c_w_sub():
    (X_s, X_t, y_s, y_t) = load_c_w()
    idx_w_o_7 = np.where(y_t[y_t!=7])
    X_t = X_t[idx_w_o_7]
    y_t = y_t[idx_w_o_7]
    return X_s, X_t, y_s, y_t

def load_a_c():
    (X_s, X_t, y_s, y_t) = load_c_a()
    return (X_t, X_s, y_t, y_s)

def load_a_w():
    (X_s, _, y_s, _) = load_a_c()
    (_, X_t, _, y_t) = load_c_w()
    return (X_s, X_t, y_s, y_t)

def load_a_d():
    (X_s, _, y_s, _) = load_a_c()
    (_, X_t, _, y_t) = load_c_d()
    return (X_s, X_t, y_s, y_t)

def load_w_c():
    (X_s, X_t, y_s, y_t) = load_c_w()
    return (X_t, X_s, y_t, y_s)

def load_w_a():
    (X_s, X_t, y_s, y_t) = load_a_w()
    return (X_t, X_s, y_t, y_s)

def load_w_d():
    (X_s, _, y_s, _) = load_w_a()
    (_, X_t, _, y_t) = load_a_d()
    return (X_s, X_t, y_s, y_t)

#%%

#
#def intra_class_transfer(X_s, X_t, y_s, y_t):
#    classes = set(y_t)
#    X_s_trans_stack = []
#    X_t_trans_stack = []
#    for c in classes:
#        mytca = tca.TCA(dim=30)
#        idx_s = np.where(y_s==c)
#        idx_t = np.where(y_t==c)
#        X_s_cl = X_s[idx_s]
#        X_t_cl = X_t[idx_t]
#        
#        print('Dimensions to transfer:')
#        print(X_s_cl.shape, X_t_cl.shape)
#        
#        X_s_trans, X_t_trans, _ = mytca.fit_transform(X_s_cl, X_t_cl)
#        
#        X_s_trans_stack.append(X_s_cl)
#        X_t_trans_stack.append(X_t_cl)
#    
#    return np.vstack(X_s_trans_stack), np.vstack(X_t_trans_stack
#                    
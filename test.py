from multitask_learning.MultitaskNN import MultitaskNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy import io

import numpy as np
np.set_printoptions(precision=2, formatter={'float': lambda x: "{0:0.3f}".format(x)})
#%%

opp_mat = io.loadmat('cross_opp.mat')['data_opp'][:,0:80]
label_opp = io.loadmat('cross_opp.mat')['data_opp'][:,459]
dsads_mat = io.loadmat('cross_dsads.mat')['data_dsads'][:,0:80]
label_dsads = io.loadmat('cross_dsads.mat')['data_dsads'][:,405]

opp_mat = StandardScaler().fit_transform(opp_mat)
dsads_mat = StandardScaler().fit_transform(dsads_mat)

#%%
clf = MultitaskNN(learning_rate=1.0, batch_size=64)
clf.fit(opp_mat, label_opp, max_iter=1500)

# clf = MLPClassifier()
# clf.fit(opp_mat, label_opp)

predictions = clf.predict(dsads_mat)
print('Accuracy: ',accuracy_score(label_dsads, predictions))

#%%
pred = clf.predict_proba(dsads_mat).T
for p, l, g in zip(pred, predictions, label_dsads.reshape(len(label_dsads), 1)):
    print(p, np.max(p), l, g)


#%% INSTANCE SELECTION
v= []
for i,p in enumerate(pred):
    conf = np.max(p)
    if conf>0.90:
        print(conf)
        v.append(i)
print(len(v),len(pred)) # index of selected instance in target domain

#%%
sel_dsads_mat = dsads_mat[v, :]
sel_label_dsads = label_dsads[v]
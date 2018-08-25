from multitask_learning.MultitaskNN import MultitaskNN
from multitask_learning import helper, tca
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing
from scipy import io

import numpy as np
np.set_printoptions(precision=2, formatter={'float': lambda x: "{0:0.3f}".format(x)})
#%%
X_s, X_t, y_s, y_t = helper.load_opp_dsads()

#pca = KernelPCA(n_components=30)
#X_s = pca.fit_transform(X_s)
#X_t = pca.fit_transform(X_t)

X_s, X_t, y_s, y_t = helper.load_opp_rla_lla()

X_t, X_t_init, y_t, y_t_init = train_test_split(X_t, y_t, test_size=len(set(y_t))*3, stratify=y_t)
#%% INITIAL LEARNING

# external classifier
clf_t = RandomForestClassifier(n_estimators=64).fit(X_t_init, y_t_init)
pred = clf_t.predict(X_t)
print(accuracy_score(y_t, pred))

clf = MultitaskNN(learning_rate=1, nn_hidden=128, batch_size=128)
clf.fit(X_s, X_t_init, y_s, y_t_init, max_iter=1000)


#%% **INITIAL** INSTANCE SELECTION FOR UNLABELED TARGET DOMAIN
# predicted using classifier for task 1, since initially task 2 is untrained and 
# totally random
proba = clf.predict_proba(X_t, 1)+0.7*clf.predict_proba(X_t, 2)+0.9*clf_t.predict_proba(X_t).T
predictions = np.argmax(proba, axis=0)

print('Accuracy: ',accuracy_score(y_t, predictions))
#pred = clf.predict_proba(X_t, 1).T
pred = (proba).T
v= []
for i,p in enumerate(pred):
    conf = np.max(p/sum(p))
    if conf>0.80:
        v.append(i)
print(len(v),len(pred)) # index of selected instance in target domain

#%% RE-TRAINING NET BY INCLUDING SELECTED TARGET DATA WITH CORRESPONDING PREDICTED LABEL
# AND intra-class transfer
sel_X_t = np.vstack([X_t_init, X_t[v, :]])
sel_y_t = np.concatenate([y_t_init, predictions[v]])

#X_s_tra, X_t_tra = helper.intra_class_transfer(X_s, sel_X_t, y_s ,sel_y_t)

#clf = MultitaskNN(learning_rate=1.0, nn_hidden=128, batch_size=128)

#clf.fit(X_s_tra, X_t_tra, y_s, sel_y_t, max_iter=1000, warm_start=False)
clf.fit(X_s, sel_X_t, y_s, sel_y_t, max_iter=1000, warm_start=True)


#%% INSTANCE SELECTION FOR UNLABELED TARGET DOMAIN, i>1

proba = clf.predict_proba(X_t, 1)+0.7*clf.predict_proba(X_t, 2)+0.9*clf_t.predict_proba(X_t).T
predictions = np.argmax(proba, axis=0)

print('Accuracy: ',accuracy_score(y_t, predictions))
#pred = clf.predict_proba(X_t, 1).T
pred = (proba).T
v= []
for i,p in enumerate(pred):
    conf = np.max(p/sum(p))
#    print(conf)
    if conf>0.85:
        v.append(i)
print(len(v),len(pred)) # index of selected instance in target domain


#%%
clf_t = RandomForestClassifier().fit(X_t_init, y_t_init)
pred = clf_t.predict(X_t)
print(accuracy_score(y_t, pred))

clf_t = MLPClassifier().fit(X_t_init, y_t_init)
pred = clf_t.predict(X_t)
print(accuracy_score(y_t, pred))

clf_t = DecisionTreeClassifier().fit(X_t_init, y_t_init)
pred = clf_t.predict(X_t)
print(accuracy_score(y_t, pred))
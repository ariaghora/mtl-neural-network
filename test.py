from multitask_learning.MultitaskNN import MultitaskNN
from multitask_learning import helper, tca, encolearning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA, PCA
from sklearn import preprocessing
from scipy import io

import numpy as np
np.set_printoptions(precision=2, formatter={'float': lambda x: "{0:0.3f}".format(x)})
#%%
X_s, X_t, y_s, y_t = helper.load_opp_dsads()


X_s, X_t, y_s, y_t, X_test, y_test = helper.load_opp_rla_lla_test()

# getting n unlabeled data
n = 1
X_t, X_t_init, y_t, y_t_init = train_test_split(X_t, y_t, test_size=len(set(y_t))*n, stratify=y_t)
#%% INITIAL LEARNING

# external classifier
clf_t = RandomForestClassifier(n_estimators=64).fit(X_t_init, y_t_init)
# check the accuracy on test data (in practice, we are not supposed to know about this acc)
pred = clf_t.predict(X_test)
print(accuracy_score(y_test, pred))

clf = MultitaskNN(learning_rate=1, nn_hidden=128, batch_size=128)
clf.fit(X_s, X_t_init, y_s, y_t_init, max_iter=1000)


#%% **INITIAL** INSTANCE SELECTION FOR UNLABELED TARGET DOMAIN
# predicted using classifier for task 1, since initially task 2 is untrained and 
# totally random
proba = clf.predict_proba(X_t, 1)+0.7*clf.predict_proba(X_t, 2)+0.9*clf_t.predict_proba(X_t).T
predictions = np.argmax(proba, axis=0)

# check the accuracy on test data (in practice, we are not supposed to know about this acc)
proba_test = clf.predict_proba(X_test, 1)+0.7*clf.predict_proba(X_test, 2)+0.9*clf_t.predict_proba(X_test).T
predictions_test = np.argmax(proba_test, axis=0)
print('Accuracy: ',accuracy_score(y_test, predictions_test))


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

#clf = MultitaskNN(learning_rate=1.0, nn_hidden=128, batch_size=128)

clf.fit(X_s, sel_X_t, y_s, sel_y_t, max_iter=1000, warm_start=True)


#%%
X_s_t = np.vstack([X_s, X_t_init])
y_s_t = np.concatenate([y_s, y_t_init])

#clf_t = RandomForestClassifier().fit(X_t_init, y_t_init)
#pred = clf_t.predict(X_test)
#print(accuracy_score(y_test, pred))
#
#clf_t = MLPClassifier().fit(X_t_init, y_t_init)
#pred = clf_t.predict(X_test)
#print(accuracy_score(y_test, pred))
#
#clf_t = DecisionTreeClassifier().fit(X_t_init, y_t_init)
#pred = clf_t.predict(X_test)
#print(accuracy_score(y_test, pred))

clf_sup = RandomForestClassifier().fit(X_s_t, y_s_t)
pred = clf_sup.predict(X_test)
print(accuracy_score(y_test, pred))

clf_sup = MLPClassifier().fit(X_s_t, y_s_t)
pred = clf_sup.predict(X_test)
print(accuracy_score(y_test, pred))

clf_sup = DecisionTreeClassifier().fit(X_s_t, y_s_t)
pred = clf_sup.predict(X_test)
print(accuracy_score(y_test, pred))

#%% ENCO learning
clf_sup = encolearning.EnCoLearning(iteration=20).fit(X_s_t, y_s_t, X_t)
pred = clf_sup.predict(X_test)
print(accuracy_score(y_test, pred))


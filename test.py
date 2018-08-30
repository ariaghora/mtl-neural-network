from multitask_learning.MultitaskNN import MultitaskNN, MultitaskSS
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
n = 5
X_t, X_t_init, y_t, y_t_init = train_test_split(X_t, y_t, test_size=len(set(y_t))*n, stratify=y_t)

#%% Multitask neural net

multitask_SS = MultitaskSS(X_s, X_t, y_s, X_t_init, y_t_init, X_test, y_test, need_expert=True, alpha=0.8, beta=0.9, gamma=0.6)
multitask_SS.prepare()
multitask_SS.advance(3, relabel=True)

#%% Standard classifier with only target domain data

clf_t = RandomForestClassifier().fit(X_t_init, y_t_init)
pred = clf_t.predict(X_test)
print(accuracy_score(y_test, pred))

clf_t = MLPClassifier().fit(X_t_init, y_t_init)
pred = clf_t.predict(X_test)
print(accuracy_score(y_test, pred))

clf_t = DecisionTreeClassifier().fit(X_t_init, y_t_init)
pred = clf_t.predict(X_test)
print(accuracy_score(y_test, pred))

clf_sup = encolearning.EnCoLearning(iteration=20).fit(X_t_init, y_t_init, X_t)
pred = clf_sup.predict(X_test)
print(accuracy_score(y_test, pred))


#%% Standard classifier with target domain data combined with source domain data
X_s_t = np.vstack([X_s, X_t_init])
y_s_t = np.concatenate([y_s, y_t_init])

clf_sup = RandomForestClassifier().fit(X_s_t, y_s_t)
pred = clf_sup.predict(X_test)
print(accuracy_score(y_test, pred))

clf_sup = MLPClassifier().fit(X_s_t, y_s_t)
pred = clf_sup.predict(X_test)
print(accuracy_score(y_test, pred))

clf_sup = DecisionTreeClassifier().fit(X_s_t, y_s_t)
pred = clf_sup.predict(X_test)
print(accuracy_score(y_test, pred))

# ENCO learning
clf_sup = encolearning.EnCoLearning(iteration=20).fit(X_s_t, y_s_t, X_t)
pred = clf_sup.predict(X_test)
print(accuracy_score(y_test, pred))
"""
Created on Mon Aug 27 16:11:25 2018

@author: hello

Implementation En-co learning algorithm in "Activity Recognition Based on 
Semi-supervised Learning", proposed by Donghai Guan, Weiwei Yuan, Young-Koo Lee,
Andrey Gavrilov and Sungyoung Lee (2017)
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

class EnCoLearning:
    def __init__(self, u=270, nc=4, iteration=20):
        self.u = u
        self.nc = 4
        self.iteration = iteration
    
    def fit(self, X_l, y_l, X_u):
        X_l_tr = X_l
        y_l_tr = y_l
        for i in range(self.iteration):     
            
            self.h1 = DecisionTreeClassifier().fit(X_l_tr, y_l_tr)
            self.h2 = GaussianNB().fit(X_l_tr, y_l_tr)
            self.h3 = KNeighborsClassifier(n_neighbors=3).fit(X_l_tr, y_l_tr)
            
            np.random.shuffle(X_u)
            X_u_a, X_u = X_u[:self.u - 1], X_u[self.u - 1:]
            
            pred_labels = np.array([self.h1.predict(X_u_a),
                                    self.h2.predict(X_u_a),
                                    self.h3.predict(X_u_a)]).T
            final_labels = stats.mode(pred_labels, axis=1)[0].T[0]
            X_l_tr = np.vstack([X_l_tr, X_u_a])
            y_l_tr = np.concatenate([y_l_tr, final_labels])
            
            if X_u_a.shape[0]<self.u:
                break
        return self
    
    def predict(self, X):
        pred_labels = np.array([self.h1.predict(X),
                                    self.h2.predict(X),
                                    self.h3.predict(X)]).T
        final_labels = stats.mode(pred_labels, axis=1)[0].T[0]
        return final_labels
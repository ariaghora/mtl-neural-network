import numpy as np
import time
import matplotlib.pyplot as plt

from toolz import itertoolz
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, KernelPCA

from libtlda.flda import FeatureLevelDomainAdaptiveClassifier
from libtlda.suba import SubspaceAlignedClassifier

from numba import jit


# sigmoid activation function
def sig(z):
    return 1/(1+(np.exp(-z)))

def sig_der(z):
    return sig(z) * (1 - sig(z))

# ReLU activation function
# @jit(nopython=True)
def relu(z):
    return z * (z > 0)

# @jit(nopython=True)
def relu_der(z):
    return 1 * (z > 0)

# @jit(nopython=True)
def loss(y, y_pred):
    l_sum = np.sum(np.multiply(y, np.log(y_pred)))#-reg
    m = y.shape[1]
    l = -(1/m)*l_sum
    return l

@jit(nopython=True)
def fast_mul(A, B):
    return A @ B

class MultitaskNN:
    def __init__(self, nn_hidden=64, learning_rate=0.07, batch_size=64, T=1.5):
        self.learning_rate = learning_rate
        self.nn_hidden = nn_hidden
        self.batch_size = batch_size
        self.T = T

    def fit(self, X, X_tar, y, y_tar, max_iter=500, warm_start=False, use_dropout=False):
        m = X.shape[0]
        n_x = X.shape[1]
        
        
        n_class_src = len(set(y))
        if len(set(y_tar)) > 0:
            n_class_tar = len(set(y_tar))
            m_tar = X_tar.shape[0]


        if not warm_start:
            ''' weight and bias initialization'''
            # shared weights
            self.W1 = np.random.randn(self.nn_hidden, n_x)
            self.b1 = np.zeros((self.nn_hidden,1))
            
            # task 1 specific weights
            self.W2_1 = np.random.randn(n_class_src, self.nn_hidden)
            self.b2_1 = np.zeros((n_class_src,1))
            
            # task 2 specific weights
            self.W2_2 = np.random.randn(n_class_src, self.nn_hidden)
            self.b2_2 = np.zeros((n_class_src,1))

        X_shuf, y_shuf = shuffle(X, y)
        
        if len(y_tar)>0:
            X_tar_shuf, y_tar_shuf = shuffle(X_tar, y_tar)

        le = LabelBinarizer()
        le.fit(list(y)+list(y_tar))
        
        if len(y_tar)>0:
            le_tar = LabelBinarizer()
            le_tar.fit(y_tar)

        bs = np.min([self.batch_size, X_shuf.shape[0]])
        batches_X = np.array_split(X_shuf, m/bs)
        batches_y = np.array_split(y_shuf, m/bs)
        tasks_1 = [1 for i in range(len(batches_y))]

        batches_X_tar = np.array([])
        batches_y_tar = np.array([])
        if len(y_tar)>0:
            batches_X_tar = np.array_split(X_tar_shuf, max(1, m_tar/self.batch_size))
            batches_y_tar = np.array_split(y_tar_shuf, max(1, m_tar/self.batch_size))
        tasks_2 = [2 for i in range(len(batches_y_tar))]

        
        # TO DO: hstack source and target batches in alternating way
        all_batches_X = list(itertoolz.interleave([batches_X, batches_X_tar]))[::-1]
        all_batches_y = list(itertoolz.interleave([batches_y, batches_y_tar]))[::-1]
        all_tasks = list(itertoolz.interleave([tasks_1, tasks_2]))[::-1]
        
        start = time.time()
        
        loss_src = []
        loss_tar = []
        for j in range(1, max_iter + 1):#progressbar.progressbar(range(max_iter)):
            batch_errors_tar = []
            batch_errors_src = []
            
            for i in range(len(all_batches_X)):
                task = all_tasks[i]
                X_new = all_batches_X[i].T
                y_new = all_batches_y[i]
                y_new = le.transform(y_new)
                y_new = y_new.T
                Z1 = fast_mul(self.W1, X_new)+self.b1
                A1 = relu(Z1)
                
                if use_dropout:
                    dropout_percent = 0.2
                    A1 *= np.random.binomial([np.ones((len(Z1), A1.shape[1]))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

                if task == 1:
                    Z2 = fast_mul(self.W2_1, A1)+self.b2_1
                    A2 = np.nan_to_num(np.nan_to_num(np.exp(Z2/self.T))/np.nan_to_num(np.sum(np.exp(Z2/self.T),axis=0)))

                    cost = loss(y_new, A2)

                    dZ2 = A2-y_new                    
                    
                    dW2 = (1./m) * fast_mul(dZ2, A1.T)
                    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

                    
                    dA1 = fast_mul(self.W2_1.T, dZ2)
                    dZ1 = dA1 * relu_der(Z1) 
                    
                    dW1 = (1./m) * fast_mul(dZ1, X_new.T)
                    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)
                    
                    self.W2_1 = self.W2_1 - self.learning_rate * dW2
                    self.b2_1 = self.b2_1 - self.learning_rate * db2
                    
                    batch_errors_src.append(cost)
                
                if task == 2:
                    Z2 = np.matmul(self.W2_2, A1)+self.b2_2
                    A2 = np.nan_to_num(np.nan_to_num(np.exp(Z2/self.T))/np.nan_to_num(np.sum(np.exp(Z2/self.T),axis=0)))

                    cost_tar = loss(y_new, A2)

                    dZ2 = A2-y_new
                    
#                    print(penalty_tar)
                    dW2 = (1./m) * fast_mul(dZ2, A1.T)
                    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

                    dA1 = fast_mul(self.W2_1.T, dZ2)
                    dZ1 = dA1 * relu_der(Z1) 
                    dW1 = (1./m) * fast_mul(dZ1, X_new.T)
                    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

                    self.W2_2 = self.W2_2 - self.learning_rate * dW2
                    self.b2_2 = self.b2_2 - self.learning_rate * db2
                    
                    batch_errors_tar.append(cost_tar)
                
                self.W1 = self.W1 - self.learning_rate * dW1
                self.b1 = self.b1 - self.learning_rate * db1

            loss_src.append(np.mean(batch_errors_src))
            loss_tar.append(np.mean(batch_errors_tar))
            if (j%100==0):
                print("Target %s loss: %s"%(j, np.mean(batch_errors_tar)))
                print("Source %s loss: %s"%(j, np.mean(batch_errors_src)))
            
        end = time.time()
        print(end-start)
        
        plt.figure()
        plt.plot(loss_src, label='source')
        plt.plot(loss_tar, label='target')
        plt.legend()
        plt.show()
        return self
    
    def predict_proba(self, X, task):
        Z1 = np.matmul(self.W1, X.T)+self.b1
        A1 = relu(Z1)

        if task == 1:
            Z2 = np.matmul(self.W2_1, A1)+self.b2_1
            A2 = np.exp(Z2)/np.sum(np.exp(Z2),axis=0)
        if task == 2:
            Z2 = np.matmul(self.W2_2, A1)+self.b2_2
            A2 = np.exp(Z2)/np.sum(np.exp(Z2),axis=0)
        return A2
    
    def predict(self, X, task):
        return np.argmax(self.predict_proba(X, task), axis=0)
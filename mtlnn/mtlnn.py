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

from tqdm import tqdm

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
    def __init__(self, nn_hidden=64, learning_rate=0.07, batch_size=200, T=1.5, 
                 dropout_percent=0.4):
        self.learning_rate = learning_rate
        self.nn_hidden = nn_hidden
        self.batch_size = batch_size
        self.T = T
        self.dropout_percent=dropout_percent

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
        for j in tqdm(range(1, max_iter + 1)):#progressbar.progressbar(range(max_iter)):
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
                    A1 *= np.random.binomial([np.ones((len(Z1), A1.shape[1]))],1-self.dropout_percent)[0] * (1.0/(1-self.dropout_percent))

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
            # if (j%100==0):
            #     print("Target %s loss: %s"%(j, np.mean(batch_errors_tar)))
            #     print("Source %s loss: %s"%(j, np.mean(batch_errors_src)))
            
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
    
class MTT:
    def __init__(self, X_s_ori, y_s, X_t_ori, y_t, nn_hidden=30, learning_rate=0.01, 
                 batch_size=200, T=2, seed=1, alpha=0.7, dropout_percent=0.4, min_confidence=0.95, 
                 max_iter=10000, num_components=40):
        self.X_s_ori = X_s_ori
        self.X_t_ori = X_t_ori
        self.model = MultitaskNN(nn_hidden=nn_hidden, learning_rate=learning_rate, 
                                 batch_size=batch_size, T=T, dropout_percent=dropout_percent)
        
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.max_iter = max_iter
        self.num_components = num_components
        
        self.y_s = y_s
        self.y_t = y_t
        
        self.seed = seed
        self.trained = False
    
    def prepare(self, initial_target_labels=False, X_t_init=[], y_t_init=[]):
        print('Pass 1')
        np.random.seed(self.seed)
        
        suba = SubspaceAlignedClassifier()
        if not initial_target_labels:
            self.X_t_init = np.empty([0, self.X_t.shape[1]])
            self.y_t_init = np.array([])
            V, CX, self.CZ = suba.subspace_alignment(self.X_s_ori, self.X_t_ori, 
                                                num_components=self.num_components)
        else:
            assert X_t_init.shape[0]>0, 'Initial target data must not be empty'
            self.X_t_init = X_t_init
            self.y_t_init = y_t_init
            V, CX, self.CZ = suba.subspace_alignment(self.X_s_ori, np.vstack([X_t_init, self.X_t_ori]),
                                                num_components=self.num_components)
            
        V, CX, CZ = suba.subspace_alignment(self.X_s_ori, self.X_t_ori, 
                                            num_components=self.num_components)
        self.X_s = self.X_s_ori @ CX # map to principal component
        self.X_s = self.X_s @ V # align to subspace
        self.X_t = self.X_t_ori @ CZ
        
        if initial_target_labels:
            self.X_t_init = self.X_t_init @ CZ
            
        self.model.fit(self.X_s, self.X_t_init, self.y_s, self.y_t_init, 
                       warm_start=False, max_iter=self.max_iter, use_dropout=True)
        
        ## TRANSDUCTION THROUGH SOURCE-SPECIFIC NET
        pred_proba_f = self.model.predict_proba(self.X_t, 1).T
        pred_proba = (pred_proba_f)
        
        # max confidence of prediction on each instance
        proba_max = pred_proba.max(axis=1)
        
        idx_gt_threshold = np.where(proba_max > self.min_confidence)
        proba_gt_threshold = proba_max[idx_gt_threshold]
        
        self.initial_selected_num = len(proba_gt_threshold)
        print('selected: ', self.initial_selected_num)
        
        # Evaluate 1st phase transduction
        pred = (pred_proba_f).argmax(axis=1)
        acc = accuracy_score(pred, self.y_t)
        print('trans acc:', acc)
        acc_sel = accuracy_score(pred[idx_gt_threshold], self.y_t[idx_gt_threshold])
        print('trans sel acc:', acc_sel)
        
        # Select label with high confidence
        self.X_trans = np.vstack([self.X_t_init, self.X_t[idx_gt_threshold]])
        self.y_trans = np.concatenate([self.y_t_init, pred[idx_gt_threshold]])
        
        self.trained = True
    
    def advance(self, step=1):
        assert self.trained == True, "prepare() function has not been called"
        np.random.seed(self.seed)
        
        trans_accs = []
        sel_instances = []
        for i in range(step):
            print('Step ',i+1)
            self.model.fit(self.X_s, self.X_trans, self.y_s, self.y_trans, warm_start=True, max_iter=self.max_iter, use_dropout=True)
            
            alpha = self.alpha
            beta = 1 - alpha
            
            pred_proba_f = self.model.predict_proba(self.X_t, 2).T
            pred_proba_g = self.model.predict_proba(self.X_t, 1).T
            pred_proba = (alpha*pred_proba_f + beta * pred_proba_g)
            
            # max confidence of prediction on each instance
            proba_max = pred_proba.max(axis=1)
            idx_gt_threshold = np.where(proba_max > self.min_confidence)
            proba_gt_threshold = proba_max[idx_gt_threshold]
            
            # Evaluate 1st phase transduction
            pred = (alpha*pred_proba_f + beta*pred_proba_g).argmax(axis=1)
            acc = accuracy_score(pred, self.y_t)
            acc_sel = accuracy_score(pred[idx_gt_threshold], self.y_t[idx_gt_threshold])
            
            print('selected: ', len(proba_gt_threshold))
            print('trans acc:', acc)
            print('trans sel acc:', acc_sel)
            
            trans_accs.append(acc)
            sel_instances.append(len(proba_gt_threshold))
    
            # Select label with high confidence
            self.X_trans = self.X_t[idx_gt_threshold]#np.vstack([X_t_init, X_t[idx_gt_threshold]])
            self.y_trans = pred[idx_gt_threshold]
        
        plt.figure()
        plt.plot(trans_accs, label='transduction accuracy')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot([self.initial_selected_num]+sel_instances[:-1], label='selected instances')
        plt.legend()
        plt.show()
    
    def predict(self, X):
        alpha = self.alpha
        beta = 1 - alpha
            
        X_t = X @ self.CZ
        
        pred_proba_f = self.model.predict_proba(X_t, 2).T
        pred_proba_g = self.model.predict_proba(X_t, 1).T
        pred = (alpha*pred_proba_f + beta*pred_proba_g).argmax(axis=1)
        return pred
    
        
        
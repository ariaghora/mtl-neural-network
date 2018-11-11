from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy as np
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

from autograd import elementwise_grad, grad
from autograd.extend import primitive
from autograd.misc import flatten
from autograd.misc.optimizers import adam

from numba import jit

from IPython import get_ipython
def in_ipynb():
    return type(get_ipython()).__module__.startswith('ipykernel.')
    
from tqdm import tqdm, tqdm_notebook

# sigmoid activation function
def sig(z):
    return 1/(1+(np.exp(-z)))

def sig_der(z):
    return sig(z) * (1 - sig(z))

# ReLU activation function
# @jit(nopython=True)
def relu(z):
    return z * (z > 0)

#relu_der = elementwise_grad(relu)
# @jit(nopython=True)
def relu_der(z):
    return 1 * (z > 0)

# @jit(nopython=True)
#@primitive
def loss(y, y_pred):
    l_sum = np.sum(np.multiply(y, np.log(y_pred)))#-reg
    m = y.shape[1]
    l = -(1/m)*l_sum
    return l

def ce(y, y_pred):
    return -(np.multiply(y, np.log(y_pred))).mean()
    

#@jit(nopython=True)
def fast_mul(A, B):
    return A @ B

class Task:
    def __init__(self, nn_hidden, n_classes, learning_rate, m, T):
        self.W = np.random.randn(n_classes, nn_hidden)
        self.b = np.zeros((n_classes, 1))
        self.learning_rate = learning_rate
        self.T = T
        self.m = m
        self.batch_errors = []
        
    def evaluate(self, shared_layer_activation):
        self.A = shared_layer_activation
        self.Z2 = fast_mul(self.W, self.A) + self.b
        self.A2 = np.nan_to_num(np.nan_to_num(np.exp(self.Z2/self.T))/np.nan_to_num(np.sum(np.exp(self.Z2/self.T),axis=0)))

        return self.A2

    def backpropagate(self, y, shared_layer_activation):
        err = loss(y, self.A2)
        
        A = shared_layer_activation        
        
        def training_loss_W(W):
            out = W@self.A+self.b#fast_mul(W, self.A) + self.b
            softmax = np.nan_to_num(np.nan_to_num(np.exp(out/self.T))/np.nan_to_num(np.sum(np.exp(out/self.T),axis=0)))
            return ce(y, softmax)
        
        def training_loss_b(b):
            out = self.W@self.A+b#fast_mul(self.W, self.A) + b
            softmax = np.nan_to_num(np.nan_to_num(np.exp(out/self.T))/np.nan_to_num(np.sum(np.exp(out/self.T),axis=0)))
            return ce(y, softmax)
        
        loss_der_W = grad(training_loss_W)
        loss_der_b = grad(training_loss_b)
        
            
        dZ2 = self.A2-y
        print(dZ2[0][0])
        cuk()
        
        # dW2 = (1./y.shape[1]) * fast_mul(dZ2, A.T)
        dW2 = loss_der_W(self.W)
        
        # db2 = (1./y.shape[1]) * np.sum(dZ2, axis=1, keepdims=True)
        db2 = loss_der_b(self.b)

        self.W = self.W - self.learning_rate * dW2
        self.b = self.b - self.learning_rate * db2
        return dZ2, self.W, err        
        
        

class MultitaskNN:
    def __init__(self, nn_hidden=64, learning_rate=0.07, batch_size=200, T=1.5, 
                 dropout_percent=0.4, verbosity=2):
        self.learning_rate = learning_rate
        self.nn_hidden = nn_hidden
        self.batch_size = batch_size
        self.T = T
        self.dropout_percent=dropout_percent
        self.verbosity = verbosity
    
    def fit2(self, X, X_tar, y, y_tar, max_iter=500, warm_start=False, 
            use_dropout=False, desc='', regularize=True):
        m = X.shape[0]
        n_x = X.shape[1]
        
        print(n_x)
        
        n_class_src = len(set(y))
        if len(set(y_tar)) > 0:
            n_class_tar = len(set(y_tar))
            m_tar = X_tar.shape[0]


        if not warm_start:
            ''' weight and bias initialization'''
            # shared weights
            self.W1 = np.random.randn(self.nn_hidden, n_x)
            self.b1 = np.zeros((self.nn_hidden,1))
            
            # task 1 (source) specific weights
            self.task_1 = Task(self.nn_hidden, n_class_src, self.learning_rate, m, self.T)
            
            # task 2 (target) specific weights
            self.task_2 = Task(self.nn_hidden, n_class_src, self.learning_rate, m, self.T)

        X_shuf, y_shuf = shuffle(X, y)
        
        if len(y_tar)>0:
            X_tar_shuf, y_tar_shuf = shuffle(X_tar, y_tar)

        # transform labels into one-hot vectors
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
        
        def get_batch(step):
            idx = step % len(all_tasks)
            X_new = all_batches_X[idx].T
            y_new = all_batches_y[idx]
            y_new = le.transform(y_new)
            y_new = y_new.T
            return X_new, y_new
        
        def model_loss(params, step):
            W, b1, W2_1, b2_1, W2_2, b2_2 = params
            prod = W @ get_batch(step)
            nonlin = relu(prod)+b1
            if use_dropout:
                nonlin *= np.random.binomial([np.ones((len(prod), nonlin.shape[1]))],
                                              1-self.dropout_percent)[0] * (1.0/(1-self.dropout_percent))
            out = (W2 @ nonlin)+b2
            prob = np.exp(out/self.T)/np.sum(np.exp(out/self.T))
            
            # if regularize:
            #     a_bar = (flatten(self.task_1.W)[0]+flatten(self.task_2.W)[0])/2
            #     a_bar_norm = np.linalg.norm(a_bar, 2)
            #     source_norm = np.linalg.norm(flatten(self.task_1.W)[0]-a_bar, 2)
            #     tar_norm = np.linalg.norm(flatten(self.task_2.W)[0]-a_bar, 2)
                
            #     reg = a_bar_norm + 0.7 * (source_norm+tar_norm)/2
            # else:
            #     reg = 0
            
            cost = loss(y_new, prob)+0.3*reg
            return cost
    
    def fit(self, X, X_tar, y, y_tar, max_iter=500, warm_start=False, 
            use_dropout=False, desc='', regularize=True):
        m = X.shape[0]
        n_x = X.shape[1]
        
        print(n_x)
        
        n_class_src = len(set(y))
        if len(set(y_tar)) > 0:
            n_class_tar = len(set(y_tar))
            m_tar = X_tar.shape[0]


        if not warm_start:
            ''' weight and bias initialization'''
            # shared weights
            self.W1 = np.random.randn(self.nn_hidden, n_x)
            self.b1 = np.zeros((self.nn_hidden,1))
            
            # task 1 (source) specific weights
            self.task_1 = Task(self.nn_hidden, n_class_src, self.learning_rate, m, self.T)
            
            # task 2 (target) specific weights
            self.task_2 = Task(self.nn_hidden, n_class_src, self.learning_rate, m, self.T)

        X_shuf, y_shuf = shuffle(X, y)
        
        if len(y_tar)>0:
            X_tar_shuf, y_tar_shuf = shuffle(X_tar, y_tar)

        # transform labels into one-hot vectors
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
        
        progress_bar = tqdm_notebook if in_ipynb() else tqdm
        for j in progress_bar(range(1, max_iter + 1), desc=desc):#progressbar.progressbar(range(max_iter)):
            batch_errors_tar = []
            batch_errors_src = []
            for i in (range(len(all_batches_X))):
                task = all_tasks[i]
                X_new = all_batches_X[i].T
                y_new = all_batches_y[i]
                y_new = le.transform(y_new)
                y_new = y_new.T
                
                def model_loss(params, _):
                    W, b1, W2, b2 = params
                    prod = W@X_new
                    nonlin = relu(prod)+b1
                    if use_dropout:
                        nonlin *= np.random.binomial([np.ones((len(prod), nonlin.shape[1]))],
                                                      1-self.dropout_percent)[0] * (1.0/(1-self.dropout_percent))
                    out = (W2 @ nonlin)+b2
                    prob = np.exp(out/self.T)/np.sum(np.exp(out/self.T))
                    
                    if regularize:
                        a_bar = (flatten(self.task_1.W)[0]+flatten(self.task_2.W)[0])/2
                        a_bar_norm = np.linalg.norm(a_bar, 2)
                        source_norm = np.linalg.norm(flatten(self.task_1.W)[0]-a_bar, 2)
                        tar_norm = np.linalg.norm(flatten(self.task_2.W)[0]-a_bar, 2)
                        
                        reg = a_bar_norm + 0.7 * (source_norm+tar_norm)/2
                    else:
                        reg = 0
                    
                    cost = loss(y_new, prob)+0.3*reg
                    return cost
                
                if task==1:
                    cost = model_loss((self.W1, self.b1, self.task_1.W, self.task_1.b))
                    self.dW1, self.db1, self.task_1.dW, self.task_1.db = grad(model_loss)((self.W1, self.b1, self.task_1.W, self.task_1.b))
                    
                    self.W1 -= self.learning_rate * self.dW1
                    self.b1 -= self.learning_rate * self.db1
                    self.task_1.W -= self.task_1.dW
                    self.task_1.b -= self.task_1.db
                    
                    batch_errors_src.append(cost)

                
                if task==2:
                    cost = model_loss((self.W1, self.b1, self.task_2.W, self.task_2.b))
                    self.dW1, self.db1, self.task_2.dW, self.task_2.db = grad(model_loss)((self.W1, self.b1, self.task_2.W, self.task_2.b))
                    
                    self.W1 -= self.learning_rate * self.dW1
                    self.b1 -= self.learning_rate * self.db1
                    self.task_2.W -= self.task_2.dW
                    self.task_2.b -= self.task_2.db
                    
                    batch_errors_tar.append(cost)
                
                # batch normalize
                mu = np.mean(self.W1, axis=0)
                var = np.var(self.W1, axis=0)
                self.W1 = (self.W1 - mu)/np.sqrt(var+1e-8)


            loss_src.append(np.mean(batch_errors_src))
            loss_tar.append(np.mean(batch_errors_tar))
            
        end = time.time()
        # print(end-start)
        
        if self.verbosity > 2:
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
            A2 = self.task_1.evaluate(A1)
        if task == 2:
            A2 = self.task_2.evaluate(A1)
        return A2
    
    def predict(self, X, task):
        return np.argmax(self.predict_proba(X, task), axis=0)
    
class MTT:
    def __init__(self, X_s_ori, y_s, X_t_ori, y_t, nn_hidden=30, learning_rate=0.01, 
                 batch_size=200, T=2, seed=1, alpha=0.7, beta=0.7, dropout_percent=0.4, min_confidence=0.95, 
                 max_iter=10000, num_components=40, use_expert=False, domain_adaptation=True,
                 verbosity=0):
        self.X_s_ori = X_s_ori
        self.X_t_ori = X_t_ori
        self.model = MultitaskNN(nn_hidden=nn_hidden, learning_rate=learning_rate, 
                                 batch_size=batch_size, T=T, dropout_percent=dropout_percent,
                                 verbosity=verbosity)
        
        self.alpha = alpha
        self.beta = beta
        self.min_confidence = min_confidence
        self.max_iter = max_iter
        self.num_components = num_components
        self.use_expert = use_expert
        self.domain_adaptation = domain_adaptation
        self.verbosity = verbosity
        
        self.y_s = y_s
        self.y_t = y_t
        
        self.seed = seed
        self.trained = False
    
    def prepare(self, initial_target_labels=False, X_t_init=[], y_t_init=[]):
        np.random.seed(self.seed)
        self.initial_target_labels = initial_target_labels
        
        suba = SubspaceAlignedClassifier()
        if not self.initial_target_labels:
            self.X_t_init = np.empty([0, self.X_t_ori.shape[1]])
            self.y_t_init = np.array([])
            
            if self.domain_adaptation:
                V, CX, self.CZ = suba.subspace_alignment(self.X_s_ori, self.X_t_ori, 
                                                    num_components=self.num_components)
        else:
            assert X_t_init.shape[0]>0, 'Initial target data must not be empty'
            self.X_t_init = X_t_init
            self.y_t_init = y_t_init
            if self.domain_adaptation:
                V, CX, self.CZ = suba.subspace_alignment(self.X_s_ori, np.vstack([X_t_init, self.X_t_ori]),
                                                    num_components=self.num_components)
            
        # V, CX, CZ = suba.subspace_alignment(self.X_s_ori, self.X_t_ori, 
        #                                     num_components=self.num_components)
        
        if self.domain_adaptation:
            self.X_s = self.X_s_ori @ CX # map to principal component
            self.X_s = self.X_s @ V # align to subspace
            self.X_t = self.X_t_ori @ self.CZ
        else:
            self.X_s = self.X_s_ori
            self.X_t = self.X_t_ori
        
        if self.initial_target_labels and self.domain_adaptation:
            self.X_t_init = self.X_t_init @ self.CZ
            
        if self.use_expert:
            assert initial_target_labels, "to use expert classifier, initial target labels must be true"
            self.expert = RandomForestClassifier(n_estimators=64).fit(self.X_t_init, self.y_t_init)
            
        self.model.fit(self.X_s, self.X_t_init, self.y_s, self.y_t_init, 
                       warm_start=False, max_iter=self.max_iter, use_dropout=True,
                       desc='Preparing', regularize=False)
        
        ## TRANSDUCTION THROUGH SOURCE-SPECIFIC NET
        pred_proba_f = self.model.predict_proba(self.X_t, 1).T
        pred_proba = (pred_proba_f)
        
        if self.initial_target_labels:
            alpha = self.alpha
            pred_proba_g = self.model.predict_proba(self.X_t, 2).T
            pred_proba = alpha * pred_proba_f + (1-alpha) * pred_proba_g
            if self.use_expert:
                pred_proba_expert = self.expert.predict_proba(self.X_t)
                pred_proba = ((1-self.beta)*pred_proba + (self.beta)*pred_proba_expert)
                       
        
        self.p=pred_proba
        
        # max confidence of prediction on each instance
        proba_max = pred_proba.max(axis=1)
        
        idx_gt_threshold = np.where(proba_max > self.min_confidence)
        proba_gt_threshold = proba_max[idx_gt_threshold]
        
        self.initial_selected_num = len(proba_gt_threshold)
        
        # Evaluate 1st phase transduction
        pred = (pred_proba_f).argmax(axis=1)
        acc = accuracy_score(pred, self.y_t)
        acc_sel = accuracy_score(pred[idx_gt_threshold], self.y_t[idx_gt_threshold])

        if self.verbosity > 1:
            print('selected      : ', self.initial_selected_num)        
            print('trans acc     :', acc)
            print('trans sel acc :', acc_sel)
        
        # Select label with high confidence
        if self.initial_target_labels:
            self.X_trans = np.vstack([self.X_t_init, self.X_t[idx_gt_threshold]])
            self.y_trans = np.concatenate([self.y_t_init, pred[idx_gt_threshold]])
        else:
            self.X_trans = self.X_t[idx_gt_threshold]
            self.y_trans = pred[idx_gt_threshold]
            
        self.trained = True
    
    def advance(self, step=1, warm_start=True, total_relabeling=True):
        assert self.trained == True, "prepare() function has not been called"
        np.random.seed(self.seed)
        
        trans_accs = []
        sel_instances = []
        for i in range(step):
            if self.use_expert:
                self.expert = RandomForestClassifier(n_estimators=64).fit(self.X_trans, self.y_trans)
                
            self.model.fit(self.X_s, self.X_trans, self.y_s, self.y_trans, 
                           warm_start=warm_start, max_iter=self.max_iter, use_dropout=True,
                           desc='Step '+str(i+1), regularize=True)
            
            alpha = self.alpha
            
            pred_proba_f = self.model.predict_proba(self.X_t, 2).T
            pred_proba_g = self.model.predict_proba(self.X_t, 1).T
            pred_proba = (alpha*pred_proba_f + (1 - alpha) * pred_proba_g)
            
            if self.use_expert:
                pred_proba_expert = self.expert.predict_proba(self.X_t)
                pred_proba = ((1-self.beta)*pred_proba + (self.beta)*pred_proba_expert)
            
            # max confidence of prediction on each instance
            proba_max = pred_proba.max(axis=1)
            idx_gt_threshold = np.where(proba_max > self.min_confidence)
            proba_gt_threshold = proba_max[idx_gt_threshold]
            
            # secondary confidence as the output
            idx_gt_threshold_sec = np.where(proba_max > self.min_confidence - 0.2)
            
            # Evaluate 1st phase transduction
            pred = (pred_proba).argmax(axis=1)
            acc = accuracy_score(self.predict(self.X_t_ori), self.y_t)
            acc_sel = accuracy_score(pred[idx_gt_threshold], self.y_t[idx_gt_threshold])
            
            
            if self.verbosity > 1:
                print('selected      : ', len(proba_gt_threshold))
                print('trans acc     :', acc)
                print('trans sel acc :', acc_sel)
            
            trans_accs.append(acc)
            sel_instances.append(len(proba_gt_threshold))
    
            # Select label with high confidence
            if self.initial_target_labels:
                self.X_trans = np.vstack([self.X_t_init, self.X_t[idx_gt_threshold]])
                self.y_trans = np.concatenate([self.y_t_init, pred[idx_gt_threshold]])
            else:
                self.X_trans = self.X_t[idx_gt_threshold]
                self.y_trans = pred[idx_gt_threshold]
                
            if not total_relabeling:
                exclude = np.where(proba_max <= self.min_confidence)
                self.X_t = self.X_t[exclude]
                print(self.X_t.shape[0])
            
            self.X_out = self.X_s_ori[idx_gt_threshold_sec]
            self.y_out = self.pred[idx_gt_threshold_sec]
        
        if self.verbosity > 2:
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
        
        if self.domain_adaptation:
            X_t = X @ self.CZ
        else:
            X_t = X
        
        pred_proba_f = self.model.predict_proba(X_t, 2).T
        pred_proba_g = self.model.predict_proba(X_t, 1).T
        pred_proba = alpha*pred_proba_f + (1 - alpha) * pred_proba_g
        
        if self.use_expert:
            pred_proba_expert = self.expert.predict_proba(X_t)
            pred_proba = ((1-self.beta)*pred_proba + self.beta*pred_proba_expert)
        
        pred = (pred_proba).argmax(axis=1)
        return pred
    
        
        
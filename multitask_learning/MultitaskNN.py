import numpy as np

from toolz import itertoolz
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# sigmoid activation function
def sig(z):
    return 1/(1+(np.exp(-z)))

def loss(y, y_pred):
    l_sum = np.sum(np.multiply(y, np.log(y_pred)))
    m = y.shape[1]
    l = -(1/m)*l_sum
    return l

class MultitaskNN:
    def __init__(self, nn_hidden=64, learning_rate=0.5, batch_size=64):
        self.learning_rate = learning_rate
        self.nn_hidden = nn_hidden
        self.batch_size = batch_size

    def fit(self, X, X_tar, y, y_tar, max_iter=500, warm_start=False):
        m = X.shape[0]
        m_tar = X_tar.shape[0]

        n_x = X.shape[1]
        n_class = len(set(y))

        if not warm_start:
            ''' weight and bias initialization'''
            # shared weights
            self.W1 = np.random.randn(self.nn_hidden, n_x)
            self.b1 = np.zeros((self.nn_hidden,1))
            
            # task 1 specific weights
            self.W2_1 = np.random.randn(n_class, self.nn_hidden)
            self.b2_1 = np.zeros((n_class,1))
            
            # task 2 specific weights
            self.W2_2 = np.random.randn(n_class, self.nn_hidden)
            self.b2_2 = np.zeros((n_class,1))

        X_shuf, y_shuf = shuffle(X, y)
        X_tar_shuf, y_tar_shuf = shuffle(X_tar, y_tar)

        le = LabelBinarizer()
        le.fit(y)
        
        if len(y_tar)>0:
            le_tar = LabelBinarizer()
            le_tar.fit(y_tar)

        batches_X = np.array_split(X_shuf, m/self.batch_size)
        batches_y = np.array_split(y_shuf, m/self.batch_size)
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


        for j in range(max_iter):
            batch_errors = []

            for i in range(len(all_batches_X)):
                task = all_tasks[i]
                X_new = all_batches_X[i].T
                y_new = all_batches_y[i]
                y_new = le.transform(y_new)
                y_new = y_new.T
                Z1 = np.matmul(self.W1, X_new)+self.b1
                A1 = sig(Z1)

                if task == 1:
                    Z2 = np.matmul(self.W2_1, A1)+self.b2_1
                    A2 = np.exp(Z2)/np.sum(np.exp(Z2),axis=0)

                    cost = loss(y_new, A2)

                    dZ2 = A2-y_new
                    dW2 = (1./m) * np.matmul(dZ2, A1.T)
                    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

                    dA1 = np.matmul(self.W2_1.T, dZ2)
                    dZ1 = dA1 * sig(Z1) * (1 - sig(Z1))
                    dW1 = (1./m) * np.matmul(dZ1, X_new.T)
                    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

                    self.W2_1 = self.W2_1 - self.learning_rate * dW2
                    self.b2_1 = self.b2_1 - self.learning_rate * db2
                
                if task == 2:
                    Z2 = np.matmul(self.W2_2, A1)+self.b2_2
                    A2 = np.exp(Z2)/np.sum(np.exp(Z2),axis=0)

                    cost = loss(y_new, A2)

                    dZ2 = A2-y_new
                    dW2 = (1./m) * np.matmul(dZ2, A1.T)
                    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

                    dA1 = np.matmul(self.W2_1.T, dZ2)
                    dZ1 = dA1 * sig(Z1) * (1 - sig(Z1))
                    dW1 = (1./m) * np.matmul(dZ1, X_new.T)
                    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

                    self.W2_2 = self.W2_2 - self.learning_rate * dW2
                    self.b2_2 = self.b2_2 - self.learning_rate * db2

                

                self.W1 = self.W1 - self.learning_rate * dW1
                self.b1 = self.b1 - self.learning_rate * db1

                batch_errors.append(cost)

            if (j%100==0):
                print("Batch %s loss: %s"%(j, np.mean(batch_errors)))

        return self
    
    def predict_proba(self, X, task):
        Z1 = np.matmul(self.W1, X.T)+self.b1
        A1 = sig(Z1)

        if task == 1:
            Z2 = np.matmul(self.W2_1, A1)+self.b2_1
            A2 = np.exp(Z2)/np.sum(np.exp(Z2),axis=0)
        if task == 2:
            Z2 = np.matmul(self.W2_2, A1)+self.b2_2
            A2 = np.exp(Z2)/np.sum(np.exp(Z2),axis=0)
        return A2
    
    def predict(self, X, task):
        return np.argmax(self.predict_proba(X, task), axis=0)

class MultitaskSS:
    def __init__(self, X_s, X_t, y_s, X_t_init, y_t_init, X_test, y_test, 
                 need_expert=True, expert=RandomForestClassifier(n_estimators=64)):
        self.clf = MultitaskNN()
        self.X_s = X_s
        self.y_s = y_s
        self.X_t = X_t
        self.X_t_init = X_t_init
        self.y_t_init = y_t_init
        self.X_test = X_test
        self.y_test = y_test
        self.clf = MultitaskNN(learning_rate=1, nn_hidden=128, batch_size=128)
        self.need_expert = need_expert
        self.expert = expert
        self.prepared = False
        
    def prepare(self):
        # train domain expert classifier
        self.clf_t = self.expert.fit(self.X_t_init, self.y_t_init)
        
        self.clf.fit(self.X_s, self.X_t_init, self.y_s, self.y_t_init, max_iter=1000)
        self.prepared = True
        
        if self.need_expert:
            proba = self.clf.predict_proba(self.X_t, 1)+0.7*self.clf.predict_proba(self.X_t, 2)+0.9*self.clf_t.predict_proba(self.X_t).T
        else:
            proba = self.clf.predict_proba(self.X_t, 1)+0.7*self.clf.predict_proba(self.X_t, 2)
            
        self.predictions = np.argmax(proba, axis=0)
        
        # check the accuracy on test data (in practice, we are not supposed to know about this acc)
        if self.need_expert:
            proba_test = self.clf.predict_proba(self.X_test, 1)+0.7*self.clf.predict_proba(self.X_test, 2)+0.9*self.clf_t.predict_proba(self.X_test).T
        else:
            proba_test = self.clf.predict_proba(self.X_test, 1)+0.7*self.clf.predict_proba(self.X_test, 2)
        predictions_test = np.argmax(proba_test, axis=0)
        print('Test accuracy: ',accuracy_score(self.y_test, predictions_test))
        
        
        pred = (proba).T
        self.v = []
        for i,p in enumerate(pred):
            conf = np.max(p/sum(p))
            if conf>0.80:
                self.v.append(i)
        print(len(self.v),len(pred))
        
    def advance(self, step=1):
        assert (self.prepared == True), "MultitaskSS has not prepared. Call prepare() beforehand."
        for i in range(step):
            sel_X_t = np.vstack([self.X_t_init, self.X_t[self.v, :]])
            sel_y_t = np.concatenate([self.y_t_init, self.predictions[self.v]])
            
            self.clf.fit(self.X_s, sel_X_t, self.y_s, sel_y_t, max_iter=1000, warm_start=True)
            
            if self.need_expert:
                proba = self.clf.predict_proba(self.X_t, 1)+0.7*self.clf.predict_proba(self.X_t, 2)+0.9*self.clf_t.predict_proba(self.X_t).T
            else:
                proba = self.clf.predict_proba(self.X_t, 1)+0.7*self.clf.predict_proba(self.X_t, 2)
            self.predictions = np.argmax(proba, axis=0)
            
            # check the accuracy on test data (in practice, we are not supposed to know about this acc)
            if self.need_expert:
                proba_test = self.clf.predict_proba(self.X_test, 1)+0.7*self.clf.predict_proba(self.X_test, 2)+0.9*self.clf_t.predict_proba(self.X_test).T
            else:
                proba_test = self.clf.predict_proba(self.X_test, 1)+0.7*self.clf.predict_proba(self.X_test, 2)
            predictions_test = np.argmax(proba_test, axis=0)
            print('Test accuracy: ',accuracy_score(self.y_test, predictions_test))
            
            
            pred = (proba).T
            self.v = []
            for i,p in enumerate(pred):
                conf = np.max(p/sum(p))
                if conf>0.80:
                    self.v.append(i)
            print(len(self.v),len(pred))
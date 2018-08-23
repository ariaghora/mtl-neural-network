import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

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

    def fit(self, X, y, max_iter=500):
        m = X.shape[0]

        n_x = X.shape[1]
        n_class = len(set(y))

        # weight and bias initialization
        self.W1 = np.random.randn(self.nn_hidden, n_x)
        self.b1 = np.zeros((self.nn_hidden,1))
        self.W2_1 = np.random.randn(n_class, self.nn_hidden)
        self.b2_1 = np.zeros((n_class,1))

        X_shuf, y_shuf = shuffle(X, y)

        le = LabelBinarizer()
        le.fit(y)

        batches_X = np.array_split(X_shuf, m/self.batch_size)
        batches_y = np.array_split(y_shuf, m/self.batch_size)


        for j in range(max_iter):
            batch_errors = []

            for i in range(len(batches_X)):
                task = 1
                X_new = batches_X[i].T
                y_new = batches_y[i]
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

                # # if task = 1
                # self.W2 = self.W2 - self.learning_rate * dW2
                # self.b2 = self.b2 - self.learning_rate * db2

                self.W1 = self.W1 - self.learning_rate * dW1
                self.b1 = self.b1 - self.learning_rate * db1

                batch_errors.append(cost)

            if (j%100==0):
                print("Batch %s loss: %s"%(j, np.mean(batch_errors)))

        return self
    
    def predict_proba(self, X):
        task = 1
        Z1 = np.matmul(self.W1, X.T)+self.b1
        A1 = sig(Z1)

        if task == 1:
            Z2 = np.matmul(self.W2_1, A1)+self.b2_1
            A2 = np.exp(Z2)/np.sum(np.exp(Z2),axis=0)
        return A2
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)+1

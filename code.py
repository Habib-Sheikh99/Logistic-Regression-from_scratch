import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

class Logistic_Regression:
    def __init__(self, learning_rate:float=0.001, max_iter:int=1000):
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        self.epsilon=1e-15
        self.weights=0
        self.bias=0
        self.plot_result=False
        self.losses=[]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propogation(self, X:pd.Series|pd.DataFrame):
        z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(z)
        return A
      
    def compute_loss(self, y_true:pd.Series, y_pred:pd.Series):
        y1=y_true * np.log(y_pred+self.epsilon)
        y2=(1 - y_true)*np.log(1-y_pred+self.epsilon)
        loss = -np.mean(y1+y2)
        return loss
      
    def fit(self, X:pd.DataFrame|pd.Series, y:pd.Series):
        n_samples, n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        #Gradient Descent
        for _ in range(self.max_iter):
            A = self.forward_propogation(X)
            self.losses.append(self.compute_loss(y_true=y, y_pred=A))
            dz=A-y
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            #Update Parameteres
            self.weights-=self.learning_rate * dw
            self.bias-=self.learning_rate*db
        plt.plot(self.losses)
      
    def predict(self, X):
        threshold = 0.50
        processed_input = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(processed_input)
        y_probability = [1 if i > threshold else 0 for i in y_pred]
        result = pd.Series(np.array(y_probability))
        if self.plot_result:
            plt.plot(y_pred, label='Predicted')
        return result



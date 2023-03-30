import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random

class SpamCallDetector:
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.lr = LogisticRegression()
        self.dt = DecisionTreeClassifier()
    
    def data_preprocessing(self, data):
        # Clean and transform data
        X = np.array(data)
        X[:, :2] = self.scaler.fit_transform(X[:, :2])
        return X
    
    def supervised_learning(self, X, y):
        # Train supervised learning model
        self.lr.fit(X, y)
    
    def unsupervised_learning(self, X):
        # Train unsupervised learning model
        self.kmeans.fit(X)
    
    def adaptive_learning(self, X, y):
        # Train adaptive learning model
        if self.dt is None:
            self.dt = DecisionTreeClassifier(random_state=0)
        if self.scaler is None:
            self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        y = np.array(y).ravel().reshape(1,-1)
        X_reshaped = X_scaled.reshape(1, -1)
        self.dt.fit(X_reshaped, y)
    
    def integration(self, X):
        # Integrate all models
        preds_supervised = self.lr.predict(X)
        preds_unsupervised = self.kmeans.predict(X)
        preds_adaptive = self.dt.predict(X)
        # Combine predictions
        preds = np.zeros(len(X))
        for i in range(len(X)):
            if preds_supervised[i] == 1:
                preds[i] = 1
            elif preds_unsupervised[i] == self.n_clusters-1:
                preds[i] = 1
            elif preds_adaptive[i] == 1:
                preds[i] = 1
        return preds
    
    def evaluate_accuracy(self, X, y):
        # Evaluate accuracy of supervised learning model
        preds = self.integration(X)
        return accuracy_score(y, preds)
    
    def generate_data(self, n):
        # Generate n random call data entries
        data = []
        for i in range(n):
            a = round(random.uniform(0, 1), 2)
            b = round(random.uniform(0, 1), 2)
            c = round(random.uniform(0, 1), 2)
            d = round(random.uniform(0, 1), 2)
            e = round(random.uniform(0, 1), 2)
            data.append([a, b, c, d, e])
        return data
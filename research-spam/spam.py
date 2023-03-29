import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

class SpamFilter:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((len(state_space), len(action_space)))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
    
    def train_supervised(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    def train_unsupervised(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X_scaled)
        y_pred = kmeans.predict(X_scaled)
        return y_pred
    
    def train_adaptive(self, transitions):
        for transition in transitions:
            state, action, reward, next_state = transition
            q_value = self.q_table[state, action]
            max_next_q_value = np.max(self.q_table[next_state, :])
            td_error = reward + self.gamma * max_next_q_value - q_value
            self.q_table[state, action] += self.alpha * td_error
    
    def evaluate_adaptive(self, states):
        predictions = []
        for state in states:
            if random.uniform(0, 1) < self.epsilon:
                action = random.choice(self.action_space)
            else:
                q_values = self.q_table[state, :]
                action = np.argmax(q_values)
            predictions.append(action)
        return predictions

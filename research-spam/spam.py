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
        self.q_table = np.zeros((len(state_space), action_space.n))
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
            print(state, action)
            print(type(state), type(action))
            print(int(state), int(action))
            print(type(int(state)), type(int(action)))
            q_value = self.q_table[int(state), int(action)]
            max_next_q_value = np.max(self.q_table[int(next_state), :])
            td_error = reward + self.gamma * max_next_q_value - q_value
            self.q_table[int(state), int(action)] += self.alpha * td_error
    
    def evaluate_adaptive(self, states):
        predictions = []
        for state in range(len(states)):
            if random.uniform(0, 1) < self.epsilon:
                n_actions = self.action_space.n
                action = random.randint(0, n_actions - 1)
            else:
                q_values = self.q_table[state, :]
                action = np.argmax(q_values)
            predictions.append(action)
        return predictions

    
    def evaluate_supervised(self, X, y):
        lr = LogisticRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy
    
    def evaluate_unsupervised(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X_scaled)
        y_pred = kmeans.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        return accuracy

    def generate_transitions(self, states, labels):
        transitions = []
        for i in range(len(states)-1):
            state = np.argmax(states[i])
            action = int(labels[i])
            reward = 1 if action == 0 else -1
            next_state = np.argmax(states[i+1])
            transitions.append((state, action, reward, next_state))
        return np.array(transitions, dtype=np.float32)




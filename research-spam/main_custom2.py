import numpy as np
from spam_custom import SpamCallDetector
import matplotlib.pyplot as plt
import gym
from gym import spaces
from spam import SpamFilter

## individual systems

# Define action space
action_space = spaces.Discrete(2) # 2 possible actions (0 or 1)

# Generate some random state spaces and labels (for demonstration purposes)
num_calls = 1000
state_spaces = []
labels = []
for i in range(num_calls):
    state_space = [np.random.normal(0, 1) for _ in range(5)]  # Example features: 5 continuous features
    label = np.random.randint(2)  # Example label: binary (0 or 1)
    state_spaces.append(state_space)
    labels.append(label)

# Initialize the spam filter
filter = SpamFilter(state_spaces, action_space)

# Train the filter using supervised learning
filter.train_supervised(state_spaces, labels)

# Evaluate the accuracy of the supervised learning filter
accuracy_supervised = filter.evaluate_supervised(state_spaces, labels)
print("Supervised Learning Accuracy:", accuracy_supervised)

# Train the filter using unsupervised learning
filter.train_unsupervised(state_spaces)

# Evaluate the accuracy of the unsupervised learning filter
accuracy_unsupervised = filter.evaluate_unsupervised(state_spaces, labels)
print("Unsupervised Learning Accuracy:", accuracy_unsupervised)

# Train the filter using the adaptive method
transitions = filter.generate_transitions(state_spaces, labels)
filter.train_adaptive(transitions)

# Evaluate the accuracy of the adaptive AI filter
accuracy_adaptive = filter.evaluate_adaptive(state_spaces)
print("Adaptive AI Accuracy:", accuracy_adaptive)

## custom system
print("Generating custom system model...")
# Create a line graph of the accuracy over time for the new model
# feature_vector - size 1x5
# feature_matrix - size nx5, n feature_vector elements in feature_matrix
accuracy_over_time = [] # expecting n data points in time, i.e. array of size n
detector = SpamCallDetector(n_clusters=1)
data = detector.generate_data(1000)
X = detector.data_preprocessing(data)
# Define the minimum and maximum values for each feature
a_min, a_max = -1, 0.3
b_min, b_max = -1, 0.5
c_min, c_max = 10.0, 50.0
d_min, d_max = -120.0, -70.0
e_min, e_max = -3.0, 0.3
y = 0
y_tot = np.zeros(len(X)) # Initialize y_tot as an array of zeros with the same length as X
for i, feature_vector in enumerate(X):
    print("feat: ", feature_vector)
    if (feature_vector[0] >= a_min and feature_vector[0] <= a_max) or (feature_vector[1] >= b_min and feature_vector[1] <= b_max) or (feature_vector[2] >= c_min and feature_vector[2] <= c_max) or (feature_vector[3] >= d_min and feature_vector[3] <= d_max) or (feature_vector[4] >= e_min and feature_vector[4] <= e_max):
        y_tot[i] = 1  # set the i-th element of y_tot to 1 if the feature vector meets the threshold conditions
print(y_tot, type(y_tot))
for i, feature_vector in enumerate(X):
    # Reshape feature vector to a 2D array with one row
    feature_vector = feature_vector.reshape(1, -1)
    print(feature_vector, type(feature_vector[0]))
    detector.adaptive_learning(feature_vector, y_tot)
    detector.unsupervised_learning(feature_vector)
    detector.supervised_learning(X, y_tot)
    accuracy = detector.evaluate_accuracy(X, y_tot)
    accuracy_over_time.append(accuracy)

print("My System Accuracy: ", accuracy_over_time)
import random
import math
# Define the labels and predicted probabilities
y_pred = detector.integration(data)
# Calculate binary cross-entropy loss
def binary_cross_entropy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        print("incorrect sizing, len(y_true) != len(y_pred)")
        return 1
    epsilon = 1e-10  # Small value to avoid division by zero
    def min_max_scaling(X):
        """
        Scale the values in X so that they fall within the range [0, 1].
        """
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        if np.any(max_vals == min_vals):
            print("error - cannot normalize")
            return X
        return (X - min_vals) / (max_vals - min_vals)
    y_true_normalized = min_max_scaling(y_true)
    y_pred_normalized = min_max_scaling(y_pred)
    print("y_true_normalized, y_pred_normalized")
    print(y_true_normalized, y_pred_normalized)
    loss = -y_true_normalized * np.log(y_pred_normalized + epsilon) - (1 - y_true_normalized) * np.log(1 - y_pred_normalized + epsilon)
    print(loss)
    loss = np.sum(loss)
    print(loss)
    max_loss = -math.log(0.5) * sum(y_true) * len(y_true)
    print(max_loss)
    return loss / max_loss
# Print the loss value
print("Binary Cross-Entropy Loss on Avg: ", binary_cross_entropy(y_tot, y_pred))
from sklearn.metrics import precision_score, recall_score, f1_score
# y_true is an array of actual class labels
# y_pred is an array of predicted class labels
precision = precision_score(y_tot, y_pred)
recall = recall_score(y_tot, y_pred)
f1 = f1_score(y_tot, y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")



# ## Graph accuracy

# # Create the figure and axes
# fig, ax = plt.subplots()
# # Plot the lines
# # Add horizontal lines for constant values
# ax.plot(accuracy_adaptive, label='adaptive')
# ax.hlines(accuracy_supervised, 0, len(accuracy_adaptive)-1, linestyles='dashed', colors='r', label='supervised')
# ax.hlines(accuracy_unsupervised, 0, len(accuracy_adaptive)-1, linestyles='dashed', colors='g', label='unsupervised')
# ax.plot(accuracy_over_time, label='custom')
# ax.set_xticks(range(0, len(accuracy_adaptive), 100))
# plt.title('Accuracy of Spam Call Filtering Model')
# plt.xlabel('Time (ms)')
# plt.ylabel('Accuracy')
# # Add a legend
# ax.legend()
# plt.savefig('Figure_2.png')
# # Create the plot on a new graph alone
# plt.clf()
# plt.plot(accuracy_over_time)
# plt.title('Accuracy of Spam Call Filtering Model - Custom System')
# plt.xlabel('Time (ms)')
# plt.ylabel('Accuracy')
# plt.savefig('Figure_3.png')
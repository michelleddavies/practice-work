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
detector = SpamCallDetector(n_clusters=3)
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
detector.supervised_learning(X, y_tot)
detector.unsupervised_learning(X)
for feature_vector in X:
    if (feature_vector[0] >= a_min and feature_vector[0] <= a_max) and (feature_vector[1] >= b_min and feature_vector[1] <= b_max) and (feature_vector[2] >= c_min and feature_vector[2] <= c_max) and (feature_vector[3] >= d_min and feature_vector[3] <= d_max) and (feature_vector[4] >= e_min and feature_vector[4] <= e_max):
        y = np.array([1])  # convert y to a 1D numpy array
    else:
        y = np.array([0])  # convert y to a 1D numpy array of zeros
    # Reshape feature vector to a 2D array with one row
    feature_vector = feature_vector.reshape(1, -1)
    print(feature_vector, type(feature_vector[0]))
    detector.adaptive_learning(feature_vector, y)
    preds = detector.integration(feature_vector)
    accuracy = detector.evaluate_accuracy(feature_vector, y)
    accuracy_over_time.append(accuracy)

print("My System Accuracy: ", accuracy_over_time)

## Graph accuracy

# Create the figure and axes
fig, ax = plt.subplots()
# Plot the lines
# Add horizontal lines for constant values
ax.plot(accuracy_adaptive, label='adaptive')
ax.hlines(accuracy_supervised, 0, len(accuracy_adaptive)-1, linestyles='dashed', colors='r', label='supervised')
ax.hlines(accuracy_unsupervised, 0, len(accuracy_adaptive)-1, linestyles='dashed', colors='g', label='unsupervised')
ax.plot(accuracy_over_time, label='custom')
ax.set_xticks(range(0, len(accuracy_adaptive), 100))
plt.title('Accuracy of Spam Call Filtering Model')
plt.xlabel('Time (ms)')
plt.ylabel('Accuracy')
# Add a legend
ax.legend()
plt.savefig('Figure_2.png')
# Create the plot on a new graph alone
plt.clf()
plt.plot(accuracy_over_time)
plt.title('Accuracy of Spam Call Filtering Model - Custom System')
plt.xlabel('Time (ms)')
plt.ylabel('Accuracy')
plt.savefig('Figure_3.png')
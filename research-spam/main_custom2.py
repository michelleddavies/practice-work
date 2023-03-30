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

# Create a line graph of the accuracy over time for the new model
# feature_vector - size 1x5
# feature_matrix - size nx5, n feature_vector elements in feature_matrix
accuracy_over_time = [] # expecting n data points in time, i.e. array of size n
detector = SpamCallDetector(n_clusters=3)
data = detector.generate_data(1000)
X = detector.data_preprocessing(data)
for feature_vector in X:
    print("Generating custom system model...")
    y = np.random.randint(2, size=len(X)) # randomizing to better suit the use case
    detector.supervised_learning(X, y)
    detector.unsupervised_learning(X)
    detector.adaptive_learning(X, y)
    preds = detector.integration(X)
    accuracy = detector.evaluate_accuracy(X, y)
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
import numpy as np
from spam_custom import SpamCallDetector
import matplotlib.pyplot as plt
import gym
from gym import spaces
from spam import SpamFilter

## custom system
detector = SpamCallDetector(n_clusters=3)
data = detector.generate_data(1000)
X = detector.data_preprocessing(data)
y = np.random.randint(2, size=len(X))
detector.supervised_learning(X, y)
detector.unsupervised_learning(X)
detector.adaptive_learning(X, y)
preds = detector.integration(X)
accuracy = detector.evaluate_accuracy(X, y)
print("My System Accuracy: ", accuracy)

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

# Create a line graph of the accuracy over time
accuracy_data = [accuracy_supervised, accuracy_unsupervised, accuracy_adaptive]
# flatten accuracy_data
accuracy_data = np.array(accuracy_data)
print(accuracy_data)

## Graph accuracy

# Create the figure and axes
for i in range(1, 11):
	fig, ax = plt.subplots()
	# Plot the lines
	# Add horizontal lines for constant values
	ax.hlines(accuracy, 0, len(accuracy_adaptive)-1, colors='k', label='custom')
	ax.plot(accuracy_adaptive, label='adaptive')
	ax.hlines(accuracy_supervised, 0, len(accuracy_adaptive)-1, linestyles='dashed', colors='r', label='supervised')
	ax.hlines(accuracy_unsupervised, 0, len(accuracy_adaptive)-1, linestyles='dashed', colors='g', label='unsupervised')
	ax.set_xticks(range(0, len(accuracy_adaptive), 100))
	plt.title('Accuracy of Spam Call Filtering Model')
	plt.xlabel('Time (ms)')
	plt.ylabel('Accuracy')
	# Add a legend
	ax.legend()
	plt.savefig('Figure_2_{}.png'.format(i))
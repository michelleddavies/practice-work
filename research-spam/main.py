import numpy as np
from spam import SpamFilter
import matplotlib.pyplot as plt
import gym
from gym import spaces

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
plt.plot(accuracy_data)
plt.title('Accuracy of Spam Call Filtering Model')
plt.xlabel('Filter Type')
plt.ylabel('Accuracy')
plt.xticks([0, 1, 2], ['Supervised', 'Unsupervised', 'Adaptive'])
plt.show()

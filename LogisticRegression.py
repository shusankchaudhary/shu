# Import necessary libraries
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv("/Users/aaronmackenzie/Desktop/Programming/Student-University.csv", header=None, names=['Exam1', 'Exam2', 'Admitted'])

# Preprocess data
# Here, we are not performing any preprocessing, but you might want to handle missing values or scale features if needed.

# Convert DataFrame to NumPy arrays
X = np.array(data[['Exam1', 'Exam2']])
y = np.array(data['Admitted'])

# Add a column of ones to X for the intercept term
X = np.column_stack((np.ones(X.shape[0]), X))

# Initialize parameters
theta = np.zeros(X.shape[1])

# Hyperparameters
alpha = 0.01  # learning rate
iterations = 10000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    J = (1 / m) * (-y @ np.log(h) - (1 - y) @ np.log(1 - h))
    return J

# Gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []

    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * X.T @ (h - y)
        theta -= alpha * gradient
        J_history.append(cost_function(X, y, theta))

    return theta, J_history

# Run gradient descent
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

# Print the learned parameters
print("Learned Parameters:")
print("Theta 0 (Intercept):", theta[0])
print("Theta 1 (Exam1):", theta[1])
print("Theta 2 (Exam2):", theta[2])

# Predictions
predictions = np.round(sigmoid(X @ theta))

# Accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)

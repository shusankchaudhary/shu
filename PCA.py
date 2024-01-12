# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('/Users/aaronmackenzie/Downloads/iris(For PCA Program).csv')

# Separate features (X) and target variable (Y)
X = data.drop("species", axis=1)
Y = data.species

# Calculate the covariance matrix of the features
cov = np.cov(X.T)

# Calculate the eigenvalues and eigenvectors of the covariance matrix
egval, egvec = np.linalg.eig(cov)

# Sort eigenvalues in descending order and get corresponding eigenvectors
sl_val = np.argsort(egval)[::-1]
sl_egvec = egvec[:, sl_val]

# Select the top 2 eigenvectors (corresponding to the 2 largest eigenvalues)
eid_sub = sl_egvec[:, :2]

# Project the original data onto the 2D subspace defined by the selected eigenvectors
x_red = np.dot(eid_sub.transpose(), X.transpose()).transpose()

# Scatter plot of the reduced data
plt.scatter(x_red[:, 0], x_red[:, -1], c=Y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Projection of Iris Dataset using PCA')
plt.show()

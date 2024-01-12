# Import necessary libraries
import pandas as pd
import numpy as np
from copy import deepcopy

# Set the number of clusters (k) to 3
k = 3

# Create a DataFrame with two columns, 'X1' and 'X2', representing the data points
data = pd.DataFrame({"X1": [5.9, 4.6, 6.2, 4.7, 5.5, 5.0, 4.9, 6.7, 5.1, 6.],
                     "X2": [3.2, 2.9, 2.8, 3.2, 4.2, 3.0, 3.1, 3.1, 3.8, 3.]})

# Create a NumPy array 'X' containing the data points
X = np.array(list(zip(data["X1"], data["X2"])))

# Initial centroids for the clusters
centroids = np.array(list(zip([6.2, 6.6, 6.5], [3.2, 3.7, 3.0])))
iterr = 0

# K-means clustering algorithm
while True:
    iterr += 1
    
    # Assign each point to the nearest centroid
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    
    # Update centroids based on the mean of points in each cluster
    new_centroids = np.array([np.mean(X[clusters == p], axis=0) for p in range(k)])
    
    # Print old and new centroids, and error (change in centroids)
    print("Old centroid:")
    print(centroids)
    print("New centroid after iteration", iterr)
    print(new_centroids)
    print("Error:", np.linalg.norm(new_centroids - centroids, axis=None))
    
    # Check convergence by comparing old and new centroids
    if np.array_equal(centroids, new_centroids):
        break
    
    # Update centroids for the next iteration
    centroids = deepcopy(new_centroids)
    
    # Print assigned clusters and add some space for clarity
    print("Clusters:", clusters)
    print("\n\n")

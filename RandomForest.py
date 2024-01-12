# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('B3-pima.csv')

# Separate features (X) and target variable (Y)
X = data.drop("Outcome", axis=1)
Y = data.Outcome

# Split data into training and testing sets with stratification
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, stratify=Y, test_size=0.3)

# Initialize Decision Tree and Random Forest classifiers
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

# Train the classifiers
dt.fit(xtrain, ytrain)
rf.fit(xtrain, ytrain)

# Make predictions
ypred1 = dt.predict(xtest)
ypred2 = rf.predict(xtest)

# Calculate accuracy for Decision Tree and Random Forest classifiers
a = accuracy_score(ytest, ypred1)
b = accuracy_score(ytest, ypred2)

# Print the accuracies
print("Accuracy of Decision Tree Classifier:", a)
print("Accuracy of Random Forest Classifier:", b)

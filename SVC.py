# Import necessary libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('glass(For SVM Program).csv')

# Separate features (X) and target variable (Y)
X = data.drop("Type", axis=1)
Y = data.Type

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25)

# Initialize SVM models with different kernel functions
model1 = SVC(kernel="rbf")  # Radial Basis Function (RBF) kernel
model2 = SVC(kernel="poly", degree=3)  # Polynomial kernel of degree 3
model3 = SVC(kernel="sigmoid", gamma=0.001)  # Sigmoid kernel with a specific gamma value

# Train the SVM models
model1.fit(xtrain, ytrain)
model2.fit(xtrain, ytrain)
model3.fit(xtrain, ytrain)

# Make predictions
ypred1 = model1.predict(xtest)
ypred2 = model2.predict(xtest)
ypred3 = model3.predict(xtest)

# Calculate accuracy for each SVM model
a = accuracy_score(ytest, ypred1)
b = accuracy_score(ytest, ypred2)
c = accuracy_score(ytest, ypred3)

# Print the accuracies
print("Accuracy with RBF kernel:", a)
print("Accuracy with Polynomial kernel (degree 3):", b)
print("Accuracy with Sigmoid kernel (gamma=0.001):", c)

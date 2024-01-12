# Import necessary libraries
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the CSV file containing the zoo data into a DataFrame
data = pd.read_csv('/Users/aaronmackenzie/Desktop/Programming/zoo_data.csv')

# Separate features (X) and target variable (Y)
X = data.drop("1.7", axis=1)  # X contains all columns except '1.7'
Y = data['1.7']  # Y contains the '1.7' column

# Split the dataset into training and testing sets (70% training, 30% testing)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3)

# Create a Decision Tree Classifier
dtc = DecisionTreeClassifier()

# Train the classifier on the training data
dtc.fit(xtrain, ytrain)

# Make predictions on the testing set
ypred = dtc.predict(xtest)

# Print the accuracy score of the model on the testing set
print("Accuracy Score:", accuracy_score(ytest, ypred))

# Print the classification report, which includes precision, recall, f1-score, and support
print("Classification Report:\n", classification_report(ytest, ypred))

# Uncomment the line below if you want to print the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(ytest, ypred))

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

# Load dataset
data = pd.read_csv("/Users/aaronmackenzie/Desktop/Programming/covid.csv")

# Encode categorical variables
le = preprocessing.LabelEncoder()
pc = le.fit_transform(data['pc'].values)
wbc = le.fit_transform(data['wbc'].values)
ast = le.fit_transform(data['ast'].values)
bc = le.fit_transform(data['bc'].values)
ldh = le.fit_transform(data['ldh'].values)
y = le.fit_transform(data['diagnosis'].values)

# Features
X = np.array(list(zip(pc, wbc, ast, bc, ldh)))

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the Naive Bayes classifier
naive_model = MultinomialNB()
naive_model.fit(xtrain, ytrain)

# Make predictions
ypred = naive_model.predict(xtest)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(ytest, ypred))
print("Classification Report:\n", classification_report(ytest, ypred))

# ROC curve
lr_prob = naive_model.predict_proba(xtest)[:, 1]
lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_prob)

# Plot ROC curve
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Naive Bayes Classifier')
pyplot.xlabel("False positive rate")
pyplot.ylabel("True positive rate")
pyplot.legend()
pyplot.show()

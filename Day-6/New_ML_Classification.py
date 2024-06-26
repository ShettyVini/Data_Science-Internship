# #1. Logistic Regression

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create an instance of LogisticRegression
logr = LogisticRegression()

# Load the Iris dataset from a CSV file
df = pd.read_csv("C:/Users/viraj/Desktop/CODES/DAY 6 DATA SCIENCE/Iris.csv")

# Prepare the feature matrix x and the target vector y
x = df.drop('Id', axis=1)  # Drop the 'Id' column
x = x.drop('Species', axis=1)  # Drop the 'Species' column from x
y = df['Species']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3)

# Print the split data (for debugging purposes)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Train the logistic regression model
logr.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = logr.predict(X_test)

# Print accuracy, classification report, and confusion matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print the coefficients (weights) of the logistic regression model
print("Coefficients (weights):", logr.coef_)

# Print the intercept of the logistic regression model
print("Intercept:", logr.intercept_)

#2. Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB

# Create an instance of GaussianNB (Naive Bayes classifier)
nb = GaussianNB()

# Split the data into training and testing sets with different random state and test size
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.4)

# Train the Naive Bayes model
nb.fit(X_train, y_train)

# Predict the labels for the test set
y_pred1 = nb.predict(X_test)

# Print the accuracy score for the Naive Bayes model
print("Naive Bayes: ", accuracy_score(y_test, y_pred1))

# 3. KNN (K-Nearest Neighbors)

from sklearn.neighbors import KNeighborsClassifier

# Create an instance of KNeighborsClassifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Train the KNN model
train = knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Print the accuracy score for the KNN model
print("KNN:" ,accuracy_score(y_test, y_pred))

#4. Decision Tree

from sklearn import tree

# Create an instance of DecisionTreeClassifier
dt = tree.DecisionTreeClassifier()

# Split the data into training and testing sets with different random state and test size
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

# Train the decision tree model
train = dt.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = dt.predict(X_test)

# Print the accuracy score for the decision tree model
print("Decision Tree:" , accuracy_score(y_test, y_pred))

# 5. Random Forest

from sklearn.ensemble import RandomForestClassifier

# Create an instance of RandomForestClassifier
rf = RandomForestClassifier()

# Split the data into training and testing sets with different random state and test size
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

# Train the random forest model
train = rf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = rf.predict(X_test)

# Print the accuracy score for the random forest model
print("RandomForest:",accuracy_score(y_test, y_pred))

# 6. Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

# Create an instance of GradientBoostingClassifier with 10 estimators
gbm = GradientBoostingClassifier(n_estimators=10)

# Split the data into training and testing sets with different random state and test size
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# Train the gradient boosting model
gbm.fit(X_train, Y_train)

# Predict the labels for the test set
y_pred = gbm.predict(X_test)

# Print the accuracy score for the gradient boosting model
print("GBM: ", accuracy_score(Y_test, y_pred))

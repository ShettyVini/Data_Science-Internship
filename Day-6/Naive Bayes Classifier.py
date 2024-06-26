from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming x and y are already defined


df=pd.read_csv("C:/Users/viraj/Desktop/CODES/DAY 6 DATA SCIENCE/Iris.csv")

x = df.drop('Id', axis=1)
x = x.drop('Species', axis=1)
y = df['Species']
# x: Features, y: Target

nb = GaussianNB()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.4)

# Train the Naive Bayes model
nb.fit(X_train, y_train)

# Predict the labels for the test set
y_pred1 = nb.predict(X_test)

# Calculate and print the accuracy score
print("Naive Bayes: ", accuracy_score(y_test, y_pred1))

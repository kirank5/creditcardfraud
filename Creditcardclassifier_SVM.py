
from sklearn import datasets
import numpy as np
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm


# #### Generating the dataset

# Loading credit card dataset
data = pd.read_csv("creditcard.csv")
# Loading the features and storing them in X
#X = data.iloc[:, 0:30]
X = data.iloc[:, :]

# Loading the labels and storing them in y
y = data["Class"]
print(X)


# #### Splitting the data into train,test and validation sets.


## Train = 60 % , Test = 20 % , Train = 20 %
X_train, X_test, X_validation = np.split(X, [int(.6 * len(X)), int(.8 * len(X))])
Y_train, Y_test, Y_validation = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])


# #### Implementing Svm

# #### Applying Linearsvc


from sklearn.svm import LinearSVC
## Creating the svm object
clf = LinearSVC(random_state=0)
# Intilizing the time object
t0 = time.time()
t1 = time.time()
## Fitting the data into the trained model
clf.fit(X_train, Y_train)
print("Training time is ", round(time.time() - t0, 3),  "seconds")


# #### Predicting the values


# Testing the data on the trained model
y_pred = clf.predict(X_test)
print("Testing time is ", round(time.time() - t1, 3),  "seconds")


# #### Calculating the accuracy


accuracy_score(Y_test,y_pred)


# #### Parameter tuning  
# 
# Let us try to tune kernel parameter of svm

# ### rbf kernel

# Creating the svm object
clf = svm.SVC(kernel='rbf', gamma=0.7)
# Creating the time object
t0 = time.time()
t1 = time.time()
# Fitting the data into the model
clf.fit(X_train, Y_train)
#Calculating the training time
print("Training time is ", round(time.time() - t0, 3),  "seconds")


# #### predicting the values

#testing the data on the trained model
y_pred = clf.predict(X_test)
# Calculating the testing time
print("Testing time is ", round(time.time() - t1, 3),  "seconds")


# #### Calculating the accuracy


accuracy_score(Y_test,y_pred)


# ### polynomial kernel

# Creating the svm object
clf = svm.SVC(kernel='poly', degree=3)
# Creating the time object
t0 = time.time()
t1 = time.time()
# Fitting the data into the model
clf.fit(X_train, Y_train)
# Calculating the training time
print("training time is ", round(time.time() - t0), "seconds")


# #### Predicting the values


# Testing the data on the trained model
y_pred = clf.predict(X_test)
# Calculating the testing time
print("testingtime is ", round(time.time() - t1), "seconds")


# #### Calculating the accuracy

accuracy_score(Y_test,y_pred)

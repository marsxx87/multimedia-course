from sklearn import svm
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

wine = datasets.load_wine()
# print(wine.data)
# print(wine.data.shape)
# print(wine.feature_names)

total_wine = wine.data.shape[0] # total_wine = 178

# only take some of the features for input
X = []
for i in range(total_wine):
    row = []
    row.append(wine.data[i, 0]) # take the value of alchohol for input
    X.append(row)

Y = wine.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 4)

clf = svm.SVC(kernel = 'linear', C = 1, gamma = 'auto')
clf.fit(X_train, Y_train)

print("predict")
print(clf.predict(X_train)) # target = Y_train
print(clf.predict(X_test)) # target = Y_test

print("Accuracy")
print(clf.score(X_train, Y_train))
print(clf.score(X_test, Y_test))
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
    row.append(wine.data[i, 2]) # take the value of alchohol for input
    X.append(row)

Y = wine.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2)
#test_size:訓練集測試集分割比例     #random_state:打亂資料集

clf = svm.SVC(kernel = 'linear', C = 3, gamma = 'auto')
#
clf.fit(X_train, Y_train)

#p.24
# print("predict")
# print(clf.predict(X_train)) # target = Y_train
# print(clf.predict(X_test)) # target = Y_test

#p.24
print("Accuracy")
print("Train accuracy:", clf.score(X_train, Y_train))  #訓練資料準確率
print("Test accuracy :", clf.score(X_test, Y_test))    #測試資料準確率
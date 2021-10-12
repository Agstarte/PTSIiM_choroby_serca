from sys import argv
import pandas as pd
import numpy as np
import keel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = np.genfromtxt("heart.csv", delimiter=",")
X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=.30,
#     random_state=1234
# )


model = KNeighborsClassifier()

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)
print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))

predict = model.predict([[61,1,4,140,207,0,2,138,1,19.0,1,1,7]])
print(predict)

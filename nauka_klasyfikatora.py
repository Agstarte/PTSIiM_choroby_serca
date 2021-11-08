import numpy as np
# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pickle


dataset = np.genfromtxt("heart.csv", delimiter=",")
X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)

# K NAJ
KNN = KNeighborsClassifier()

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    KNN.fit(X_train, y_train)
    predict = KNN.predict(X_test)
    scores.append(accuracy_score(y_test, predict))


pickle.dump(KNN, open('heart_KNN.pkl', 'wb'), protocol=4)

MLP = MLPClassifier(max_iter=300)

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    MLP.fit(X_train, y_train)
    predict = MLP.predict(X_test)
    scores.append(accuracy_score(y_test, predict))


pickle.dump(MLP, open('heart_MLP.pkl', 'wb'), protocol=4)

SVC = SVC(probability=True, kernel='linear')

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    SVC.fit(X_train, y_train)
    predict = SVC.predict(X_test)
    scores.append(accuracy_score(y_test, predict))


pickle.dump(SVC, open('heart_SVC.pkl', 'wb'), protocol=4)

NB = GaussianNB()

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    NB.fit(X_train, y_train)
    predict = NB.predict(X_test)
    scores.append(accuracy_score(y_test, predict))


pickle.dump(NB, open('heart_NB.pkl', 'wb'), protocol=4)

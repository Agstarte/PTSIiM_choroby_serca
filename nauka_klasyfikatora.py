import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pickle


dataset = np.genfromtxt("heart.csv", delimiter=",")
X = dataset[1:, :-1]
y = dataset[1:, -1].astype(int)

clf = KNeighborsClassifier()

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))


pickle.dump(clf, open('klasyfikator_chorob_serca.pkl', 'wb'), protocol=4)

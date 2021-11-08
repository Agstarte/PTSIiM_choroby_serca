import pickle

import numpy as np

label = {
        1: 'Negatywna',
        2: 'Pozytywna'
    }

test_data = [[43, 1, 4, 140, 207, 0, 2, 138, 1, 19.0, 1, 1, 7]]
classifiers_pkl = {
    'KNN': 'heart_KNN.pkl',
    'MLP': 'heart_MLP.pkl',
    'NB': 'heart_NB.pkl',
    'SVC': 'heart_SVC.pkl',

}

out = dict()

for clf in classifiers_pkl:
    out[clf] = dict()

    model = pickle.load(open(classifiers_pkl[clf], 'rb'))
    prognosis = model.predict(test_data)[0]

    out[clf]['prognosis'] = label[prognosis]
    out[clf]['predict_proba'] = np.max(model.predict_proba(test_data)) * 100


print(out)

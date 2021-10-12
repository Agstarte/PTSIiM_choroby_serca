import pickle

import numpy as np

test_data = [[56, 1, 4, 140, 207, 0, 2, 138, 1, 19.0, 1, 1, 7]]

clf = pickle.load(open('klasyfikator_chorob_serca.pkl', 'rb'))
prognosis = clf.predict(test_data)[0]
label = {
    1: 'Negatywna',
    2: 'Pozytywna'
}
print(f'Prognoza: {label[prognosis]}\n'
      f'Prawdopodobieństwo: {np.max(clf.predict_proba(test_data)) * 100}')

import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc

def balance_data(data):
    cls_0 = data[data.IND_BOM_1_2 == 0]
    cls_1 = data[data.IND_BOM_1_2 == 1]

    while len(cls_1) < len(cls_0):
        cls_1 = cls_1.append(cls_1)

    return cls_0.append(cls_1.iloc[:len(cls_0)])

def get_roc_auc(y, pred):
    false_positive_rate, true_positive_rate, _ = roc_curve(y, pred)
    return auc(false_positive_rate, true_positive_rate)

# READ DATA

data = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
data = data.sample(frac=0.1, random_state=100)
data = balance_data(data)

X, y = data.iloc[:, 1:-2], data.iloc[:, -2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# ESTIMATORS TUNNING

# options = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 600, 750, 1024]

# est_train = []
# est_test = []

# for x in options:
#     clf = RandomForestClassifier(n_estimators=x, max_depth=8)
#     clf.fit(X_train, y_train)

#     train_pred = clf.predict(X_train)
#     est_train.append(get_roc_auc(y_train, train_pred))

#     test_pred = clf.predict(X_test)
#     est_test.append(get_roc_auc(y_test, test_pred))

#     print (x)

# print (est_train)
# print (est_test)

# MAX DEPTH TUNNING

# import numpy as np
# options = np.linspace(1, 32, 32)

# md_train = []
# md_test = []

# for x in options:
#     clf = RandomForestClassifier(max_depth=x, n_estimators=128)
#     clf.fit(X_train, y_train)

#     train_pred = clf.predict(X_train)
#     md_train.append(get_roc_auc(y_train, train_pred))

#     test_pred = clf.predict(X_test)
#     md_test.append(get_roc_auc(y_test, test_pred))

#     print (x)
#     print (md_train)
#     print (md_test)

# MIN SAMPLES SPLIT TUNNING

# import numpy as np
# options = np.linspace(0.01, 0.10, 10)

# mss_train = []
# mss_test = []

# for x in options:
#     clf = GradientBoostingClassifier(min_samples_split=x)
#     clf.fit(X_train, y_train)

#     train_pred = clf.predict(X_train)
#     mss_train.append(get_roc_auc(y_train, train_pred))

#     test_pred = clf.predict(X_test)
#     mss_test.append(get_roc_auc(y_test, test_pred))

#     print (x)

# print (mss_train)
# print (mss_test)

# MAX FEATURES TUNNING

import numpy as np
options = np.linspace(16, 240, 15)

mf_train = []
mf_test = []

for x in options:
    x = int(x)
    clf = RandomForestClassifier(max_features=x, max_depth=16)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    mf_train.append(get_roc_auc(y_train, train_pred))

    test_pred = clf.predict(X_test)
    mf_test.append(get_roc_auc(y_test, test_pred))

    print (x)

print (mf_train)
print (mf_test)
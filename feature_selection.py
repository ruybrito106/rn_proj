import pandas as pd
import time

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from project import Project

def minus(a):
    return (time.time() - a) / 60.0

def balance_data(data):
    cls_0 = data[data.IND_BOM_1_2 == 0]
    cls_1 = data[data.IND_BOM_1_2 == 1]

    while len(cls_1) < len(cls_0):
        cls_1 = cls_1.append(cls_1)

    return cls_0.append(cls_1.iloc[:len(cls_0)])

# READ DATA

st = time.time()

data = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
data = data.sample(frac=0.1, random_state=100)
data = balance_data(data)

X, y = data.iloc[:, 1:-2], data.iloc[:, -2]

cur = time.time()
print ('Data read in {0} mins'.format(minus(st)))

for n in [50, 80, 100, 120, 150, 180, 200, 220]:
    pca = PCA(n_components=n)
    pca.fit(X)
    X_pca = pca.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3333, random_state=100)

    clf = GradientBoostingClassifier(n_estimators=500, loss='exponential', verbose=0)
    history = clf.fit(X_train, y_train)

    project = Project()
    project.display_metrics(clf, X_test, y_test, history, mode=1)

    nxt = time.time()
    print ('n={0} (GB) ready in {1} mins'.format(n, minus(cur)))

    clf = RandomForestClassifier(n_estimators=500, max_depth=30, verbose=0)
    history = clf.fit(X_train, y_train)

    project = Project()
    project.display_metrics(clf, X_test, y_test, history, mode=1)

    cur = time.time()
    print ('n={0} (RF) ready in {1} mins'.format(n, minus(nxt)))
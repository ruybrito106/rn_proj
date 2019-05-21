# from pandas import read_csv, concat
# from sklearn.utils import shuffle
# import numpy as np

# def expand_until(df, length):
#     while len(df) <= length:
#         df = df.append(df)
#     return df.iloc[:length]

# raw_data = read_csv('data/raw.csv', sep='\t', header=0)

# def from_total(n):
#     return float(n) / float(len(raw_data))

# test_data = raw_data.sample(frac=0.25, random_state=200)
# train_data = raw_data.drop(test_data.index)

# test_data.to_csv('data/test.csv', sep='\t')
# print("Samples on test: {}".format(from_total(len(test_data))))

# chunks = np.array_split(train_data, 5)

# k = 0
# for c in chunks:
#     print("Samples on chunk {}: {}".format(k, from_total(len(c))))
#     current_class_train = c.sample(frac=0.6666, random_state=200)
#     current_class_valid = c.drop(current_class_train.index)

#     # TRAINING
#     train_class_0 = current_class_train[current_class_train.IND_BOM_1_2 == 0]
#     train_class_1 = current_class_train[current_class_train.IND_BOM_1_2 == 1]

#     train_class_1 = expand_until(train_class_1, len(train_class_0))
#     train_class_1 = shuffle(train_class_1)

#     train_all = shuffle(concat([train_class_0, train_class_1]))
#     train_all.to_csv('data/chunk_{}/train.csv'.format(k), sep='\t')

#     # print("---| train_class_0 samples: {}", len(train_class_0))
#     # print("---| train_class_1 samples: {}", len(train_class_1))

#     # VALIDATION
#     valid_class_0 = current_class_valid[current_class_valid.IND_BOM_1_2 == 0]
#     valid_class_1 = current_class_valid[current_class_valid.IND_BOM_1_2 == 1]

#     valid_class_1 = expand_until(valid_class_1, len(valid_class_0))
#     valid_class_1 = shuffle(valid_class_1)

#     valid_all = shuffle(concat([valid_class_0, valid_class_1]))
#     valid_all.to_csv('data/chunk_{}/valid.csv'.format(k), sep='\t')

#     # print("---| valid_class_0 samples: {}", len(valid_class_0))
#     # print("---| valid_class_1 samples: {}", len(valid_class_1))

#     k += 1

import pandas as pd
from sklearn.utils import shuffle

def structure():
    data = pd.read_csv('data/raw.csv', sep='\t', header=0, index_col=0)
    data = shuffle(data)

    test_len = int(0.2 * len(data))
    data.iloc[:test_len, :].to_csv('data/test.csv', sep='\t')

    train = data.iloc[test_len:, :]

    ensemble_train_size = int(0.3 * len(data))
    train.iloc[:ensemble_train_size, :].to_csv('data/ensemble_train.csv', sep='\t')

    train = train.iloc[ensemble_train_size:, :]

    chunk_len = int(0.25 * len(train))
    chunk_divs = [x * chunk_len for x in range(1,4)]

    train.iloc[:chunk_divs[0], :].to_csv('data/0.csv', sep='\t')
    train.iloc[chunk_divs[0]:chunk_divs[1], :].to_csv('data/1.csv', sep='\t')
    train.iloc[chunk_divs[1]:chunk_divs[2], :].to_csv('data/2.csv', sep='\t')
    train.iloc[chunk_divs[2]:, :].to_csv('data/3.csv', sep='\t')

def balance_data(data):
    cls_0 = data[data.IND_BOM_1_2 == 0]
    cls_1 = data[data.IND_BOM_1_2 == 1]

    if len(cls_1) < len(cls_0):
        while len(cls_1) < len(cls_0):
            cls_1 = cls_1.append(cls_1)
        return cls_0.append(cls_1.iloc[:len(cls_0)])
    else:
        while len(cls_0) < len(cls_1):
            cls_0 = cls_0.append(cls_0)
        return cls_1.append(cls_0.iloc[:len(cls_1)])

def read_chunk_structured(index, balance=True):
    data = pd.read_csv('data/{0}.csv'.format(index), sep='\t', header=0, index_col=0)
    if balance:
        data = balance_data(data)
    return shuffle(data)

def read_ensemble_structured(balance=True):
    data = pd.read_csv('data/ensemble_train.csv', sep='\t', header=0, index_col=0)
    if balance:
        data = balance_data(data)
    return shuffle(data)
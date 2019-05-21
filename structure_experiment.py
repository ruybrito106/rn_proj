from pandas import read_csv, concat
from sklearn.utils import shuffle
import numpy as np

def expand_until(df, length):
    while len(df) <= length:
        df = df.append(df)
    return df.iloc[:length]

raw_data = read_csv('data/raw.csv', sep='\t', header=0)

def from_total(n):
    return float(n) / float(len(raw_data))

test_data = raw_data.sample(frac=0.25, random_state=200)
train_data = raw_data.drop(test_data.index)

test_data.to_csv('data/test.csv', sep='\t')
print("Samples on test: {}".format(from_total(len(test_data))))

chunks = np.array_split(train_data, 5)

k = 0
for c in chunks:
    print("Samples on chunk {}: {}".format(k, from_total(len(c))))
    current_class_train = c.sample(frac=0.6666, random_state=200)
    current_class_valid = c.drop(current_class_train.index)

    # TRAINING
    train_class_0 = current_class_train[current_class_train.IND_BOM_1_2 == 0]
    train_class_1 = current_class_train[current_class_train.IND_BOM_1_2 == 1]

    train_class_1 = expand_until(train_class_1, len(train_class_0))
    train_class_1 = shuffle(train_class_1)

    train_all = shuffle(concat([train_class_0, train_class_1]))
    train_all.to_csv('data/chunk_{}/train.csv'.format(k), sep='\t')

    # print("---| train_class_0 samples: {}", len(train_class_0))
    # print("---| train_class_1 samples: {}", len(train_class_1))

    # VALIDATION
    valid_class_0 = current_class_valid[current_class_valid.IND_BOM_1_2 == 0]
    valid_class_1 = current_class_valid[current_class_valid.IND_BOM_1_2 == 1]

    valid_class_1 = expand_until(valid_class_1, len(valid_class_0))
    valid_class_1 = shuffle(valid_class_1)

    valid_all = shuffle(concat([valid_class_0, valid_class_1]))
    valid_all.to_csv('data/chunk_{}/valid.csv'.format(k), sep='\t')

    # print("---| valid_class_0 samples: {}", len(valid_class_0))
    # print("---| valid_class_1 samples: {}", len(valid_class_1))

    k += 1

    

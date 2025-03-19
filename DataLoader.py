import os

import Neural

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import Util


def load(path, column=None):
    try:
        if column is None:
            return pd.read_csv(path, delim_whitespace=True, header=None)
        else:
            return pd.read_csv(path, delim_whitespace=True, header=None, index_col=column)
    except FileNotFoundError:
        return load(f'../{path}', column=column)


def merge_sets(set1, set2):
    return pd.concat([set1, set2], axis=0).reset_index(drop=True)


def load_all(scaled_set_test_size=None):
    x_train = load('UCI HAR Dataset/train/X_train.txt')
    y_train = load('UCI HAR Dataset/train/y_train.txt')
    x_test = load('UCI HAR Dataset/test/X_test.txt')
    y_test = load('UCI HAR Dataset/test/y_test.txt')
    features = load('UCI HAR Dataset/features.txt')
    x_train.columns = features[1]
    x_test.columns = features[1]
    labels = load('UCI HAR Dataset/activity_labels.txt', 0)
    if scaled_set_test_size is None:
        return x_train, Neural.categorise(y_train), x_test, Neural.categorise(y_test), labels
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            StandardScaler().fit_transform(merge_sets(x_train, x_test)),
            Neural.categorise(merge_sets(y_train, y_test)),
            test_size=scaled_set_test_size)
        return x_train, y_train, x_test, y_test, labels

import random
import numpy as np


def split(X, Y, dev_ratio=0.1):
    size = int(len(X) * (1 - dev_ratio))
    label = np.array(range(len(X)))
    SelectT = random.sample(range(len(X)), size)  # np.random.randint(0, len(X) - 1, size)
    train_x = X[SelectT]
    train_y = Y[SelectT]
    SelectV = np.delete(label, SelectT)
    valid_x = X[SelectV]
    valid_y = Y[SelectV]
    # print(len(label), len(SelectT), len(SelectV))
    # return X[:size], Y[:size], X[size:], Y[size:]
    return train_x, train_y, valid_x, valid_y

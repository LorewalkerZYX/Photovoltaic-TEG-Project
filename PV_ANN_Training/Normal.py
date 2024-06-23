import numpy as np


Mt = 5.899922758130535
St = 0.223066273897046
# Me = 14.2077
# Se = 11.0996
# Me = 15.6417
# Se = 9.4549
Temp = 37


def recover(y):
    # y /= const
    for i in range(len(y)):
        y[i, 0] = y[i, 0] * Temp  # * Se + Me
        y[i, 1] = y[i, 1] * St + Mt
        y[i, 1] = np.exp(y[i, 1])
    return y


def normalize(x, input=True):
    temp = x

    if input:
        v = 0.64  # hte = [0-0.64]
        T = 100  # ff = [263.15-363.15]
        solar = 1000  # n_ratio = [0-1000]
        coating = 1  # p_ratio = [0 1]
        qin = 1000  # n_ratio = [0-1000]
        morph = 3  # rhoc_h = [0 1 2 3]
        Conv = 24  # Conv = [1-25]

        for i in range(len(temp)):

            temp[i, 0] = (temp[i, 0] - 0) / solar
            temp[i, 1] = (temp[i, 1] - 263.15) / T
            temp[i, 2] = (temp[i, 2] - 1) / Conv
            temp[i, 3] = (temp[i, 3] - 0) / v
            temp[i, 4] = (temp[i, 4] - 0) / qin
            temp[i, 5] = (temp[i, 5] - 0) / coating
            temp[i, 6] = (temp[i, 6] - 0) / morph

    else:
        for k in range(len(temp)):
            temp[k][0] = temp[k][0] / Temp  # (temp[k][0] - Me) / Se
            temp[k][1] = np.log(temp[k][1])
            temp[k][1] = (temp[k][1] - Mt) / St
    return temp

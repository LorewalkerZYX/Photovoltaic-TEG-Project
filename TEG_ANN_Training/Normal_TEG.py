import numpy as np

# Log normal data
M1 = -0.505981311426065  # PDmax mean
S1 = 2.155558247522873  # PDmax std
# Me = 14.2077
# Se = 11.0996
M2 = 5.811022220803706  # Tpv mean
S2 = 0.113912923738362  # Tpv std
'''
# normal data
M1 = 1.925777027326522  # PDmax mean
S1 = 2.662560640450558  # PDmax std
# Me = 14.2077
# Se = 11.0996
M2 = 336.173497244210  # Tpv mean
S2 = 39.522300250047400  # Tpv std
# Temp = 37
'''

def recover(y):
    # y /= const
    for i in range(len(y)):
        # y[i, 0] = y[i, 0] * 33
        # y[i, 1] = y[i, 1] * 594 + 263.15
        
        y[i, 0] = y[i, 0] * S1 + M1
        y[i, 1] = y[i, 1] * S2 + M2
    outy = np.exp(y)
    
    return outy


def normalize(x, input=True):
    temp = x

    if input:
        qin = 1000  # qin = [0-1000]
        Wn = 8  # Wn = [1-9]
        Wp = 8  # Wn = [1-9]
        hte = 25  # Wn = [5-30]
        rhoc_e = 9.9e-8  # rhoc_e = [1e-9 - 1e-7]
        rhoc_t = 9.9e-3  # rhoc_t = [1e-6 - 1e-4]
        Tamb = 100  # ff = [263.15-363.15]
        Conv = 24  # Convection = [1- 25]

        for i in range(len(temp)):
            temp[i, 0] = (temp[i, 0] - 0) / qin
            temp[i, 1] = (temp[i, 1] - 1) / Wn
            temp[i, 2] = (temp[i, 2] - 1) / Wp
            temp[i, 3] = (temp[i, 3] - 5) / hte
            temp[i, 4] = (temp[i, 4] - 1e-9) / rhoc_e
            temp[i, 5] = (temp[i, 5] - 1e-6) / rhoc_t
            temp[i, 6] = (temp[i, 6] - 263.15) / Tamb
            temp[i, 7] = (temp[i, 7] - 1) / Conv

    else:
        for k in range(len(temp)):
            # temp[k][0] = temp[k][0] / 33
            # temp[k][1] = (temp[k][1] - 263.15) / 594
            
            temp[k][0] = np.log(temp[k][0])
            temp[k][0] = (temp[k][0] - M1) / S1
            temp[k][1] = np.log(temp[k][1])
            temp[k][1] = (temp[k][1] - M2) / S2
            
    return temp

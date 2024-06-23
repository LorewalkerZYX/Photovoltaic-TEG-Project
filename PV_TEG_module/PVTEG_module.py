# PVTEG module
# Avalable on https://github.com/LorewalkerZYX/Photovoltaic-TEG-Project.git
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sko.GA import GA
import xlsxwriter
import time
import random
from scipy.interpolate import interp1d
from scipy import integrate

Batch_size = 64
epoch = 2000
learning_rate = 0.001
Morphology = ['Planar', 'Upright pyramids', 'V grooves', 'Spherical caps']
Coating = ['NO', 'YES']

M1 = -0.505981311426065  # PDmax mean
S1 = 2.155558247522873  # PDmax std
M2 = 5.811022220803706  # Tpv mean
S2 = 0.113912923738362  # Tpv std

POWER = pd.read_excel('POWER1.xlsx')

POWER = POWER.iloc[:, 4:]
POWER = POWER.to_numpy()
print(POWER)


'''
M1 = -0.660649080962676  # PDmax mean
S1 = 2.178540029563096  # PDmax std
M2 = 5.811647554754661  # Tpv mean
S2 = 0.114118136985078  # Tpv std
Mt = 5.899922758130535
St = 0.223066273897046
'''


def recover_pv_h(y):
    # y /= const
    y[0] = y[0] * 37  # * Se + Me
    y[1] = y[1] * St + Mt
    y[1] = np.exp(y[1])
    return y


def recover_pv(y):
    # y /= const
    y[0] = y[0] * 37  # * Se + Me
    return y


def recover_teg(y):
    # y /= const
    y[0] = y[0] * S1 + M1
    y[1] = y[1] * S2 + M2
    outy = np.exp(y)

    return outy


def normalize_pv(x, input=True):
    temp = x

    v = 0.67  # hte = [0-0.67]
    T = 100  # ff = [263.15-363.15]
    solar = 1000  # n_ratio = [0-1000]
    coating = 1  # p_ratio = [0 1]
    morph = 3  # rhoc_h = [0 1 2 3]
    temp[0] = (temp[0] - 0) / v
    temp[1] = (temp[1] - 263.15) / T
    temp[2] = (temp[2] - 0) / solar
    temp[3] = (temp[3] - 0) / coating
    temp[4] = (temp[4] - 0) / morph
    return temp


def normalize_pv_h(x, input=True):
    temp = x

    v = 0.64  # hte = [0-0.67]
    T = 100  # ff = [263.15-363.15]
    solar = 1000  # n_ratio = [0-1000]
    coating = 1  # p_ratio = [0 1]
    morph = 3  # rhoc_h = [0 1 2 3]
    Conv = 24  # Conv = [1-25]
    qin = 1000  # n_ratio = [0-1000]

    temp[0] = (temp[0] - 0) / solar
    temp[1] = (temp[1] - 263.15) / T
    temp[2] = (temp[2] - 1) / Conv
    temp[3] = (temp[3] - 0) / v
    temp[4] = (temp[4] - 0) / qin
    temp[5] = (temp[5] - 0) / coating
    temp[6] = (temp[6] - 0) / morph
    return temp


def normalize_teg(x):
    temp = x
    qin = 1000  # qin = [0-1000]
    Wn = 8  # Wn = [1-9]
    Wp = 8  # Wn = [1-9]
    hte = 25  # Wn = [5-30]
    rhoc_e = 9.9e-8  # rhoc_e = [1e-9 - 1e-7]
    rhoc_t = 9.9e-3  # rhoc_t = [1e-6 - 1e-4]
    Tamb = 100  # ff = [263.15-363.15]
    Conv = 24  # Convection = [1- 25]
    temp[0] = (temp[0] - 0) / qin
    temp[1] = (temp[1] - 1) / Wn
    temp[2] = (temp[2] - 1) / Wp
    temp[3] = (temp[3] - 5) / hte
    temp[4] = (temp[4] - 1e-9) / rhoc_e
    temp[5] = (temp[5] - 1e-6) / rhoc_t
    temp[6] = (temp[6] - 263.15) / Tamb
    temp[7] = (temp[7] - 1) / Conv
    return temp

# create net


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_layer):
        super(Net, self).__init__()
        self.input = nn.Linear(n_feature, n_hidden)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(n_hidden, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(n_hidden, n_output)
        self.layernum = n_layer

    def forward(self, x):
        out = self.input(x)
        out = self.relu(out)
        for i in range(self.layernum):
            out = self.hidden(out)
            out = self.relu(out)
        out = self.out(out)
        return out


PV = Net(5, 400, 1, 5)
PV.load_state_dict(torch.load('TEGPV_testRall.pkl'))
PV_H = Net(7, 700, 2, 5)
PV_H.load_state_dict(torch.load('PV_H_N700L5.pkl'))
TEG = Net(8, 700, 2, 4)
TEG.load_state_dict(torch.load('TEG_test0.pkl'))


def PV_TEG(S, Tamb, volt, coat, morph, Wn, Wp, Hte, Rhoce, Rhoct, Conv):
    T0 = Tamb
    Probe = [volt, T0, S, coat, morph]
    temp = Probe.copy()
    PV_in = normalize_pv(temp)
    pv_torch = torch.Tensor(PV_in)
    temp0 = PV(pv_torch)
    temp0 = temp0.cpu().data.numpy()
    I0 = recover_pv(temp0)  # mA/cm^2
    P0 = I0[0] * volt * 10  # W/m^2
    Probe1 = [S * (1 - P_non) - P0, Wn, Wp, Hte, Rhoce, Rhoct, Tamb, Conv]
    temp1 = Probe1.copy()
    TEG_in = normalize_teg(temp1)
    teg_torch = torch.Tensor(TEG_in)
    temp2 = TEG(teg_torch)
    temp2 = temp2.cpu().data.numpy()
    TEG0 = recover_teg(temp2)  # [W/m^2, K]
    Tpv = TEG0[1]
    P_teg = TEG0[0]

    while abs(Tpv - T0) > 0.001:
        # print(P0, I0[0], P_teg, Tpv, T0)
        T0 = Tpv
        Probe = [volt, T0, S, coat, morph]
        temp = Probe.copy()
        PV_in = normalize_pv(temp)
        pv_torch = torch.Tensor(PV_in)
        temp0 = PV(pv_torch)
        temp0 = temp0.cpu().data.numpy()
        I0 = recover_pv(temp0)  # mA/cm^2
        P0 = I0[0] * volt * 10  # W/m^2
        Probe1 = [S * (1 - P_non) - P0, Wn, Wp, Hte, Rhoce, Rhoct, Tamb, Conv]
        temp1 = Probe1.copy()
        TEG_in = normalize_teg(temp1)
        teg_torch = torch.Tensor(TEG_in)
        temp2 = TEG(teg_torch)
        temp2 = temp2.cpu().data.numpy()
        TEG0 = recover_teg(temp2)  # [W/m^2, K]
        Tpv = TEG0[1]
        P_teg = TEG0[0]
    return P0, I0[0], P_teg, TEG0[1], T0


def PV_T(S, Tamb, volt, coat, morph):
    T0 = Tamb
    Probe = [volt, T0, S, coat, morph]
    temp = Probe.copy()
    PV_in = normalize_pv(temp)
    pv_torch = torch.Tensor(PV_in)
    temp0 = PV(pv_torch)
    temp0 = temp0.cpu().data.numpy()
    I0 = recover_pv(temp0)  # mA/cm^2
    P0 = I0[0] * volt * 10  # W/m^2
    return P0, I0[0]


def PV_Heat_uT(S, Tamb, Conv, volt, coat, morph):
    T0 = Tamb
    i = 0
    Probe = [volt, T0, S, coat, morph]
    temp = Probe.copy()
    PV_in = normalize_pv(temp)
    pv_torch = torch.Tensor(PV_in)
    temp0 = PV(pv_torch)
    temp0 = temp0.cpu().data.numpy()
    I0 = recover_pv(temp0)  # mA/cm^2
    P0 = I0[0] * volt * 10  # W/m^2
    T1 = (S * (1 - P_non) - P0) / (2 * Conv) + Tamb
    while abs(T1 - T0) > 0.001:
        # print(T0, P0)
        T0 = T1
        Probe = [volt, T0, S, coat, morph]
        temp = Probe.copy()
        PV_in = normalize_pv(temp)
        pv_torch = torch.Tensor(PV_in)
        temp0 = PV(pv_torch)
        temp0 = temp0.cpu().data.numpy()
        I0 = recover_pv(temp0)  # mA/cm^2
        P0 = I0[0] * volt * 10  # W/m^2
        T1 = (S * (1 - P_non) - P0) / (2 * Conv) + Tamb

    return P0, T1


def PV_Heat(S, Tamb, Conv, volt, coat, morph):
    P0 = 0
    S0 = S * (1 - P_non) - P0
    Probe = [S, Tamb, Conv, volt, S * (1 - P_non) - P0, coat, morph]
    temp = Probe.copy()
    PV_H_in = normalize_pv_h(temp)
    pv_h_torch = torch.Tensor(PV_H_in)
    temp0 = PV_H(pv_h_torch)
    temp0 = temp0.cpu().data.numpy()
    T = recover_pv_h(temp0)  # mA/cm^2
    Tpv = T[1]  # K
    Probe = [volt, Tpv, S, coat, morph]
    temp = Probe.copy()
    PV_in = normalize_pv(temp)
    pv_torch = torch.Tensor(PV_in)
    temp0 = PV(pv_torch)
    temp0 = temp0.cpu().data.numpy()
    I0 = recover_pv(temp0)  # mA/cm^2
    P1 = I0[0] * volt * 10  # W/m^2
    while abs(P1 - P0) > 0.001:
        # print(P0)
        P0 = P1
        S0 = S * (1 - P_non) - P0
        Probe = [S, Tamb, Conv, volt, S * (1 - P_non) - P0, coat, morph]
        temp = Probe.copy()
        PV_H_in = normalize_pv_h(temp)
        pv_h_torch = torch.Tensor(PV_H_in)
        temp0 = PV_H(pv_h_torch)
        temp0 = temp0.cpu().data.numpy()
        T = recover_pv_h(temp0)  # mA/cm^2
        Tpv = T[1]  # K
        Probe = [volt, Tpv, S, coat, morph]
        temp = Probe.copy()
        PV_in = normalize_pv(temp)
        pv_torch = torch.Tensor(PV_in)
        temp0 = PV(pv_torch)
        temp0 = temp0.cpu().data.numpy()
        I0 = recover_pv(temp0)  # mA/cm^2
        P1 = I0[0] * volt * 10  # W/m^2
    return P1, I0[0], T[1]


def MaxPV_TEG(S, Tamb, coat, morph, Wn, Wp, Hte, Rhoce, Rhoct, Conv):
    V = range(0, 64)
    P = 0
    P1 = 0
    P2 = 0
    T1 = 0
    temp = 0
    for i in range(len(V)):
        Output = PV_TEG(S, Tamb, V[i] / 100, coat, morph, Wn, Wp, Hte, Rhoce, Rhoct, Conv)
        Q = Output[0] + Output[2]
        if Output[0] < 0:
            break
        if Q > P:
            P = Q
            P1 = Output[0]  # PV power
            P2 = Output[2]  # TEG Power
            T1 = Output[3]  # PV temperature
            temp = i

    return P, V[temp] / 100, P1, P2, T1


def MaxPV_T(S, Ta, coat, morph):
    V = range(0, 640)
    P = 0
    P1 = 0
    P2 = 0
    T1 = 0
    temp = 0
    for i in range(len(V)):
        Output = PV_T(S, Ta, V[i] / 1000, coat, morph)
        Q = Output[0]
        if Output[0] < 0:
            break
        if Q > P:
            P = Q
            temp = i

    return P, V[temp] / 1000


def MaxPV(S, Tamb, coat, morph, Conv):
    V = range(0, 64)
    P = 0
    P1 = 0
    P2 = 0
    T1 = 0
    temp = 0
    for i in range(len(V)):
        Output = PV_Heat_uT(S, Tamb, Conv, V[i] / 100, coat, morph)
        Q = Output[0]
        if Output[0] < 0:
            break
        if Q > P:
            P = Q
            T1 = Output[1]  # PV temperature
            temp = i

    return P, V[temp] / 1000, T1


Solar = 414.02
T0 = 273.15
Temperature = T0 + 27.74  # [-5, 10, 25, 40]
C = 1
M = 1  # [Planar, upright pyramids, V grooves, spherical caps]
N = 5
P = N
H = 10
Re = 1e-8
Rt = 1e-5
Convection = 10.64
Conv_hs = Convection * 15 / 4
P_non = 0.16


# save in excel
workbook = xlsxwriter.Workbook('./evaluation/PVTEG_realtime_T1_one.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1', 'Solar irradiance')
worksheet.write('B1', 'Tamb')
worksheet.write('C1', 'Coating')
worksheet.write('D1', 'Morphology')
worksheet.write('E1', 'Rhoc_e')
worksheet.write('F1', 'Rhoc_t')
worksheet.write('G1', 'Convection')
worksheet.write('H1', 'Voltage(V)')
worksheet.write('I1', 'PD_pv(W/m^2)')
worksheet.write('J1', 'Current (mA/cm^2)')
worksheet.write('K1', 'PD_teg(W/m^2)')
worksheet.write('L1', 'T_pv(K)')
worksheet.write('M1', 'All Power(W/m^2)')
worksheet.write('N1', 'Average Calculation Time')
worksheet.write('O1', 'Non absorbed ratio')
worksheet.write('P1', 'HTE')
worksheet.write('Q1', 'Average simulation time')
worksheet.write('R1', 'Integrate Power')


R0_PD_sum = 0
R0__PD_sum = 0
R1__PD_sum = 0
Averagetime = 0

S = POWER[:, 0]

T = POWER[:, 2]

WS = 3.96 * POWER[:, 3] / 4 + 5.86

L = len(S)

X1 = np.linspace(0, L, L)

Y1 = np.zeros(L)
print(len(Y1), len(X1))
print(WS)
Y2 = np.zeros(L)
j = 0

# worksheet.write(1, 0, Solar)
# worksheet.write(1, 1, Temperature)
worksheet.write(1, 2, Coating[C])
worksheet.write(1, 3, Morphology[M])
worksheet.write(1, 4, Re)
worksheet.write(1, 5, Rt)
# worksheet.write(1, 6, Convection)
worksheet.write(1, 14, P_non)
worksheet.write(1, 15, H)

for i in range(len(S)):
    start = time.time()
    # Temp = MaxPV_TEG(S[i], T[i], C, M, N, P, H, Re, Rt, WS[i])
    # Temp = PV_TEG(Solar, Temperature, V[i] / 1000, C, M, N, P, H, Re, Rt, Convection)  # PV_TEG
    # Temp = PV_Heat(Solar, Temperature, Convection, V[i] / 1000, C, M)
    if S[i] == 0:
        Temp = [0, 0, 0, 0, T[i] + 273.15]
    else:
        Temp = MaxPV_TEG(S[i], T[i] + 273.15, C, M, N, P, H, Re, Rt, WS[i])
        Y2[j] = Temp[4]
        worksheet.write(j + 1, 18, Y2[j])
        j += 1
        # Temp = MaxPV(S[i], T[i] + 273.15, C, M, WS[i])
    end = time.time()
    Averagetime = Averagetime + end - start
    # print(Temp[2])
    Y1[i] = Temp[0]

    worksheet.write(i + 1, 1, T[i] + 273.15)
    # worksheet.write(i + 1, 5, rho_t[i])
    worksheet.write(i + 1, 6, WS[i])
    worksheet.write(i + 1, 7, Temp[1])
    worksheet.write(i + 1, 8, Temp[2])
    #worksheet.write(i + 1, 9, Temp[2] / Temp[1] / 10)
    worksheet.write(i + 1, 10, Temp[3])
    worksheet.write(i + 1, 11, Temp[4])
    worksheet.write(i + 1, 12, Temp[0])
    worksheet.write(i + 1, 0, S[i])


#Averagetime = Averagetime / (i + 1)
worksheet.write(1, 16, Averagetime)
# v1 = integrate.trapz(Y1, X1)
v2 = integrate.trapz(S, X1)
worksheet.write(1, 17, v2)
print(v2)
workbook.close()

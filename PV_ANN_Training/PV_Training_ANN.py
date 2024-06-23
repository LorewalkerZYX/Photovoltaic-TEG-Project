# PVTEG PV ANN training
# Avalable on https://github.com/LorewalkerZYX/Photovoltaic-TEG-Project.git

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import xlsxwriter
import LoadDataS
import SetSeed
import SplitData
import Normal


# Set the random seed manually for reproducibility.

SetSeed.seed_torch(10)
device = torch.device('cuda:0')

[trainx, trainy, validx, validy, test_x, test_y] = LoadDataS.LoadData()

testing_x = test_x.copy()
testing_y = test_y.copy()

# normalization
trainx = Normal.normalize(trainx)
validx = Normal.normalize(validx)
testx = Normal.normalize(test_x)
trainy = Normal.normalize(trainy, False)
validy = Normal.normalize(validy, False)
Y_test = Normal.normalize(test_y, False)


# dataset
train_size = trainx.shape[0]
valid_size = validx.shape[0]

Batch_size = 64
epoch = 2000
learning_rate = 0.001
hidden_layers = 5
hidden_feature = 800
n = 0
step = 1
# print(train_size, valid_size)

# trainsfer numpy to torch
x = torch.from_numpy(trainx)
x = x.type(torch.FloatTensor)

y = torch.from_numpy(trainy)
y = y.type(torch.FloatTensor)

X_dev = torch.from_numpy(validx)
X_dev = X_dev.type(torch.FloatTensor)

Y_dev = torch.from_numpy(validy)
Y_dev = Y_dev.type(torch.FloatTensor)

train_data = Data.TensorDataset(x, y)
val_data = Data.TensorDataset(X_dev, Y_dev)

X_test = torch.from_numpy(testx)
X_test = X_test.type(torch.FloatTensor)


loader = Data.DataLoader(
    dataset=train_data,
    batch_size=Batch_size,
    shuffle=True,
)

val_loader = Data.DataLoader(
    dataset=val_data,
    batch_size=Batch_size,
    shuffle=False
)

SetSeed.seed_torch(42)  # 58


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


Loss_Function = nn.MSELoss()

net = Net(7, hidden_feature, 2, hidden_layers)
net = net.to(device)
optimzer = torch.optim.Adam(
    net.parameters(),
    lr=learning_rate
    # weight_decay=0.001
)

stepsize = [1900]

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optimzer,
    milestones=stepsize,
    gamma=0.1
)


# save in excel
workbook = xlsxwriter.Workbook('PV_H_N%dL%dSeed=42.xlsx' %
                               (hidden_feature, hidden_layers))
worksheet = workbook.add_worksheet()
# worksheet2 = workbook.add_worksheet()


worksheet.write('A1', 'epoch')
worksheet.write('B1', 'training loss')
worksheet.write('C1', 'validation loss')
worksheet.write('D1', 'Test Current Data')
worksheet.write('F1', 'Predict Current Data')
worksheet.write('E1', 'Test Temperature')
worksheet.write('G1', 'Predict Temperature')
worksheet.write('H1', 'Current Relative error')
worksheet.write('I1', 'Temperature Relative error')
worksheet.write('J1', 'Current Average Relative error')
worksheet.write('K1', 'Temperature Average Relative error')


def TrainGA(epoch):
    # seed_torch(sd)
    for i in range(epoch):
        train_loss = 0.0
        val_loss = 0.0
        temp_loss = 0.0
        temp_val = 0.0

        net.train()
        for num, (batch_x, batch_y) in enumerate(loader):
            optimzer.zero_grad()
            out = net(batch_x.to(device))
            loss = Loss_Function(out, batch_y.to(device))
            loss.backward()
            optimzer.step()
            temp_loss += loss.item()
        scheduler.step()
        train_loss = temp_loss / (train_size / Batch_size)
        net.eval()
        with torch.no_grad():
            for epnum, (val_x, val_y) in enumerate(val_loader):
                val_out = net(val_x.to(device))
                dev_loss = Loss_Function(val_out, val_y.to(device))
                temp_val += dev_loss.cpu().data.numpy()

        val_loss = temp_val / (valid_size / Batch_size)
        print('epoch: %d' % i, 'training loss:', train_loss, '|',
              'validation loss:', val_loss)
        worksheet.write(i + 1, 0, i + 1)
        worksheet.write(i + 1, 1, train_loss)
        worksheet.write(i + 1, 2, val_loss)
    return train_loss


# start training
TrainGA(epoch)
torch.save(net.state_dict(), 'PV_H_N%dL%d.pkl' %
           (hidden_feature, hidden_layers))
# test data
test_out = net(X_test.to(device))
t_out = test_out.cpu().data.numpy()

Predict_y = np.abs(Normal.recover(t_out))

lengthT = len(t_out)
Ap = 0
Aq = 0
for j in range(lengthT):
    worksheet.write(j + 1, 3, testing_y[j, 0])
    worksheet.write(j + 1, 4, testing_y[j, 1])
    worksheet.write(j + 1, 5, Predict_y[j, 0])
    worksheet.write(j + 1, 6, Predict_y[j, 1])
    RelativeE_PD = np.abs(Predict_y[j, 0] - testing_y[j, 0]) / testing_y[j, 0]
    RelativeE_T = np.abs(Predict_y[j, 1] - testing_y[j, 1]) / testing_y[j, 1]
    worksheet.write(j + 1, 7, RelativeE_PD)
    worksheet.write(j + 1, 8, RelativeE_T)
    Ap += RelativeE_PD
    Aq += RelativeE_T

Aq /= lengthT
Ap /= lengthT
worksheet.write(1, 9, Ap)
worksheet.write(1, 10, Aq)
workbook.close()

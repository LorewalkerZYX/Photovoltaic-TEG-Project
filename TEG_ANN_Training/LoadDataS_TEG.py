import pandas as pd


def LoadData():
    # preparing data
    Train_X = pd.read_excel('Train_In.xlsx')
    Train_Y = pd.read_excel('Train_Out.xlsx')
    Valid_X = pd.read_excel('Valid_In.xlsx')
    Valid_Y = pd.read_excel('Valid_Out.xlsx')
    Test_X = pd.read_excel('TestR0_In.xlsx')
    Test_Y = pd.read_excel('TestR0_Out.xlsx')
    TrainX = Train_X.iloc[:, :]
    TrainY = Train_Y.iloc[:, :]
    ValidX = Valid_X.iloc[:, :]
    ValidY = Valid_Y.iloc[:, :]
    TestX = Test_X.iloc[:, :]
    TestY = Test_Y.iloc[:, :]

    X_train = TrainX.to_numpy()
    Y_train = TrainY.to_numpy()
    X_Valid = ValidX.to_numpy()
    Y_Valid = ValidY.to_numpy()
    X_Test = TestX.to_numpy()
    Y_Test = TestY.to_numpy()
    return X_train, Y_train, X_Valid, Y_Valid, X_Test, Y_Test

import os
import pandas as pd
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary as summary
import pennylane as qml
from create_data import *
from loss_function import *


# Create model with input is beta (M,K)
# Pre-processing: Hidden layer (M*K, 8)
# QCNN
# Post-processing: Hidden layer (4, K*tau_p)

# Output reshape (K, tau_p) --> softmax --> argmax --> become pilot_index (K,)
# Calculate rate_dl with pilot_index and beta --> sumrate
# Calculate chanel-estimation error

# Un-supervised learning so need to make sure output is still save the gradient for model to update

# loss-function is -(rate_dl)


# Read data
def import_from_files(args):
    filename = f'{args.number_UE}UE_{args.number_AP}AP_{args.pilot_length}pilot.csv'
    file_path = os.path.join(filename)
    if os.path.isfile(file_path):
        # Load the CSV file
        data = pd.read_csv(file_path)
        print(f"Data loaded from {file_path}")
    else:
        raise ValueError("No dataset option or file not found")
    return data


def dataloader_creation(args):
    data = import_from_files(args)
    data = np.array(data)
    train_data = data[:8, :]
    test_data = data[8:4+8, :]
    sum_dl_rate = data[:, 0]
    mean_dl_rate = np.mean(sum_dl_rate)
    min_rate_list = data[:, 1]
    mean_min_list = np.mean(min_rate_list)
    max_rate_list = data[:, 1]
    mean_max_list = np.mean(max_rate_list)
    x_train = train_data[:, 3: args.number_AP * args.number_UE + 3]
    x_test = test_data[:, 3: args.number_AP * args.number_UE + 3]
    X_train = torch.tensor(x_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)

    y_train = train_data[:, args.number_AP * args.number_UE + 3:].astype(np.int64)
    Y_train = torch.unsqueeze(torch.tensor(y_train), 2)
    Y_train_hot = torch.scatter(torch.zeros((len(x_train), args.number_UE, args.pilot_length)), 2, Y_train, 1)
    y_test = test_data[:, args.number_AP * args.number_UE + 3:].astype(np.int64)
    Y_test = torch.unsqueeze(torch.tensor(y_test), 2)
    Y_test_hot = torch.scatter(torch.zeros((len(x_test), args.number_UE, args.pilot_length)), 2, Y_test, 1)
    # train_dataset = TensorDataset(X_train)
    train_dataset = list(X_train)
    test_dataset = list(X_test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return mean_dl_rate, mean_min_list, mean_max_list, train_dataloader, test_dataloader


# Define model

n_qubits = 8
weight_shapes = {
    "weights_0": 3,
    "weights_1": 3,
    "weights_2": 1,
    "weights_3": 1,
    "weights_4": 1,
    "weights_5": 3,
    "weights_6": 3,
    "weights_7": 1,
    "weights_8": 1,
}
Pooling_out = [1, 3, 5, 7]
dev = qml.device("default.qubit", wires=n_qubits)


def U_4(params, wires):  # 3 params 2 qubit
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])  # (source, target)
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(params[2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi / 2, wires=wires[0])


def conv_layer1(U, params, Uname):
    if Uname == 'U_4':  # parameter 3
        U(params[0:3], wires=[0, 1])
        U(params[3:6], wires=[2, 3])
        U(params[6:9], wires=[4, 5])
        U(params[9:12], wires=[6, 7])
        U(params[12:15], wires=[8, 9])

        U(params[15:18], wires=[1, 2])
        U(params[18:21], wires=[3, 4])
        U(params[21:24], wires=[5, 6])
        U(params[24:27], wires=[7, 8])
        U(params[27:30], wires=[9, 0])


def U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
          wires):  # 15 params, Convolutional Circuit 10
    qml.U3(*weights_0, wires=wires[0])
    qml.U3(*weights_1, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights_2, wires=wires[0])
    qml.RZ(weights_3, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(weights_4, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(*weights_5, wires=wires[0])
    qml.U3(*weights_6, wires=wires[1])


# Unitary Ansatz for Pooling Layer
def Pooling_ansatz1(weights_0, weights_1, wires):  # 2 params
    qml.CRZ(weights_0, wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(weights_1, wires=[wires[0], wires[1]])


@qml.qnode(dev)
def qnode(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7,
          weights_8):  # , weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, weights_16, weights_17
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # QCNN
    # --------------------------------------------------------- Convolutional Layer1 ---------------------------------------------------------#
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])

    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])

    # --------------------------------------------------------- Pooling Layer1 ---------------------------------------------------------#
    ## Pooling Circuit  Block 2 weights_7, weights_8
    Pooling_ansatz1(weights_7, weights_8, wires=[0, 1])
    Pooling_ansatz1(weights_7, weights_8, wires=[2, 3])
    Pooling_ansatz1(weights_7, weights_8, wires=[4, 5])
    Pooling_ansatz1(weights_7, weights_8, wires=[6, 7])

    result = [qml.expval(qml.PauliZ(wires=i)) for i in Pooling_out]
    return result





# Problem 1: No. dimension of loss_function
# Problem 2: how to keep gradient property of model
def model_training(args, train_dataloader, test_dataloader):
    clayer_1 = torch.nn.Linear(args.number_AP * args.number_UE, 8)
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    clayer_2 = torch.nn.Linear(4, args.number_UE * args.pilot_length)
    layers = [clayer_1, qlayer, clayer_2]
    model = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # loss = torch.nn.CrossEntropyLoss()
    summary(model, (args.number_AP * args.number_UE,))
    epochs = args.epochs
    dl_rate_train = np.zeros(epochs)
    dl_rate_test = np.zeros(epochs)
    start_time = time.time()
    for epoch in range(epochs):
        running_loss_train = 0
        running_loss_test = 0
        model.train()
        for i, xs in enumerate(train_dataloader):
            opt.zero_grad()
            output_QCNN_train = model(xs)
            loss = loss_function_PA(args, xs, output_QCNN_train, epoch, i)  # mean of loss_value in batch_item
            print(f'LOSS: {loss}')
            loss.backward()
            # print(f'LOSS-BACKWARD: {loss.backward()}')
            opt.step()
            running_loss_train += loss.item()  # Cumulative loss_function of each batch in whole train_dataloader in float type
        avg_loss = running_loss_train / len(train_dataloader)
        print(f"[{epoch}]/[{epochs}] Loss_evaluated_train: {avg_loss}")
        dl_rate_train[epoch] = avg_loss * (-1)

        model.eval()
        with torch.no_grad():
            for i, xt in enumerate(test_dataloader):
                output_QCNN_test = model(xt)
                loss_evaluated_test = loss_function_PA(args, xt, output_QCNN_test, epoch, i)
                running_loss_test += loss_evaluated_test.item()
            avg_loss_test = running_loss_test / len(test_dataloader)
            print(f"[{epoch}]/[{epochs}] Loss_evaluated_testing: {avg_loss_test}")
            dl_rate_test[epoch] = avg_loss_test * (-1)

        # print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))
    end_time = time.time()
    running_time = end_time - start_time
    print(f'Running time: {running_time: .4f} seconds.')
    return dl_rate_train, dl_rate_test

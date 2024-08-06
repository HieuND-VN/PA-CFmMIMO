import os
import numpy as np
from numpy import vstack
import pandas as pd
from torchsummary import summary as summary
import pennylane as qml
from pennylane import numpy as np
import torch



class HQCNN():
    def __init__(self, args):
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.num_ap = args.num_ap
        self.num_ue = args.num_ue
        self.tau_p = args.tau_p














device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires):  # 15 params, Convolutional Circuit 10
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




n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)
Pooling_out = [1, 3, 5, 7]

@qml.qnode(dev)
def qnode(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8):  # , weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, weights_16, weights_17
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    # qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    # QCNN
    # --------------------------------------------------------- Convolutional Layer1 ---------------------------------------------------------#
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])
    # U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[8, 9])

    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])
    # U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 8])
    # U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[9, 0])

    # --------------------------------------------------------- Pooling Layer1 ---------------------------------------------------------#
    ## Pooling Circuit  Block 2 weights_7, weights_8
    Pooling_ansatz1(weights_7, weights_8, wires=[0, 1])
    Pooling_ansatz1(weights_7, weights_8, wires=[2, 3])
    Pooling_ansatz1(weights_7, weights_8, wires=[4, 5])
    Pooling_ansatz1(weights_7, weights_8, wires=[6, 7])
    # Pooling_ansatz1(weights_7, weights_8, wires=[8,9])

    # --------------------------------------------------------- Convolutional Layer2 ---------------------------------------------------------#
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[1, 3])
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[5, 7])
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[3, 5])
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[7, 9])
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[9, 1])

    ##--------------------------------------------------------- Pooling Layer2 ---------------------------------------------------------#
    ### Pooling Circuit  Block 2 weights_7, weights_8
    # Pooling_ansatz1(weights_16, weights_17, wires=[1,3])
    # Pooling_ansatz1(weights_16, weights_17, wires=[3,5])
    # Pooling_ansatz1(weights_16, weights_17, wires=[5,7])
    # Pooling_ansatz1(weights_16, weights_17, wires=[7,9])

    # conv_layer1(U, params9, U2)
    # pooling_layer1(Pooling_ansatz1, params5, V)
    result = [qml.expval(qml.PauliZ(wires=i)) for i in Pooling_out]
    return result


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

clayer_1 = torch.nn.Linear(1016, 8)
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
clayer_2 = torch.nn.Linear(4, 2)
# clayer_3 = torch.nn.Linear(128, 2)
softmax = torch.nn.Softmax(dim=1)  # torch.nn.sigmoid()
layers = [clayer_1, qlayer, clayer_2, softmax]  # clayer_1,
model = torch.nn.Sequential(*layers)  # .to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()

summary(model, (1016,))
batch_size = 64  # 5
batches = 33600 // batch_size  # 200 // batch_size

data_loader = torch.utils.data.DataLoader(list(zip(X_train, Y_train_hot)), batch_size=64, shuffle=True, drop_last=True)
val_data_loder = torch.utils.data.DataLoader(list(zip(X_val, Y_val_hot)), batch_size=64, shuffle=True, drop_last=True)
epochs = 10
count = 0
for epoch in range(epochs):
    running_loss = 0
    running_loss_val = 0
    model.train()
    for xs, ys in data_loader:
        count += 1
        if count % 10 == 0:
            print(count)
        opt.zero_grad()

        loss_evaluated = loss(model(xs), ys)
        loss_evaluated.backward()
        opt.step()

        # running_loss += loss_evaluated
        running_loss += loss_evaluated.item()
        # print("loss_evaluated:", loss_evaluated)
    avg_loss = running_loss / batches
    print(f"[{epoch}]/[{epochs}] Loss_evaluated: {avg_loss}")

    model.eval()
    with torch.no_grad():
        y_pred_tr = model(X_train)
        predictions_tr = torch.argmax(y_pred_tr, axis=1).detach().numpy()
        correct_tr = [1 if p == p_true else 0 for p, p_true in zip(predictions_tr, Y_train)]
        accuracy_tr = sum(correct_tr) / len(correct_tr)
        print("training_Accuracy : ", accuracy_tr)
        for xv, yv in val_data_loder:
            loss_evaluated_val = loss(model(xv), yv)
            running_loss_val += loss_evaluated_val.item()
        avg_loss_val = running_loss_val / batches * 8
        y_pred_val = model(X_val)
        predictions_val = torch.argmax(y_pred_val, axis=1).detach().numpy()
        correct_val = [1 if p == p_true else 0 for p, p_true in zip(predictions_val, Y_val)]
        accuracy_val = sum(correct_val) / len(correct_val)
        print("Validation_Accuracy : ", accuracy_val)
    f = open("test10.txt", 'a')
    f.write("Epoch : %f\n" % epoch)
    f.write("Avg_loss_traing: %f\n" % avg_loss)
    f.write("Acc_training: %f\n" % accuracy_tr)
    f.write("Avg_loss_validation: %f\n" % avg_loss_val)
    f.write("Acc_validation: %f \n" % accuracy_val)
    f.close()
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    predictions = torch.argmax(y_pred, axis=1).detach().numpy()
correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, Y_test)]
accuracy = sum(correct) / len(correct)
print(f"Accuracy: {accuracy * 100}%")
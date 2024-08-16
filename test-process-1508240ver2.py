import numpy as np
import torch
from torch.utils.data import DataLoader
import pennylane as qml
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

D = 1  # in kilometer
d1 = 0.05  # in kilometer
d0 = 0.01  # in kilometer
h_ap = 15  # in meter
h_ue = 1.7  # in meter
B = 20  # in MHz
f = 1900  # in MHz
L = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_ap) - (1.1 * np.log10(f) - 0.7) * h_ue + (1.56 * np.log10(f) - 0.8)
P_d = 0.2  # downlink power: 200 mW
p_u = 0.1  # uplink power: 100 mW
p_p = 0.1  # pilot power: 100mW
noise_figure = 9  # dB
T = 290  # noise temperature in Kelvin
noise_power = (B * 10 ** 6) * (1.381 * 10 ** (-23)) * T * (10 ** (noise_figure / 10))  # Thermal noise in W
rho = 1/noise_power
# rho_d = P_d / noise_power
rho_d = 1
# rho_u = rho_p = p_u / noise_power
rho_u = rho_p = 1/2
sigma_shd = 8  # in dB
D_cor = 0.1
tau_c = 200

num_train = 192
num_test = 16

num_ue = 20
num_ap  = 40
tau_p = 10
total_sample = num_train  + num_test
data = np.zeros((total_sample, num_ue * num_ap))
for i in range(total_sample):
    AP_position = np.random.uniform(-D, D, size=(num_ap, 2))
    UE_position = np.random.uniform(-D, D, size=(num_ue, 2))

    AP_expanded = AP_position[:, np.newaxis, :]  # Shape: (num_ap, 1, 2)
    UE_expanded = UE_position[np.newaxis, :, :]

    distanceUE2AP = np.sqrt(np.sum((AP_expanded - UE_expanded) ** 2, axis=2))
    pathloss = np.zeros_like(distanceUE2AP)
    pathloss[(distanceUE2AP < d0)] = -L - 15 * np.log10(d1) - 20 * np.log10(d0)
    pathloss[(distanceUE2AP >= d0) & (distanceUE2AP <= d1)] = -L - 15 * np.log10(d1) - 20 * np.log10(
        distanceUE2AP[(distanceUE2AP >= d0) & (distanceUE2AP <= d1)])
    pathloss[(distanceUE2AP > d1)] = -L - 35 * np.log10(distanceUE2AP[(distanceUE2AP > d1)]) + np.random.normal(0,1) * 8
    beta = 10 ** (pathloss / 10) * rho
    # if i == 0:
    #     print(beta)
    data[i] = beta.flatten()

batch_sz = 16
data_train = data[:num_train]
data_test = data[num_train:]
X_train = torch.tensor(data_train).float()
X_test = torch.tensor(data_test).float()
train_dataset = list(X_train)
test_dataset = list(X_test)
train_dataloader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, drop_last=True)

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


def dl_rate_calculator_w_pilot_probs(pilot_probs, beta, tau_p, epoch, batch_no,sample):
    num_ap, num_ue = beta.shape
    phi_inner_product = torch.matmul(pilot_probs, pilot_probs.T)
    phi_inner_product_squared = phi_inner_product
    inner_sum = torch.matmul(beta, phi_inner_product_squared)
    numerator_c = torch.sqrt(torch.tensor(rho_p * tau_p, dtype=torch.float32)) * beta  # Shape: (M, K)
    denominator_c = tau_p * rho_p * inner_sum + rho
    c_mk = numerator_c / denominator_c
    gamma = torch.sqrt(torch.tensor(tau_p * rho_p, dtype=torch.float32)) * beta * c_mk
    etaa = 1 / torch.sum(gamma, dim=1)
    eta = etaa[:, None].expand(-1, num_ue)
    DS = rho_d * (torch.sum(torch.sqrt(eta) * gamma, dim=0)) ** 2  # Shape (num_ue,)
    eta_gamma = eta * gamma
    product = torch.sum(eta_gamma[:, :, None] * beta[:, None, :], dim=0)
    BU = torch.sum(product, dim=0)
    UI = torch.zeros(num_ue, device=beta.device)
    flag = torch.matmul(pilot_probs, pilot_probs.T)
    flag = flag.float().to(dtype=torch.float32)
    for k in range(num_ue):
        sum_term_1 = 0
        for j in range(num_ue):
            sum_term_2 = 0
            if j!=k:
                for m in range(num_ap):
                    sum_term_2 += (torch.sqrt(eta[m,j])*gamma[m,j]*beta[m,k])/(beta[m,j])

            sum_term_1 += (sum_term_2**2)*(flag[k,j]**2)
        UI[k] = rho_d * sum_term_1

    sinr = DS / (UI + BU + rho**2)
    # if epoch == 0 and batch_no == 0 and sample == 0:
        # print(f'[{epoch}]\[{batch_no}]\[{sample}]: numerator_c: \n{numerator_c}\n')
        # print(f'[{epoch}]\[{batch_no}]\[{sample}]: denomirator_c: \n{denominator_c}\n')
        # print(f'[{epoch}]\[{batch_no}]\[{sample}]: c_mk: \n{c_mk}\n')
        # print(f'[{epoch}]\[{batch_no}]\[{sample}]: gamma: \n{gamma}\n')
        # print(f'[{epoch}]\[{batch_no}]\[{sample}]: eta: \n{eta}\n')
        # print(f'[{epoch}]\[{batch_no}]\[{sample}]: DS: \n{DS}\n')
        # print(f'[{epoch}]\[{batch_no}]\[{sample}]: UI: \n{UI}\n')
        # print(f'[{epoch}]\[{batch_no}]\[{sample}]: BU: \n{BU}\n')
        # print(f'[{epoch}]\[{batch_no}]\[{sample}]: sinr: \n{sinr}\n')
    return torch.log2(1 + sinr)

def loss_function_PA(beta, output, epoch, batch_no):
    # batch_sz = 1
    sum_rate_each_batch = torch.zeros(batch_sz)
    beta = beta.reshape(batch_sz, num_ap, num_ue)
    pilot_probs_1 = output.reshape(batch_sz, num_ue, tau_p)
    pilot_probs = F.softmax(pilot_probs_1, dim=-1)

    pilot_index = torch.argmax(pilot_probs, dim=-1) # loss grad_fn from here
    torch_one_hot = F.one_hot(pilot_index, num_classes = tau_p)

    pilot_probs_process = pilot_probs * torch_one_hot
    # print(f'[{epoch}]\[{batch_no}]: Output of model: \n{pilot_probs_1}')
    # print(f'[{epoch}]\[{batch_no}]: Probability of output: \n{pilot_probs} \n:{pilot_index}')
    # print(f'============================')
    # if epoch == 0 and batch_no ==0:
    #     print(f'[{epoch}]\[{batch_no}]: beta inside loss_function: \n{beta}\n')
    #     print(f'[{epoch}]\[{batch_no}]:output after reshape: \n{pilot_probs_1}\n')
    #     print(f'[{epoch}]\[{batch_no}]:output after softmax: \n{pilot_probs}\n')
    #     print(f'[{epoch}]\[{batch_no}]:output multiply with one-hot-coding: \n{pilot_probs_process}\n')
    #     print(f'==========================================================================')
    for i in range(batch_sz):
        sum_rate_each_batch[i] = torch.sum(
            dl_rate_calculator_w_pilot_probs(pilot_probs[i], beta[i], tau_p, epoch, batch_no, i))
        # print(f'[{epoch}]\[{batch_no}]Sum-rate of each batch: {sum_rate_each_batch}\n')
    return torch.mean(sum_rate_each_batch)*(-1)

class CustomModel(nn.Module):
    def __init__(self, num_ap, num_ue, tau_p, qnode, weight_shapes):
        super(CustomModel, self).__init__()

        # Define layers
        self.clayer_1 = nn.Linear(num_ap * num_ue, 8)
        # Placeholder for QML layer
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer_2 = nn.Linear(4, num_ue * tau_p)

    def forward(self, x):
        # print(f'First: x: {x}')
        # print("Input contains NaN values:", torch.isnan(x).any())
        x = self.clayer_1(x)
        # if torch.isnan(x).any():
        #     print("NaNs detected after clayer_1")
        # print(f'Second: x: {x}')
        x = self.qlayer(x)
        # print(f'Third: x: {x}')
        # if torch.isnan(x).any():
        #     print("NaNs detected after qlayer")
        x = self.clayer_2(x)
        # print(f'Fourth: x: {x}')
        # if torch.isnan(x).any():
        #     print("NaNs detected after clayer_2")
        return x

no_epochs = 1
def train_model(model, train_dataloader, test_dataloader, lr=0.01):
    opt = optim.Adam(model.parameters(), lr=lr)
    epochs = no_epochs
    for epoch in range(epochs):
        running_loss_train = 0
        running_loss_test = 0
        model.train()
        for i, xs in enumerate(train_dataloader):
            opt.zero_grad()
            output_QCNN_train = model(xs)
            loss = loss_function_PA(xs, output_QCNN_train, epoch, i)  # mean of loss_value in batch_item
            loss.backward()
            opt.step()
            running_loss_train += loss.item()  # Cumulative loss_function of each batch in whole train_dataloader in float type
        avg_loss = running_loss_train / len(train_dataloader)
        print(f"--------> [{epoch}]/[{epochs}] >>Loss_evaluated_train: {avg_loss}")
    #     model.eval()
    #     with torch.no_grad():
    #         for i, xt in enumerate(test_dataloader):
    #             output_QCNN_test = model(xt)
    #             loss_evaluated_test = loss_function_PA(xt, output_QCNN_test, epoch, i)
    #             running_loss_test += loss_evaluated_test.item()
    #         avg_loss_test = running_loss_test / len(test_dataloader)
    #         # print(f"[{epoch}]/[{epochs}] Loss_evaluated_testing: {avg_loss_test}")
    #     # print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))
    # end_time = time.time()
    # running_time = end_time - start_time
    # print(f'Running time: {running_time: .4f} seconds.')

model = CustomModel(num_ap, num_ue, tau_p, qnode, weight_shapes)
train_model(model, train_dataloader, test_dataloader)
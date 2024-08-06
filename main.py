import time
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torchsummary import summary as summary
from SystemModel import SysMod
from Data import generate_dataset
if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('-num_ap', "--number_AP", type=int, default=40, help="Number of Access Point.")
    parser.add_argument('-num_ue', "--number_UE", type=int, default=30, help="Number of User Equipment.")
    parser.add_argument('-tau_p', "--pilot_length", type=int, default=10, help="Number of pilot sequences.") #Also the number of pilot = tau_p
    parser.add_argument('-length', "--area_length", type=int, default=1, help="Area length, from -1 to 1.")
    parser.add_argument('-f', "--frequency", type=int, default=1900, help="Frequency 1900 MHz.")
    parser.add_argument('-B', "--bandwidth", type=int, default=20e6, help="Frequency 20 MHz.")
    parser.add_argument('-h_ap', "--height_AP", type=float, default=15, help="Height of Access Point (in km).")
    parser.add_argument('-h_ue', "--height_UE", type=float, default=1.7, help="Height of User Equipment (in km).")
    parser.add_argument('-p', "--power_pilot", type=float, default=1, help="Power of pilot signal (in mW)")
    parser.add_argument('-P', "--power_AP", type=float, default=100, help="Transmission power of each AP")
    parser.add_argument('-d1', "--distance1", type=float, default=0.05, help="Distance range number 1")
    parser.add_argument('-d0', "--distance0", type=float, default=0.01, help="Distance range number 0")

    # parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-num_train', "--training_sample", type=int, default=5120, help="Number of training samples.")
    parser.add_argument('-num_valid', "--validating_sample", type=int, default=0, help="Number of validating samples.")
    parser.add_argument('-num_test', "--testing_sample", type=int, default=1280, help="Number of testing samples.")
    parser.add_argument('-batch', "--batch_size", type=int, default=64, help="Size of batch data")

    args = parser.parse_args()
    system = SysMod(args)
    '''
    Generate dataset as csv file, so next time I can easily use it
    The structure of data set is like
    
    -------------------------------------------------------------------------------------------
    | sum_rate | min_rate | beta with size [num_ap]*[num_ue] | pilot_index with size [num_ue] |
    -------------------------------------------------------------------------------------------
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    |          |          |                                  |                                |
    -------------------------------------------------------------------------------------------
    '''
    generate_dataset(args, system)
    end_time = time.time()
    running_time = end_time - start_time
    print(f'Done generating data after{running_time: .6f} seconds')














    # ------------------------------------------------------------------------------------ #
    '''
    total_sample = args.training_sample + args.validating_sample + args.testing_sample
    data = []
    for i in range(total_sample):
        data.append(system.data_sample_generator())
    data = np.array(data)
    #--------------------------------------------------------------------#
    training_data = data[:args.training_sample,:]
    validating_data = data[args.training_sample:args.training_sample + args.validating_sample,:]
    testing_data = data[args.training_sample + args.validating_sample:, :]
    # --------------------------------------------------------------------#
    sum_dl_rate_train = training_data[:,0]
    mean_dl_rate_train = np.mean(sum_dl_rate_train)
    x_train = training_data[:,1: 1+args.number_AP*args.number_UE]
    y_train = training_data[:, 1+args.number_AP*args.number_UE:].astype(np.int64)
    # --------------------------------------------------------------------#
    sum_dl_rate_valid = validating_data[:, 0]
    x_valid = validating_data[:,1 : 1+args.number_AP * args.number_UE]
    y_valid = validating_data[:, 1+args.number_AP * args.number_UE:].astype(np.int64)
    # --------------------------------------------------------------------#
    sum_dl_rate_test = testing_data[:, 0]
    x_test = testing_data[:,1 : 1+args.number_AP * args.number_UE]
    y_test = testing_data[:, 1+args.number_AP * args.number_UE:].astype(np.int64)
    # --------------------------------------------------------------------#
    X_train = torch.tensor(x_train).float()
    Y_train = torch.unsqueeze(torch.tensor(y_train), 2)
    Y_train_hot = torch.scatter(torch.zeros((len(x_train), args.number_UE,args.pilot_length)), 2, Y_train, 1)
    # --------------------------------------------------------------------#
    X_valid = torch.tensor(x_valid).float()
    Y_valid = torch.unsqueeze(torch.tensor(y_valid), 2)
    Y_valid_hot = torch.scatter(torch.zeros((len(x_valid), args.number_UE, args.pilot_length)), 2, Y_valid, 1)
    # --------------------------------------------------------------------#
    X_test = torch.tensor(x_test).float()
    Y_test = torch.unsqueeze(torch.tensor(y_test), 2)
    Y_test_hot = torch.scatter(torch.zeros((len(x_test), args.number_UE, args.pilot_length)), 2, Y_test, 1)


    train_data_loader = torch.utils.data.DataLoader(list(zip(X_train, Y_train_hot)), batch_size = args.batch_size, shuffle = True, drop_last = True)
    val_data_loader = torch.utils.data.DataLoader(list(zip(X_valid, Y_valid_hot)), batch_size = args.batch_size, shuffle = True, drop_last = True)
    test_data_loader = torch.utils.data.DataLoader(list(zip(X_test, Y_test_hot)), batch_size = args.batch_size, shuffle = True, drop_last = True)

    import pennylane as qml
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


    n_qubits = 8
    dev = qml.device("default.qubit", wires=n_qubits)
    Pooling_out = [1, 3, 5, 7]


    @qml.qnode(dev)
    def qnode(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7,
              weights_8):  # , weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, weights_16, weights_17
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

    clayer_1 = torch.nn.Linear(args.number_AP*args.number_UE, 8)
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    clayer_2 = torch.nn.Linear(4, args.number_UE*args.pilot_length)
    # clayer_3 = torch.nn.Linear(128, 2)
    layers = [clayer_1, qlayer, clayer_2]  # clayer_1,
    model = torch.nn.Sequential(*layers)  # .to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    summary(model, (args.number_AP*args.number_UE,))

    epochs = 40
    count = 0
    dl_rate_train = np.zeros(epochs) #of [training_sample] samples --> need to mean
    for epoch in range(epochs):
        running_loss = 0
        running_loss_val = 0
        model.train()
        rate_sum_epoch = 0
        for i, (xs, ys) in enumerate(train_data_loader):
            opt.zero_grad()
            output_QCNN_b4_softmax = model(xs).reshape(args.batch_size, args.number_UE,args.pilot_length)
            output_QCNN = F.softmax(output_QCNN_b4_softmax, dim=-1) #(batch_size, num_ue, tau_p)
            rate_sum_epoch += system.sum_dl_rate_calculator(xs, output_QCNN, i)
            loss_evaluated = loss(output_QCNN, ys)
            loss_evaluated.backward()
            opt.step()
            running_loss += loss_evaluated.item()
        dl_rate_train[epoch] = (rate_sum_epoch)/args.training_sample
        avg_loss = running_loss / args.batch_size
        print(f"[{epoch}]/[{epochs}] Loss_evaluated: {avg_loss}")
        print(f"[{epoch}]/[{epochs}] Average rate of 300 samples in training process: {dl_rate_train[epoch]}, compared to mean DL rate by greedy: {mean_dl_rate_train}")

        model.eval()
        with torch.no_grad():
            # y_pred_tr = model(X_train)
            # predictions_tr = torch.argmax(y_pred_tr, axis=1).detach().numpy()
            # correct_tr = [1 if p == p_true else 0 for p, p_true in zip(predictions_tr, Y_train)]
            # accuracy_tr = sum(correct_tr) / len(correct_tr)
            # print("training_Accuracy : ", accuracy_tr)
            for xt, yt in test_data_loader:
                output_QCNN_test_b4_softmax = model(xt).reshape(args.batch_size, args.number_UE,args.pilot_length)
                output_QCNN_test = F.softmax(output_QCNN_test_b4_softmax, dim=-1)


                loss_evaluated_val = loss(output_QCNN_test, yt)
                running_loss_val += loss_evaluated_val.item()
            avg_loss_val = running_loss_val / args.batch_size
            print(f"[{epoch}]/[{epochs}] Loss_evaluated_testing: {avg_loss_val}")

        #     y_pred_val = model(X_val)
        #     predictions_val = torch.argmax(y_pred_val, axis=1).detach().numpy()
        #     correct_val = [1 if p == p_true else 0 for p, p_true in zip(predictions_val, Y_val)]
        #     accuracy_val = sum(correct_val) / len(correct_val)
        #     print("Validation_Accuracy : ", accuracy_val)
        # f = open("test10.txt", 'a')
        # f.write("Epoch : %f\n" % epoch)
        # f.write("Avg_loss_traing: %f\n" % avg_loss)
        # f.write("Acc_training: %f\n" % accuracy_tr)
        # f.write("Avg_loss_validation: %f\n" % avg_loss_val)
        # f.write("Acc_validation: %f \n" % accuracy_val)
        # f.close()
        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))
    print(dl_rate_train)
    # model.eval()
    # with torch.no_grad():
    #     y_pred = model(X_test)
    #     predictions = torch.argmax(y_pred, axis=1).detach().numpy()
    # correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, Y_test)]
    # accuracy = sum(correct) / len(correct)
    # print(f"Accuracy: {accuracy * 100}%")
    
    
    '''

import numpy as np
from scipy.linalg import cholesky, eigh
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

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
rho_d = P_d / noise_power
rho_u = rho_p = p_u / noise_power

sigma_shd = 8  # in dB
D_cor = 0.1
tau_c = 200


# per-user net throughputs S_cf_Ak = spectral_bandwidth*((1-tau_p/tau_c)/2)*R_cf_A,k and A \in {d,u}
def position_generator(num_ue, num_ap):
    AP_position = np.random.uniform(-D, D, size=(num_ap, 2))
    UE_position = np.random.uniform(-D, D, size=(num_ue, 2))
    return AP_position, UE_position


def LSF_calculator(AP_position, UE_position):
    # AP_position: shape (9,M,2)
    AP_expanded = AP_position[:, np.newaxis, :]  # Shape: (num_ap, 1, 2)
    UE_expanded = UE_position[np.newaxis, :, :]

    distanceUE2AP = np.sqrt(np.sum((AP_expanded - UE_expanded) ** 2, axis=2))
    pathloss = np.zeros_like(distanceUE2AP)
    pathloss[(distanceUE2AP < d0)] = -L - 15 * np.log10(d1) - 20 * np.log10(d0)
    pathloss[(distanceUE2AP >= d0) & (distanceUE2AP <= d1)] = -L - 15 * np.log10(d1) - 20 * np.log10(
        distanceUE2AP[(distanceUE2AP >= d0) & (distanceUE2AP <= d1)])
    pathloss[(distanceUE2AP > d1)] = -L - 35 * np.log10(distanceUE2AP[(distanceUE2AP > d1)]) #+ np.random.normal(0,1) * 8
    beta = 10 ** (pathloss / 10)
    return beta


def c_calculator(pilot_index, beta, tau_p):
    num_ap, num_ue = beta.shape
    c_mk = np.zeros_like(beta)
    for k in range(num_ue):
        mask = (pilot_index == pilot_index[k])

        sum_val = np.sum(beta[:, mask], axis=1)  # Shape (num_ap,)

        numerator = np.sqrt(rho_p * tau_p) * beta[:, k]
        denominator = tau_p * rho_p * sum_val + 1

        c_mk[:, k] = numerator / denominator
    return c_mk


def sinr_calculator(pilot_index, beta, gamma):
    num_ap, num_ue = beta.shape
    etaa = 1 / (np.sum(gamma, axis=1))
    eta = np.tile(etaa[:, np.newaxis], (1, num_ue))
    DS = rho_d * (np.sum(np.sqrt(eta) * gamma, axis=0)) ** 2  # ( K,)
    # BU
    eta_gamma = eta * gamma
    product = np.sum(eta_gamma[:, :, np.newaxis] * beta[:, np.newaxis, :], axis=0)
    BU = np.sum(product, axis=0)
    # UI
    UI = np.zeros(num_ue)
    flag = (pilot_index[:, np.newaxis] == pilot_index).astype(float)
    for k in range(num_ue):
        mask = np.arange(num_ue) != k
        sum_squared = np.sum(((np.sum(
            np.sqrt(eta[:, mask]) * gamma[:, mask] * (beta[:, k][:, np.newaxis] / beta[:, mask]), axis=0)) ** 2) * flag[
                                 k, mask])
        UI[k] = rho_d * sum_squared
    return (DS / (UI + BU + 1))


def dl_rate_calculator(pilot_index, beta, tau_p):
    c_mk = c_calculator(pilot_index, beta, tau_p)
    gamma = np.sqrt(tau_p * rho_p) * beta * c_mk
    sinr = sinr_calculator(pilot_index, beta, gamma)
    return np.log2(1 + sinr)


def greedy_assignment(beta, tau_p, N=20):
    num_ap, num_ue = beta.shape
    pilot_index = np.random.randint(tau_p, size=num_ue)
    # pilot_init = pilot_index.copy() #Use to compare with random assignment
    for n in range(N):
        dl_rate = dl_rate_calculator(pilot_index, beta, tau_p)
        k_star = np.argmin(dl_rate)
        sum_beta = np.zeros(tau_p)
        for tau in range(tau_p):  # too much for loops, optimize latter!!
            for m in range(num_ap):
                for k in range(num_ue):
                    if (k != k_star) and (pilot_index[k] == tau):
                        sum_beta[tau] += beta[m, k]
        pilot_index[k_star] = np.argmin(sum_beta)

    rate_list = dl_rate_calculator(pilot_index, beta, tau_p)
    sum_rate = np.sum(rate_list)
    min_rate = np.min(rate_list)
    max_rate = np.max(rate_list)
    return pilot_index, sum_rate, min_rate, max_rate


def generate_datasample(num_ue, num_ap, tau_p):
    # generate randomly location of M APs, AP_group
    AP_position, UE_position = position_generator(num_ue, num_ap)
    # sh_AP = SSF_calculator(AP_position)
    # sh_UE = SSF_calculator(UE_position)
    # z_shd = sh_AP[:, np.newaxis] + sh_UE[np.newaxis, :]
    beta = LSF_calculator(AP_position, UE_position)  # uncorrelated shadow fading
    # print(f'LSF: \n{beta}')
    pilot_index, sum_rate, min_rate, max_rate = greedy_assignment(beta, tau_p, N=20)
    beta_flatten = beta.flatten()
    pilot_index_flatten = pilot_index.flatten()
    sample = np.concatenate(
        (
            np.array([sum_rate]),
            np.array([min_rate]),
            np.array([max_rate]),
            beta_flatten,
            pilot_index_flatten
        )
    )
    return sample, beta


def generate_dataset_new(num_ue, num_ap, tau_p):
    dataset = []
    for i in range(50):
        dataset.append(generate_datasample(num_ue, num_ap, tau_p))
        if not (i % 10):
            print(f'Generating data sample [{i}]/[{50}]')

    dataset = np.array(dataset)
    df = pd.DataFrame(dataset,
                      columns=['sum_rate', 'min_rate', 'max_rate'] + [f'beta_{i}' for i in range(num_ap * num_ue)] + [
                          f'tau_{i}' for
                          i in range(
                              num_ue)])
    filename = f"{num_ue}UE_{num_ap}AP_{tau_p}pilot.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def generate_data_small_scenario(args):
    num_ue = 5
    num_ap = 10
    total_sample = args.training_sample + args.testing_sample
    data = []
    for sample in range(total_sample):
        AP_position, UE_position = position_generator(num_ue, num_ap)
        beta = LSF_calculator(AP_position, UE_position)
        beta_flatten = beta.flatten()
        data[sample].append(beta_flatten)

    data = np.array(data)
    train_data = data[:args.training_samples, :]
    test_data = data[args.training_samples:, :]
    X_train = torch.tensor(train_data, dtype=torch.float64)
    X_test = torch.tensor(test_data, dtype=torch.float64)
    train_dataset = list(X_train)
    test_dataset = list(X_test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader

import numpy as np
from scipy.linalg import cholesky, eigh
import pandas as pd




D = 1 # in kilometer
d1 = 0.05 # in kilometer
d0 = 0.01 # in kilometer
h_ap = 15 # in meter
h_ue = 1.7 # in meter
B = 20 # in MHz
f = 1900 # in MHz
L = 46.3 + 33.9*np.log10(f) - 13.82*np.log10(h_ap) - (1.1*np.log10(f)-0.7)*h_ue + (1.56*np.log10(f)-0.8)
P_d = 0.2 # downlink power: 200 mW
p_u = 0.1 # uplink power: 100 mW
p_p = 0.1 # pilot power: 100mW
noise_figure = 9 #dB
T = 290 #noise temperature in Kelvin
noise_power = (B*10**6) * (1.381*10**(-23)) * T * (10**(noise_figure/10)) # Thermal noise in W
rho_d = P_d/noise_power
rho_u = rho_p = p_u/noise_power

sigma_shd = 8 # in dB
D_cor = 0.1
tau_c = 200

# per-user net throughputs S_cf_Ak = spectral_bandwidth*((1-tau_p/tau_c)/2)*R_cf_A,k and A \in {d,u}
def position_generator(num_ue, num_ap):
    # AP_position = np.zeros((9,num_ap,2))
    AP_position= np.random.uniform(-D, D, size = (num_ap,2))
    UE_position = np.random.uniform(-D, D, size=(num_ue, 2))
    # for i in range(1, 9):
    #     shift = np.array([0, 0])
    #     if i in [1, 5, 6]:
    #         shift[0] += D
    #     if i in [2, 5, 7]:
    #         shift[1] += D
    #     if i in [3, 7, 8]:
    #         shift[0] -= D
    #     if i in [4, 6, 8]:
    #         shift[1] -= D
    #     AP_position[i, :, :] = AP_position[0, :, :] + shift

    # UE_position = np.zeros((9, num_ue, 2))
    # for i in range(1, 9):
    #     shift = np.array([0, 0])
    #     if i in [1, 5, 6]:
    #         shift[0] += D
    #     if i in [2, 5, 7]:
    #         shift[1] += D
    #     if i in [3, 7, 8]:
    #         shift[0] -= D
    #     if i in [4, 6, 8]:
    #         shift[1] -= D
    #     UE_position[i, :, :] = UE_position[0, :, :] + shift
    return AP_position, UE_position

# def make_positive_definite(matrix, regularization = 1e-6):
#     """ Modify matrix to make it positive definite by adding epsilon to the diagonal """
#     eigenvalues, _ = eigh(matrix)
#     if np.all(eigenvalues > 0):
#         return matrix
#     else:
#         # Add regularization term
#         matrix += np.eye(matrix.shape[0]) * regularization
#         return matrix

# def SSF_calculator(position_array):
#     size = len(position_array[0])
#     Dist = np.zeros((size,size))
#     for i in range(size):
#         for j in range(size):
#             Dist[i,j] = min([np.linalg.norm(position_array[0, i, :] - position_array[l, j, :]) for l in range(9)])
#
#     Cor = np.exp(-np.log(2)*Dist/D_cor)

    # # Make sure Cor always positive definite
    # Cor = (Cor + Cor.T)/2
    # Cor = make_positive_definite(Cor)
    #
    #
    # A1 = cholesky(Cor, lower = True)
    # x1 = np.random.rand(size)
    # sh = np.dot(A1,x1)
    # for p in range(size):
    #     sh[p] = (1 / np.sqrt(2)) * sigma_shd * sh[p] / np.linalg.norm(A1[p, :])
    #
    # return sh

def LSF_calculator(AP_position, UE_position):
    # AP_position: shape (9,M,2)
    AP_expanded = AP_position[:, np.newaxis, :]  # Shape: (num_ap, 1, 2)
    UE_expanded = UE_position[np.newaxis, :, :]
    # UE_position_expanded = UE_position[np.newaxis, :, :] # shape (1,K,2)
    # dist_diff = AP_position[:, :, np.newaxis, :] - UE_position_expanded # shape (9,M,K,2)
    distanceUE2AP = np.sqrt(np.sum((AP_expanded - UE_expanded) ** 2, axis=2))
    # distanceUE2AP = np.min(np.sqrt(np.sum((dist_diff) ** 2, axis=-1)), axis =0) # shape (M,K)
    # print(f'------------DISTANCE------------\t {np.max(distanceUE2AP)} \t {np.min(distanceUE2AP)}')
    # distanceUE2AP = np.min(distance_total, axis=0)
    pathloss = np.zeros_like(distanceUE2AP)
    pathloss[(distanceUE2AP < d0)] = -L - 15 * np.log10(d1) - 20 * np.log10(d0)
    pathloss[(distanceUE2AP >= d0) & (distanceUE2AP <= d1)] = -L - 15 * np.log10(d1) - 20 * np.log10(
        distanceUE2AP[(distanceUE2AP >= d0) & (distanceUE2AP <= d1)])
    pathloss[(distanceUE2AP > d1)] = -L - 35 * np.log10(distanceUE2AP[(distanceUE2AP > d1)]) + np.random.normal(0, 1) * 8
    beta = 10**(pathloss/10)
    return beta

def c_calculator(pilot_index, beta, tau_p):
    num_ap, num_ue = beta.shape
    c_mk = np.zeros_like(beta)
    for k in range(num_ue):
        # Mask for indices with the same pilot index as k
        mask = (pilot_index == pilot_index[k])

        # Sum beta values where mask is True
        sum_val = np.sum(beta[:, mask], axis=1)  # Shape (num_ap,)

        # Compute numerator and denominator for c_mk
        numerator = np.sqrt(rho_p * tau_p) * beta[:, k]
        denominator = tau_p * rho_p * sum_val + 1

        # Assign values to c_mk
        c_mk[:, k] = numerator / denominator
    return c_mk

def sinr_calculator(pilot_index, beta, gamma):
    num_ap, num_ue = beta.shape
    etaa = 1/(np.sum(gamma, axis=1))
    eta = np.tile(etaa[:, np.newaxis], (1, num_ue))
    # sinr = np.zeros_like(pilot_index)
    DS = rho_d * (np.sum(np.sqrt(eta[:, np.newaxis]) * gamma, axis=0)) ** 2 #( K,)

    #BU
    eta_gamma = eta * gamma
    product = np.sum(eta_gamma[:, :, np.newaxis] * beta[:, np.newaxis, :], axis=0)
    BU = np.sum(product, axis=0)


    # UI
    UI = np.zeros(num_ue)
    flag = (pilot_index[:, np.newaxis] == pilot_index).astype(float)
    for k in range(num_ue):
        # Create a mask for valid j values where j != k
        mask = np.arange(num_ue) != k
        sum_squared = np.sum(
            ((np.sum(np.sqrt(eta[:, np.newaxis]) * gamma[:, mask] * (beta[:, k][:, np.newaxis] / beta[:, mask]), axis=0)) ** 2) *
            flag[k, mask]
        )
        UI[k] = rho_d*sum_squared

        # sum_inner = np.sum(eta[:, np.newaxis] * gamma * beta[:, k][:, np.newaxis], axis=0)
        # BU[k] = rho_d*np.sum(sum_inner)

    return (DS / (UI+BU+1))
def dl_rate_calculator(pilot_index, beta, tau_p):
    c_mk = c_calculator(pilot_index, beta, tau_p)
    gamma = np.sqrt(tau_p*rho_p)*beta*c_mk
    sinr = sinr_calculator(pilot_index, beta, gamma)
    return np.log2(1+sinr)

def greedy_assignment(beta, tau_p, N = 20):
    num_ap, num_ue = beta.shape
    pilot_index = np.random.randint(tau_p, size=num_ue)
    # pilot_init = pilot_index.copy() #Use to compare with random assignment
    for n in range(N):
        dl_rate = dl_rate_calculator(pilot_index, beta,tau_p)
        k_star = np.argmin(dl_rate)
        sum_beta = np.zeros(tau_p)
        for tau in range(tau_p): #too much for loops, optimize latter!!
            # mask = (pilot_index != k_star) & (pilot_index == tau)
            # sum_beta[tau] = np.sum(beta[:, mask])
            for m in range(num_ap):
                for k in range(num_ue):
                    if (k!= k_star) and (pilot_index[k]==tau):
                        sum_beta[tau] += beta[m,k]
        pilot_index[k_star] = np.argmin(sum_beta)

    rate_list = dl_rate_calculator(pilot_index, beta, tau_p)
    sum_rate = np.sum(rate_list)
    min_rate = np.min(rate_list)
    return pilot_index, sum_rate, min_rate


def generate_datasample(num_ue, num_ap, tau_p):
    '''
    generate randomly location of AP
    AP_group[0] in the center, around [0,0]
    8 groups around AP_group[0] --
    '''
    #generate randomly location of M APs, AP_group
    AP_position, UE_position = position_generator(num_ue, num_ap)
    # sh_AP = SSF_calculator(AP_position)
    # sh_UE = SSF_calculator(UE_position)
    # z_shd = sh_AP[:, np.newaxis] + sh_UE[np.newaxis, :]
    beta = LSF_calculator(AP_position, UE_position) #uncorrelated shadow fading
    pilot_index, sum_rate, min_rate = greedy_assignment(beta, tau_p, N = 20)

    beta_flatten = beta.flatten()
    pilot_index_flatten = pilot_index.flatten()
    sample = np.concatenate(
        (
            np.array([sum_rate]),
            np.array([min_rate]),
            beta_flatten,
            pilot_index_flatten
        )
    )
    return sample

def  generate_dataset_new(num_ue, num_ap,tau_p):
    dataset = []
    for i in range(1280):
        dataset.append(generate_datasample(num_ue, num_ap, tau_p))
        if not (i%10):
            print(f'Generating data sample [{i}]/[{1280}]')

    dataset = np.array(dataset)
    # print(f'Sum_rate first: {dataset[0:10, 0]}')
    df = pd.DataFrame(dataset,
                      columns=['sum_rate', 'min_rate'] + [f'beta_{i}' for i in range(num_ap * num_ue)] + [f'tau_{i}' for  i in range(num_ue)])
    if (num_ue == 20) and (num_ap == 40) and (tau_p == 5):
        # case1: 20 UEs - 40 APs - 05 pilots
        df.to_csv('20UE_40AP_05pilot.csv', index = False)
    elif (num_ue == 20) and (num_ap == 40) and (tau_p == 10):
        # case2: 20 UEs - 40 APs - 10 pilots
        df.to_csv('20UE_40AP_10pilot.csv', index = False)
    elif (num_ue == 20) and (num_ap == 40) and (tau_p == 15):
        # case3: 20 UEs - 40 APs - 15 pilots
        df.to_csv('20UE_40AP_15pilot.csv', index = False)
    elif (num_ue == 20) and (num_ap == 60) and (tau_p == 5):
        # case4: 20 UEs - 60 APs - 5 pilots
        df.to_csv('20UE_60AP_05pilot.csv', index = False)
    elif (num_ue == 20) and (num_ap == 60) and (tau_p == 10):
        # case5: 20 UEs - 60 APs - 10 pilots
        df.to_csv('20UE_60AP_10pilot.csv', index = False)
    elif (num_ue == 20) and (num_ap == 60) and (tau_p == 15):
        # case6: 20 UEs - 60 APs - 15 pilots
        df.to_csv('20UE_60AP_15pilot.csv', index = False)
    elif (num_ue == 30) and (num_ap == 40) and (tau_p == 5):
        # case7: 30 UEs - 40 APs - 5 pilots
        df.to_csv('30UE_40AP_05pilot.csv', index = False)
    elif (num_ue == 30) and (num_ap == 40) and (tau_p == 10):
        # case8: 30 UEs - 40 APs - 10 pilots
        df.to_csv('30UE_40AP_10pilot.csv', index = False)
    elif (num_ue == 30) and (num_ap == 40) and (tau_p == 15):
        # case9: 30 UEs - 40 APs - 15 pilots
        df.to_csv('30UE_40AP_15pilot.csv', index = False)
    elif (num_ue == 30) and (num_ap == 40) and (tau_p == 20):
        # case10: 30 UEs - 40 APS - 20 pilots
        df.to_csv('30UE_40AP_20pilot.csv', index = False)
    elif (num_ue == 30) and (num_ap == 60) and (tau_p == 5):
        # case11: 30 UEs - 60 APs - 5 pilots
        df.to_csv('30UE_60AP_05pilot.csv', index = False)
    elif (num_ue == 30) and (num_ap == 60) and (tau_p == 10):
        # case12: 30 UEs - 60 APs - 10 pilots
        df.to_csv('30UE_60AP_10pilot.csv', index = False)
    elif (num_ue == 30) and (num_ap == 60) and (tau_p == 15):
        # case13: 30 UEs - 60 APs - 15 pilots
        df.to_csv('30UE_60AP_15pilot.csv', index = False)
    elif (num_ue == 30) and (num_ap == 60) and (tau_p== 20):
        # case14: 30 UEs - 60 APs - 20 pilots
        df.to_csv('30UE_60AP_20pilot.csv', index = False)
    else:
        print("No accurate choice")




import torch
import numpy as np
import torch.nn.functional as F


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

# def c_calculator_torch(pilot_index, beta, tau_p):
#     num_ap, num_ue = beta.shape
#
#     c_mk = torch.zeros_like(beta)
#     for k in range(num_ue):
#         mask = (pilot_index == pilot_index[k])
#         sum_val = torch.sum(beta[:, mask], dim=1)  # Shape (num_ap,)
#         numerator = torch.sqrt(torch.tensor(rho_p * tau_p, dtype=torch.float32)) * beta[:, k]
#         denominator = tau_p * rho_p * sum_val + 1
#         c_mk[:, k] = numerator / denominator
#     return c_mk

# def sinr_calculator_torch(pilot_index, beta, gamma):
#     num_ap, num_ue = beta.shape
#     etaa = 1 / torch.sum(gamma, dim=1)
#     eta = etaa[:, None].expand(-1, num_ue)
#     DS = rho_d * (torch.sum(torch.sqrt(eta) * gamma, dim=0)) ** 2  # Shape (num_ue,)
#     eta_gamma = eta * gamma
#     product = torch.sum(eta_gamma[:, :, None] * beta[:, None, :], dim=0)
#     BU = torch.sum(product, dim=0)
#
#
#     UI = torch.zeros(num_ue, device=beta.device)
#     flag = (pilot_index[:, None] == pilot_index).float()
#     for k in range(num_ue):
#         mask = torch.arange(num_ue, device=beta.device) != k
#         sum_squared = torch.sum(
#             ((torch.sum(
#                 torch.sqrt(eta[:, mask]) * gamma[:, mask] * (beta[:, k][:, None] / beta[:, mask]), dim=0
#             ) ** 2) * flag[k, mask])
#         )
#         UI[k] = rho_d * sum_squared
#     return DS / (UI + BU + 1)

def dl_rate_calculator_w_pilot_index(pilot_index_torch, beta_torch, tau_p):
    num_ap, num_ue = beta_torch.shape
    #c_mk
    c_mk = torch.zeros_like(beta_torch)
    for k in range(num_ue):
        mask = (pilot_index_torch == pilot_index_torch[k])
        sum_val = torch.sum(beta_torch[:, mask], dim=1)  # Shape (num_ap,)
        numerator = torch.sqrt(torch.tensor(rho_p * tau_p, dtype=torch.float32)) * beta_torch[:, k]
        denominator = tau_p * rho_p * sum_val + 1
        c_mk[:, k] = numerator / denominator
    #gamma
    gamma = torch.sqrt(torch.tensor(rho_p * tau_p, dtype=torch.float32)) * beta_torch * c_mk
    #eta
    etaa = 1 / torch.sum(gamma, dim=1)
    eta = etaa[:, None].expand(-1, num_ue)
    #sinr
    DS = rho_d * (torch.sum(torch.sqrt(eta) * gamma, dim=0)) ** 2  # Shape (num_ue,)
    eta_gamma = eta * gamma
    product = torch.sum(eta_gamma[:, :, None] * beta_torch[:, None, :], dim=0)
    BU = torch.sum(product, dim=0)
    UI = torch.zeros(num_ue, device=beta_torch.device)
    flag = (pilot_index_torch[:, None] == pilot_index_torch).float()
    for k in range(num_ue):
        mask = torch.arange(num_ue, device=beta_torch.device) != k
        sum_squared = torch.sum(
            ((torch.sum(
                torch.sqrt(eta[:, mask]) * gamma[:, mask] * (beta_torch[:, k][:, None] / beta_torch[:, mask]), dim=0
            ) ** 2) * flag[k, mask])
        )
        UI[k] = rho_d * sum_squared
    sinr = DS / (UI + BU + 1)
    return torch.log2(1+sinr)

def dl_rate_calculator_w_pilot_probs(pilot_probs, beta, tau_p, epoch, batch_no,sample):
    num_ap, num_ue = beta.shape
    phi_inner_product = torch.matmul(pilot_probs, pilot_probs.T)
    phi_inner_product_squared = phi_inner_product ** 2
    inner_sum = torch.matmul(beta, phi_inner_product_squared)
    numerator_c = torch.sqrt(torch.tensor(rho_p * tau_p, dtype=torch.float32)) * beta  # Shape: (M, K)
    denominator_c = tau_p * rho_p * inner_sum + 1
    c_mk = numerator_c / denominator_c  # Shape: (M, K)
    gamma = torch.sqrt(torch.tensor(tau_p * rho_p, dtype=torch.float32)) * beta * c_mk
    # etaa = 1 / torch.sum(gamma, dim=1)
    etaa = torch.ones(num_ap)
    eta = etaa[:, None].expand(-1, num_ue)
    DS = rho_d * (torch.sum(torch.sqrt(eta) * gamma, dim=0)) ** 2  # Shape (num_ue,)
    eta_gamma = eta * gamma
    product = torch.sum(eta_gamma[:, :, None] * beta[:, None, :], dim=0)
    BU = torch.sum(product, dim=0)
    UI = torch.zeros(num_ue, device=beta.device)
    # flag = (pilot_index_torch[:, None] == pilot_index_torch).float()
    flag = torch.matmul(pilot_probs, pilot_probs.T)
    for k in range(num_ue):
        #
        # mask = torch.arange(num_ue, device=beta.device) != k
        # sum_squared = torch.sum(
        #     ((torch.sum(
        #         torch.sqrt(eta[:, mask]) * gamma[:, mask] * (beta[:, k][:, None] / beta[:, mask]), dim=0
        #     ) ** 2) * flag[k, mask])
        # )
        sum_term_1 = 0
        for j in range(num_ue):
            sum_term_2 = 0
            if j!=k:
                for m in range(num_ap):
                    # sum_term_2 += (torch.sqrt(eta[m,j])*gamma[m,j]*beta[m,k])/(beta[m,j])
                    sum_term_2 += (torch.sqrt(eta[m, j]) * gamma[m, j] * beta[m, k])

                    print(f'beta[m,j]: {sum_term_2}')
            sum_term_1 += (sum_term_2**2)*(flag[k,j]**2)
            print(f'sum_term_2: {sum_term_2}')
        UI[k] = rho_d * sum_term_1

        # print(f'sum_squared: {sum_term_1}')
    sinr = DS / (UI + BU + 1)
    print(f'DL_RATE_CALCULATOR: epoch:{epoch} --- batch_no: {batch_no} --- sample:{sample}')
    # print(f'phi_inner_product[0]: \n{phi_inner_product[0]}')
    # print(f'inner_sum[0]: \n{inner_sum[0]}')
    # print(f'numerator_c[0]: \n{numerator_c[0]}')
    # print(f'denominator_c[0]: \n{denominator_c[0]}')
    # print(f'c_mk[0]: \n{c_mk[0]}')
    # print(f'gamma[0]: \n{gamma[0]}')
    # print(f'eta[0]: \n{eta[0]}')
    # print(f'DS[0]: {DS}')
    # print(f'BU[0]: {BU}')

    print(f'UI[0]: {UI}')
    print(f'sinr[0]: {sinr}')
    return torch.log2(1 + sinr)

def softargmax(x, beta=1e10):
    # Ensure x is a PyTorch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # Create the range of indices for the last dimension
    x_range = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device)

    # Apply softmax with scaling factor beta
    softmax_values = F.softmax(x * beta, dim=-1)

    # Compute the weighted sum (softargmax) along the last dimension
    return torch.sum(softmax_values * x_range, dim=-1)

def loss_function_PA(args, beta, output, epoch, batch_no):
    sum_rate_each_batch = torch.zeros(args.batch_size)
    beta = beta.reshape(args.batch_size, args.number_AP, args.number_UE)
    pilot_probs = output.reshape(args.batch_size, args.number_UE, args.pilot_length)
    print(f'----->  OUTPUT: {pilot_probs[0]}')
    pilot_probs = F.softmax(pilot_probs, dim=-1)
    print(f'----->  BETA: {beta[0]}')
    print(f'----->  PILOT PROBABILITY: {pilot_probs[0]}')

    # pilot_index = torch.argmax(pilot_probs, dim=-1) # loss grad_fn from here
    # pilot_index = softargmax(pilot_probs)
    #Try to calculate rate by pilot_probs, seeing pilot_probs as pilot_signals
    for i in range(args.batch_size):
        sum_rate_each_batch[i] = torch.sum(dl_rate_calculator_w_pilot_probs(pilot_probs[i], beta[i], args.pilot_length, epoch, batch_no, i))

    print(f'-----> SUM_RAT_EACH_BATCH[0]: {sum_rate_each_batch}')
    #calculate with pilot_index
    # for i in range(args.batch_size):
    #     sum_rate_each_batch[i] = torch.sum(dl_rate_calculator_w_pilot_index(pilot_index[i], beta[i], args.pilot_length))
    print(f'')
    print(f'-----> TORCH MEAN: {torch.mean(sum_rate_each_batch)*(-1)}')
    return torch.mean(sum_rate_each_batch)*(-1)

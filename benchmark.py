"""
    Benchmark to calculate total data rate of system through several benchmark
"""
import numpy as np
from SystemModel import SysMod

sysmod = SysMod
def random_assignment(tau_p, num_ue):
    return np.random.randint(tau_p-1, size = num_ue)

def master_AP_assignment(tau_p, num_ue, beta):
    pilot_index = -1*np.ones(num_ue)
    pilot_index[0:tau_p] = np.random.permutation(tau_p)
    beta_transpose = np.transpose(beta)
    for k in range(tau_p, num_ue):
        m_star = np.argmax(beta_transpose[k]) #master AP
        interference = np.zeros(tau_p)
        for tau in range(tau_p):
            interference[tau] = np.sum(beta_transpose[pilot_index == tau, m_star])
        pilot_index[k] = np.argmin(interference)
    return pilot_index

def greedy_assignment(tau_p, num_ue, num_ap, beta, pilot_index_init, N):
    pilot_index = pilot_index_init
    for n in range(N):
        dl_rate = sysmod.dl_rate_calculator(pilot_index)
        k_star = np.argmin(dl_rate)
        sum_beta = np.zeros(tau_p)
        for tau in range(tau_p):
            for m in range(num_ap):
                for k in range(num_ue):
                    if (k != k_star) and (pilot_index[k] == tau):
                        sum_beta[tau] += beta[m, k]
        pilot_index[k_star] = np.argmin(sum_beta)
    return pilot_index

def K_mean_located_assignment():
    return 1

def located_assignment():
    return 1

def
import time
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torchsummary import summary as summary
from SystemModel import SysMod
from model import *
from plot import *
from create_data import *



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-num_ap', "--number_AP", type=int, default=40, help="Number of Access Point.")
    parser.add_argument('-num_ue', "--number_UE", type=int, default=20, help="Number of User Equipment.")
    parser.add_argument('-tau_p', "--pilot_length", type=int, default=3, help="Number of pilot sequences.") #Also the number of pilot = tau_p
    parser.add_argument('-length', "--area_length", type=int, default=1000, help="Area length, from -1 to 1.")
    parser.add_argument('-f', "--frequency", type=int, default=1900, help="Frequency 1900 MHz.")
    parser.add_argument('-B', "--bandwidth", type=int, default=20e6, help="Frequency 20 MHz.")
    parser.add_argument('-h_ap', "--height_AP", type=float, default=15, help="Height of Access Point (in km).")
    parser.add_argument('-h_ue', "--height_UE", type=float, default=1.7, help="Height of User Equipment (in km).")
    parser.add_argument('-p', "--power_pilot", type=float, default=1, help="Power of pilot signal (in mW)")
    parser.add_argument('-P', "--power_AP", type=float, default=100, help="Transmission power of each AP")
    parser.add_argument('-d1', "--distance1", type=float, default=0.05, help="Distance range number 1")
    parser.add_argument('-d0', "--distance0", type=float, default=0.01, help="Distance range number 0")
    parser.add_argument('-num_train', "--training_sample", type=int, default=100, help="Number of training samples.")
    parser.add_argument('-num_valid', "--validating_sample", type=int, default=0, help="Number of validating samples.")
    parser.add_argument('-num_test', "--testing_sample", type=int, default=30, help="Number of testing samples.")
    parser.add_argument('-batch', "--batch_size", type=int, default=10, help="Size of batch data")
    parser.add_argument('-epc', "--epochs", type=int, default=5, help="Number of epoch")
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.0001, help="Learning rate of model")
    # parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])

    args = parser.parse_args()
    # system = SysMod(args)   # Change dl_rate_calculator for loss function
                            # Add function calculate channel-estimation error

    # Import data from .csv files
    # mean_rate, mean_min, mean_max, min_train, max_train, min_test, max_test, train_dataloader, test_dataloader = dataloader_creation(args)

    #add another metric that compute the channel-estimation error to show that pilot assignment help channel estimation process
    # dl_rate_train, dl_rate_test = model_training(args, train_dataloader, test_dataloader)

    # plot_dl_rate(dl_rate_train, dl_rate_test, mean_rate, mean_min, mean_max)

    train_dataloader, test_dataloader = generate_data_small_scenario(args)
    dl_rate_train, dl_rate_test = model_training(args, train_dataloader, test_dataloader)

    # generate_dataset_new(num_ue = 3, num_ap = 5, tau_p = 2)

    # D = 1  # in kilometer
    # d1 = 0.05  # in kilometer
    # d0 = 0.01  # in kilometer
    # h_ap = 15  # in meter
    # h_ue = 1.7  # in meter
    # B = 20  # in MHz
    # f = 1900  # in MHz
    # L = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_ap) - (1.1 * np.log10(f) - 0.7) * h_ue + (
    #             1.56 * np.log10(f) - 0.8)
    # P_d = 0.2  # downlink power: 200 mW
    # p_u = 0.1  # uplink power: 100 mW
    # p_p = 0.1  # pilot power: 100mW
    # noise_figure = 9  # dB
    # T = 290  # noise temperature in Kelvin
    # noise_power = (B * 10 ** 6) * (1.381 * 10 ** (-23)) * T * (10 ** (noise_figure / 10))  # Thermal noise in W
    # rho_d = P_d / noise_power
    # rho_u = rho_p = p_u / noise_power
    #
    # sigma_shd = 8  # in dB
    # D_cor = 0.1
    # tau_c = 200
    # sample, beta = generate_datasample(num_ue = 3, num_ap = 5, tau_p = 2)
    # print(f'rho_d: {rho_d}\t rho_p: {rho_p}')
    # beta_rho_d = beta*rho_d
    # beta_rho_p = beta * rho_p
    # print(f'beta_rho_d: \n{beta_rho_d}')
    # print(f'beta_rho_P: \n{beta_rho_p}')
    #
    # pilot_index, sum_rate, min_rate, max_rate = greedy_assignment(beta, tau_p = 2, N=20)
    # print(f'SUM_RATE: {sum_rate}\n MIN_RATE: {min_rate}\n MAX_RATE: {max_rate}\n')
    #
    # print(L)














    # ------------------------------------------------------------------------------------ #

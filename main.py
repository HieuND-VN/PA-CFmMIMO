import time
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torchsummary import summary as summary
from SystemModel import SysMod
from model import *
from plot import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-num_ap', "--number_AP", type=int, default=40, help="Number of Access Point.")
    parser.add_argument('-num_ue', "--number_UE", type=int, default=20, help="Number of User Equipment.")
    parser.add_argument('-tau_p', "--pilot_length", type=int, default=5, help="Number of pilot sequences.") #Also the number of pilot = tau_p
    parser.add_argument('-length', "--area_length", type=int, default=1000, help="Area length, from -1 to 1.")
    parser.add_argument('-f', "--frequency", type=int, default=1900, help="Frequency 1900 MHz.")
    parser.add_argument('-B', "--bandwidth", type=int, default=20e6, help="Frequency 20 MHz.")
    parser.add_argument('-h_ap', "--height_AP", type=float, default=15, help="Height of Access Point (in km).")
    parser.add_argument('-h_ue', "--height_UE", type=float, default=1.7, help="Height of User Equipment (in km).")
    parser.add_argument('-p', "--power_pilot", type=float, default=1, help="Power of pilot signal (in mW)")
    parser.add_argument('-P', "--power_AP", type=float, default=100, help="Transmission power of each AP")
    parser.add_argument('-d1', "--distance1", type=float, default=0.05, help="Distance range number 1")
    parser.add_argument('-d0', "--distance0", type=float, default=0.01, help="Distance range number 0")
    parser.add_argument('-num_train', "--training_sample", type=int, default=1280, help="Number of training samples.")
    parser.add_argument('-num_valid', "--validating_sample", type=int, default=0, help="Number of validating samples.")
    parser.add_argument('-num_test', "--testing_sample", type=int, default=0, help="Number of testing samples.")
    parser.add_argument('-batch', "--batch_size", type=int, default=4, help="Size of batch data")
    parser.add_argument('-epc', "--epochs", type=int, default=5, help="Number of epoch")
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.00001, help="Learning rate of model")
    # parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])

    args = parser.parse_args()
    # system = SysMod(args)   # Change dl_rate_calculator for loss function
                            # Add function calculate channel-estimation error

    # Import data from .csv files
    mean_rate, mean_min, mean_max, train_dataloader, test_dataloader = dataloader_creation(args)

    #add another metric that compute the channel-estimation error to show that pilot assignment help channel estimation process
    dl_rate_train, dl_rate_test = model_training(args, train_dataloader, test_dataloader)

    plot_dl_rate(dl_rate_train, dl_rate_test, mean_rate, mean_min, mean_max)













    # ------------------------------------------------------------------------------------ #

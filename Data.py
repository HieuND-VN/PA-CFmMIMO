import numpy as np
import pandas as pd



def generate_dataset(args, system):
    num_ap = system.num_ap
    num_ue = system.num_ue

    total_dataset = []
    total_sample = args.training_sample + args.validating_sample + args.testing_sample
    for i in range(total_sample):
        print(f'Generating data sample [{i}]/[{total_sample}]')
        total_dataset.append(system.data_sample_generator())

    total_dataset = np.array(total_dataset)

    df = pd.DataFrame(total_dataset, columns=['sum_rate', 'min_rate'] + [f'beta_{i}' for i in range(num_ap*num_ue)] + [f'tau_{i}' for i in range(num_ue)])




    if (args.number_UE == 20) and (args.number_AP == 40) and (args.pilot_length == 5):
        # case1: 20 UEs - 40 APs - 05 pilots
        df.to_csv('20UE_40AP_05pilot.csv', index = False)
    elif (args.number_UE == 20) and (args.number_AP == 40) and (args.pilot_length == 10):
        # case2: 20 UEs - 40 APs - 10 pilots
        df.to_csv('20UE_40AP_10pilot.csv', index = False)
    elif (args.number_UE == 20) and (args.number_AP == 40) and (args.pilot_length == 15):
        # case3: 20 UEs - 40 APs - 15 pilots
        df.to_csv('20UE_40AP_15pilot.csv', index = False)
    elif (args.number_UE == 20) and (args.number_AP == 60) and (args.pilot_length == 5):
        # case4: 20 UEs - 60 APs - 5 pilots
        df.to_csv('20UE_60AP_05pilot.csv', index = False)
    elif (args.number_UE == 20) and (args.number_AP == 60) and (args.pilot_length == 10):
        # case5: 20 UEs - 60 APs - 10 pilots
        df.to_csv('20UE_60AP_10pilot.csv', index = False)
    elif (args.number_UE == 20) and (args.number_AP == 60) and (args.pilot_length == 15):
        # case6: 20 UEs - 60 APs - 15 pilots
        df.to_csv('20UE_60AP_15pilot.csv', index = False)
    elif (args.number_UE == 30) and (args.number_AP == 40) and (args.pilot_length == 5):
        # case7: 30 UEs - 40 APs - 5 pilots
        df.to_csv('30UE_40AP_05pilot.csv', index = False)
    elif (args.number_UE == 30) and (args.number_AP == 40) and (args.pilot_length == 10):
        # case8: 30 UEs - 40 APs - 10 pilots
        df.to_csv('30UE_40AP_10pilot.csv', index = False)
    elif (args.number_UE == 30) and (args.number_AP == 40) and (args.pilot_length == 15):
        # case9: 30 UEs - 40 APs - 15 pilots
        df.to_csv('30UE_40AP_15pilot.csv', index = False)
    elif (args.number_UE == 30) and (args.number_AP == 40) and (args.pilot_length == 20):
        # case10: 30 UEs - 40 APS - 20 pilots
        df.to_csv('30UE_40AP_20pilot.csv', index = False)
    elif (args.number_UE == 30) and (args.number_AP == 60) and (args.pilot_length == 5):
        # case11: 30 UEs - 60 APs - 5 pilots
        df.to_csv('30UE_60AP_05pilot.csv', index = False)
    elif (args.number_UE == 30) and (args.number_AP == 60) and (args.pilot_length == 10):
        # case12: 30 UEs - 60 APs - 10 pilots
        df.to_csv('30UE_60AP_10pilot.csv', index = False)
    elif (args.number_UE == 30) and (args.number_AP == 60) and (args.pilot_length == 15):
        # case13: 30 UEs - 60 APs - 15 pilots
        df.to_csv('30UE_60AP_15pilot.csv', index = False)
    elif (args.number_UE == 30) and (args.number_AP == 60) and (args.pilot_length == 20):
        # case14: 30 UEs - 60 APs - 20 pilots
        df.to_csv('30UE_60AP_20pilot.csv', index = False)
    else:
        print("No accurate choice")


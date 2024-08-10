import time
from create_data import *


num_ue_example = [20,30]
num_ap_example = [40,60]
tau_p_example = [5,10,15,20]

for num_ue in num_ue_example:
    for num_ap in num_ap_example:
        for tau_p in tau_p_example:
            start_time = time.time()
            generate_dataset_new(num_ue, num_ap, tau_p)
            end_time = time.time()
            running_time = end_time - start_time
            print(f'Generating {num_ue}UE_{num_ap}AP_{tau_p}pilot.csv dataset... {running_time: .3f}s')


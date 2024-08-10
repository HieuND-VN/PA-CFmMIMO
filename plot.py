import numpy as np
import matplotlib.pyplot as plt

def plot_dl_rate(dl_rate_train, dl_rate_test, mean_rate, mean_min, mean_max):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot the first subplot
    axs[0].plot(dl_rate_train, marker='o', linestyle='-', color='b', label='Avg DL rate')
    axs[0].set_title('Average Downlink Rate in Training')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Rate')
    axs[0].grid(True)

    # Add a straight line to the first subplot
    mean_rate_greedy = mean_rate*np.ones(len(dl_rate_train))
    mean_min_greedy = mean_min*np.ones(len(dl_rate_train))
    mean_max_greedy = mean_max*np.ones(len(dl_rate_train))
     # Example: horizontal line at y=0.5
    axs[0].plot(mean_rate_greedy, linestyle='--', color='g', label='mean_rate_greedy')

    # Plot the second subplot
    axs[1].plot(dl_rate_test, marker='o', linestyle='--', color='r', label='Data 2')
    axs[1].set_title('Average Downlink Rate in Testing')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Rate')
    axs[1].grid(True)

    # Add a straight line to the second subplot
    axs[1].plot(mean_rate_greedy, linestyle='-.', color='m', label='mean_rate_greedy')

    # Add legends to each subplot
    axs[0].legend()
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()
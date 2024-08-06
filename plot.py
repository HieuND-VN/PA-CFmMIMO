import numpy as np
import matplotlib.pyplot as plt

# Create a NumPy array of shape (40,)
data = np.array([0.35193422 ,0.35173404 ,0.35181034 ,0.35188407 ,0.35189207 ,0.35187054,
 0.3518908  ,0.35189173 ,0.35179748 ,0.35181877 ,0.35192076 ,0.352085,
 0.35208574 ,0.35201422 ,0.35205462 ,0.35204653 ,0.35204711 ,0.35202509,
 0.35202667 ,0.35202206 ,0.35201982 ,0.35211959 ,0.35212046 ,0.35193733,
 0.35193734 ,0.35193734 ,0.3519291  ,0.3519291  ,0.3519293  ,0.3519293,
 0.3519293  ,0.3519293  ,0.3519293  ,0.3519293  ,0.3519293  ,0.3519293,
 0.3519293  ,0.3519293  ,0.3519293  ,0.35192917])  # Example data; replace with your actual array
greedy = 0.3511739964380227*np.ones(40)
# Create a line plot
plt.plot(data, marker='o', linestyle='-', color='b', label = "HQCNN")  # Marker and linestyle can be customized
plt.plot(greedy, linestyle='--', color='r', label = "greedy")
# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of a (40,) Array')
plt.legend()

# Display the plot
plt.show()
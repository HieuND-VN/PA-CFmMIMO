import numpy as np
import matplotlib.pyplot as plt

# # Create a NumPy array of shape (40,)
# data = np.array([0.35193422 ,0.35173404 ,0.35181034 ,0.35188407 ,0.35189207 ,0.35187054,
#  0.3518908  ,0.35189173 ,0.35179748 ,0.35181877 ,0.35192076 ,0.352085,
#  0.35208574 ,0.35201422 ,0.35205462 ,0.35204653 ,0.35204711 ,0.35202509,
#  0.35202667 ,0.35202206 ,0.35201982 ,0.35211959 ,0.35212046 ,0.35193733,
#  0.35193734 ,0.35193734 ,0.3519291  ,0.3519291  ,0.3519293  ,0.3519293,
#  0.3519293  ,0.3519293  ,0.3519293  ,0.3519293  ,0.3519293  ,0.3519293,
#  0.3519293  ,0.3519293  ,0.3519293  ,0.35192917])  # Example data; replace with your actual array
# greedy = 0.3511739964380227*np.ones(40)
# # Create a line plot
# plt.plot(data, marker='o', linestyle='-', color='b', label = "HQCNN")  # Marker and linestyle can be customized
# plt.plot(greedy, linestyle='--', color='r', label = "greedy")
# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Line Plot of a (40,) Array')
# plt.legend()

# Display the plot
plt.show()

import numpy as np
import scipy.linalg as la
import cvxpy as cp

# Parameters
M = 100  # Number of access points
K = 40  # Number of terminals
D = 1  # Square area in km^2
tau = 20  # Training length
B = 20  # MHz
Hb = 15  # Base station height in meters
Hm = 1.65  # Mobile height in meters
f = 1900  # Frequency in MHz
aL = (1.1 * np.log10(f) - 0.7) * Hm - (1.56 * np.log10(f) - 0.8)
L = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(Hb) - aL
power_f = 0.2  # Downlink power in mW
noise_p = 10 ** ((-203.975 + 10 * np.log10(20 * 10 ** 6) + 9) / 10)  # Noise power
Pd = power_f / noise_p  # Normalized receive SNR
Pp = Pd  # Pilot power
d0 = 0.01  # km
d1 = 0.05  # km
N = 200  # Number of realizations
sigma_shd = 8  # Shadowing standard deviation in dB
D_cor = 0.1  # Correlation distance

# Initialize results
R_cf_min = np.zeros(N)
R_cf_opt_min = np.zeros(N)
R_sc_min = np.zeros(N)
R_sc_opt_min = np.zeros(N)
R_cf_user = np.zeros((N, K))
R_sc_user = np.zeros((N, K))

for n in range(N):
 print(n)

 # Random locations of M APs
 AP = np.zeros((M, 2, 9))
 AP[:, :, 0] = np.random.uniform(-D / 2, D / 2, (M, 2))

 # Wrapped around (8 neighbor cells)
 D1 = np.zeros((M, 2))
 D1[:, 0] = D * np.ones(M)
 AP[:, :, 1] = AP[:, :, 0] + D1

 D2 = np.zeros((M, 2))
 D2[:, 1] = D * np.ones(M)
 AP[:, :, 2] = AP[:, :, 0] + D2

 D3 = np.zeros((M, 2))
 D3[:, 0] = -D * np.ones(M)
 AP[:, :, 3] = AP[:, :, 0] + D3

 D4 = np.zeros((M, 2))
 D4[:, 1] = -D * np.ones(M)
 AP[:, :, 4] = AP[:, :, 0] + D4

 D5 = np.zeros((M, 2))
 D5[:, 0] = D
 D5[:, 1] = -D
 AP[:, :, 5] = AP[:, :, 0] + D5

 D6 = np.zeros((M, 2))
 D6[:, 0] = -D
 D6[:, 1] = D
 AP[:, :, 6] = AP[:, :, 0] + D6

 D7 = np.zeros((M, 2))
 D7 = D * np.ones((M, 2))
 AP[:, :, 7] = AP[:, :, 0] + D7

 D8 = np.zeros((M, 2))
 D8 = -D * np.ones((M, 2))
 AP[:, :, 8] = AP[:, :, 0] + D8

 # Random locations of K terminals
 Ter = np.zeros((K, 2, 9))
 Ter[:, :, 0] = np.random.uniform(-D / 2, D / 2, (K, 2))

 # Wrapped around (8 neighbor cells)
 D1 = np.zeros((K, 2))
 D1[:, 0] = D * np.ones(K)
 Ter[:, :, 1] = Ter[:, :, 0] + D1

 D2 = np.zeros((K, 2))
 D2[:, 1] = D * np.ones(K)
 Ter[:, :, 2] = Ter[:, :, 0] + D2

 D3 = np.zeros((K, 2))
 D3[:, 0] = -D * np.ones(K)
 Ter[:, :, 3] = Ter[:, :, 0] + D3

 D4 = np.zeros((K, 2))
 D4[:, 1] = -D * np.ones(K)
 Ter[:, :, 4] = Ter[:, :, 0] + D4

 D5 = np.zeros((K, 2))
 D5[:, 0] = D
 D5[:, 1] = -D
 Ter[:, :, 5] = Ter[:, :, 0] + D5

 D6 = np.zeros((K, 2))
 D6[:, 0] = -D
 D6[:, 1] = D
 Ter[:, :, 6] = Ter[:, :, 0] + D6

 D7 = np.zeros((K, 2))
 D7 = D * np.ones((K, 2))
 Ter[:, :, 7] = Ter[:, :, 0] + D7

 D8 = np.zeros((K, 2))
 D8 = -D * np.ones((K, 2))
 Ter[:, :, 8] = Ter[:, :, 0] + D8

 # Generate U matrix
 U, _, _ = la.svd(np.random.randn(tau, tau))

 # Create the shadowing matrix
 Dist = np.zeros((M, M))
 Cor = np.zeros((M, M))

 for m1 in range(M):
  for m2 in range(M):
   Dist[m1, m2] = min([
    np.linalg.norm(AP[m1, :, 0] - AP[m2, :, 0]),
    np.linalg.norm(AP[m1, :, 0] - AP[m2, :, 1]),
    np.linalg.norm(AP[m1, :, 0] - AP[m2, :, 2]),
    np.linalg.norm(AP[m1, :, 0] - AP[m2, :, 3]),
    np.linalg.norm(AP[m1, :, 0] - AP[m2, :, 4]),
    np.linalg.norm(AP[m1, :, 0] - AP[m2, :, 5]),
    np.linalg.norm(AP[m1, :, 0] - AP[m2, :, 6]),
    np.linalg.norm(AP[m1, :, 0] - AP[m2, :, 7]),
    np.linalg.norm(AP[m1, :, 0] - AP[m2, :, 8])
   ])
   Cor[m1, m2] = np.exp(-np.log(2) * Dist[m1, m2] / D_cor)

 A1 = la.cholesky(Cor, lower=True)
 x1 = np.random.randn(M)
 sh_AP = np.dot(A1, x1)

 for m in range(M):
  sh_AP[m] = (1 / np.sqrt(2)) * sigma_shd * sh_AP[m] / np.linalg.norm(A1[m, :])

 # K correlated shadowing matrix
 Dist = np.zeros((K, K))
 Cor = np.zeros((K, K))

 for k1 in range(K):
  for k2 in range(K):
   Dist[k1, k2] = min([
    np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, 0]),
    np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, 1]),
    np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, 2]),
    np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, 3]),
    np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, 4]),
    np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, 5]),
    np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, 6]),
    np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, 7]),
    np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, 8])
   ])
   Cor[k1, k2] = np.exp(-np.log(2) * Dist[k1, k2] / D_cor)

 A2 = la.cholesky(Cor, lower=True)
 x2 = np.random.randn(K)
 sh_Ter = np.dot(A2, x2)

 for k in range(K):
  sh_Ter[k] = (1 / np.sqrt(2)) * sigma_shd * sh_Ter[k] / np.linalg.norm(A2[k, :])

 # Small-cell system: pilot assignment and shadowing impact
 C = np.zeros((K, K, M))
 # Fill C with appropriate data based on your system model

 H = np.zeros((M, K))
 for k in range(K):
  for m in range(M):
   d = np.linalg.norm(AP[m, :, 0] - Ter[k, :, 0])
   H[m, k] = (10 ** (-L / 20)) / d ** 2

 # Shadowing impact
 H *= 10 ** (sh_AP / 20)

 # Cell-free system: pilot assignment and shadowing impact
 # Use the results from small-cell system to configure the cell-free system as needed

 # Optimize power control
 R = cp.Variable((M, K), complex=True)
 constraints = []
 for m in range(M):
  for k in range(K):
   constraints.append(cp.norm(R[m, k], 'fro') <= Pd)
 objective = cp.Maximize(cp.sum(cp.log(1 + R @ R.H / noise_p)))
 prob = cp.Problem(objective, constraints)
 prob.solve()

 # Compute rates
 R_cf = np.zeros((K,))
 R_sc = np.zeros((K,))

 # Update results based on optimized values
 R_cf_min[n] = np.min(R_cf)
 R_sc_min[n] = np.min(R_sc)
 R_cf_user[n, :] = R_cf
 R_sc_user[n, :] = R_sc

print("Cell-free system minimum rate:", np.mean(R_cf_min))
print("Small-cell system minimum rate:", np.mean(R_sc_min))

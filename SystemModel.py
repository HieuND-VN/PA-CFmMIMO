import numpy as np


class SysMod():
    def __init__(self, args):
        self.num_ap = args.number_AP
        self.num_ue = args.number_UE
        self.tau_p = args.pilot_length
        self.length = args.area_length #region from -1 to 1 in both axis (km)
        self.f = args.frequency
        self.h_ap = args.height_AP #km
        self.h_ue = args.height_UE #km
        self.d1 = args.distance1
        self.d0 = args.distance0
        self.B = args.bandwidth
        self.L = 46.5 + 33.9*np.log10(self.f) - 13.82*np.log10(self.h_ap) - (1.1*np.log10(self.f) - 0.7)*self.h_ue + (1.56*np.log10(self.f) - 0.8)
        self.Pd = (self.B*10**(-17.4)*10**(-3))**-1
        self.p = args.power_pilot #1mW, Power UL of pilot signal
        self.P = args.power_AP #mW, Power Downlink in each AP
        self.eta = self.P / self.num_ue
        self.pilot_index_random = np.random.randint(self.tau_p-1, size = self.num_ue)
        self.batch_size = args.batch_size

        self.AP_position = self.AP_location_generator()

    
    def AP_location_generator(self):
        sensitivity_1 = 0.1
        sensitivity_2 = 0.2
        AP_position = np.zeros(self.num_ap, dtype = "complex")
        theta = np.linspace(0,2*np.pi,self.num_ap)
        for i in range(self.num_ap):
            AP_position[i] = self.length * sensitivity_1 * np.cos(theta[i]) + 1j * self.length * sensitivity_2 * np.sin(theta[i])
        return AP_position

    def distance_calculator(self, UE_position):
        diff = self.AP_position[:, np.newaxis] - UE_position[np.newaxis, :]
        return (np.sqrt(np.real(diff) ** 2 + np.imag(diff) ** 2)).reshape(self.num_ap, self.num_ue)

    def LSF_calculator(self, distanceUE2AP):
        pathloss = np.zeros_like(distanceUE2AP)
        pathloss[distanceUE2AP > self.d1] = -self.L - 35 * np.log10(distanceUE2AP[distanceUE2AP > self.d1])
        pathloss[(distanceUE2AP <= self.d1) & (distanceUE2AP > self.d0)] = -self.L - 15 * np.log10(self.d1) - 20 * np.log10(distanceUE2AP[(distanceUE2AP <= self.d1) & (distanceUE2AP > self.d0)])
        pathloss[distanceUE2AP <= self.d0] = -self.L - 15 * np.log10(self.d1) - 20 * np.log10(self.d0)
        beta = self.Pd*10**(pathloss/10)
        return beta

    def calculate_c(self, pilot_index, beta):
        c = np.zeros((self.num_ap, self.num_ue))
        for m in range(self.num_ap):
            for k in range(self.num_ue):
                # Tính tử số
                numerator = np.sqrt(self.tau_p * self.p) * beta[m, k]
                sum = 0
                # Tính mẫu sổ
                for j in range(self.num_ue):
                    if (pilot_index[j] == pilot_index[k]):
                        sum += beta[m, j]
                denominator = self.tau_p * self.p * sum + 1
                c[m, k] = numerator / denominator
        return c

    def sinr_calculator(self, pilot_index, beta, gamma):
        sinr = np.zeros(self.num_ue)
        for k in range(self.num_ue):
            # DS_k
            numerator = 0
            numerator = self.P * self.eta * (np.sum(gamma[:, k])) ** 2

            # denominator
            demoninator = 0
            # UI
            UI = 0
            sum_12 = 0
            for j in range(self.num_ue):
                # if (j==k): continue;
                # if (pilot_index[j] == pilot_index[k]): continue;
                if (j != k) and (pilot_index[j] == pilot_index[k]):
                    sum_11 = 0
                    for m in range(self.num_ap):
                        sum_11 += (gamma[m, j] * beta[m, k] / beta[m, j])
                else:
                    sum_11 = 0
                sum_12 += sum_11 ** 2
            UI = self.P * self.eta * sum_12

            # BU
            BU = 0
            sum_BU = 0
            for j in range(self.num_ue):
                for m in range(self.num_ap):
                    sum_BU += gamma[m, j] * beta[m, k]
            BU = self.P * self.eta * sum_BU
            denominator = UI + BU + 1
            sinr[k] = numerator / denominator
        return sinr
    def dl_rate_calculator(self, pilot_index, beta):
        c_mk = self.calculate_c(pilot_index,beta)
        gamma = np.sqrt(self.tau_p * self.p) * beta * c_mk
        sinr = self.sinr_calculator(pilot_index, beta, gamma)
        return np.log2(1+sinr)

    def sum_dl_rate_calculator(self,beta, pilot_index, i):
        sum_rate_batch = 0
        pilot_index = pilot_index.detach().numpy() #(batch_size, num_ue, tau_p)
        pilot_index = np.argmax(pilot_index, axis=-1)
        # print(f'{i}.. pilot_index information: {np.shape(pilot_index)}')
        beta = beta.numpy()
        beta = beta.reshape(self.batch_size, self.num_ap, self.num_ue)
        for i in range(len(beta)):
            sum_rate_batch += np.sum(self.dl_rate_calculator(pilot_index[i], beta[i]))
        return sum_rate_batch
    def greedy_assignment(self, beta, N):
        # So sánh với random Pilot-Assignment thì sẽ so với 2 initial pilot_index như nhau
        # pilot_index = self.pilot_index_random


        pilot_index = np.random.randint(self.tau_p-1, size = self.num_ue)
        for n in range(N):
            dl_rate = self.dl_rate_calculator(pilot_index, beta)
            k_star = np.argmin(dl_rate)
            sum_beta = np.zeros(self.tau_p)
            for tau in range(self.tau_p):
                for m in range(self.num_ap):
                    for k in range(self.num_ue):
                        if (k != k_star) and (pilot_index[k] == tau):
                            sum_beta[tau] += beta[m, k]
            pilot_index[k_star] = np.argmin(sum_beta)

        rate_list = self.dl_rate_calculator(pilot_index, beta)
        sum_rate = np.sum(rate_list)
        min_rate = np.min(rate_list)
        return pilot_index, sum_rate, min_rate

    def data_sample_generator(self):
        UE_position = np.random.uniform(low=-self.length, high=self.length, size=self.num_ue) + 1j*np.random.uniform(low=-self.length, high=self.length, size=self.num_ue)
        distanceUE2AP = self.distance_calculator(UE_position)
        beta = self.LSF_calculator(distanceUE2AP)
        pilot_index, sum_rate, min_rate = self.greedy_assignment(beta, N = 20)

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


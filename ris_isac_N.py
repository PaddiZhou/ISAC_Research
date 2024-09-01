import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(42)
# Parameter Initialization
# N = 169  # Number of RIS elements
M = 8    # Number of antennas at the base station
K = 4    # Number of users

noise_power_dbm = -80  # Noise power in dBm
noise_power_linear = 10 ** (noise_power_dbm / 10)  # Convert noise power to linear scale

# Channel and Path Loss Parameters
d_BS_RIS = 50  # Distance from Base Station to RIS in meters
d_RIS_User = 4  # Distance from RIS to User in meters
d_BS_User = 50 + 4 * np.random.randn()  # Distance from Base Station to User with random perturbation

C0_db = -30  # Path loss constant in dB

D0 = 1  # Reference distance in meters

alpha_BR = 2.5  # Path loss exponent from BS to RIS
alpha_Ru = 2.5  # Path loss exponent from RIS to User
alpha_Bu = 3.5  # Path loss exponent from BS to User

beta_BR = 10 ** (3 / 10)  # Rician factor for BS to RIS link
beta_Ru = 10 ** (3 / 10)  # Rician factor for RIS to User link
beta_Bu = 0  # Rician factor for BS to User link (pure Rayleigh fading)

# Path Loss Calculation Function
def path_loss(d, alpha, C0_db, D0):
    PL = 10 ** (C0_db / 10) * (d / D0) ** (-alpha)  # Path loss model
    return PL

# Rician Channel Generation Function
def rician_channel(N, M, beta, PL):
    LoS_component = np.sqrt(PL) * (np.ones((N, M)) + 1j * np.ones((N, M)))  # Line-of-sight component
    NLoS_component = np.sqrt(PL) * (np.random.randn(N, M) + 1j * np.random.randn(N, M))  # Non-line-of-sight component
    H = np.sqrt(beta / (1 + beta)) * LoS_component + np.sqrt(1 / (1 + beta)) * NLoS_component  # Rician channel matrix
    return H



# Loop over Power Range for Performance Evaluation
RIS_elements_number = np.linspace(100, 200, 6)  # Power range in dBm from 10 to 20
sum_rate = np.zeros(len(RIS_elements_number))
counter = 0

for N in RIS_elements_number:
    N = int(N)
    # Channel Generation
    PL_BS_RIS = path_loss(d_BS_RIS, alpha_BR, C0_db, D0)  # Path loss from BS to RIS
    PL_RIS_User = path_loss(d_RIS_User, alpha_Ru, C0_db, D0)  # Path loss from RIS to User
    PL_BS_User = path_loss(d_BS_User, alpha_Bu, C0_db, D0)  # Path loss from BS to User

    G = rician_channel(N, M, beta_BR, PL_BS_RIS)  # Channel from BS to RIS
    hr_k = rician_channel(K, N, beta_Ru, PL_RIS_User)  # Channel from RIS to User
    hd_k = rician_channel(K, M, beta_Bu, PL_BS_User)  # Direct channel from BS to User

    # Random Phase Angle Generation and Phase Vector Calculation
    theta = 2 * np.pi * np.random.rand(N, 1)  # Random phase angles
    phi = np.exp(1j * theta)  # Corresponding complex exponentials
    Phi_random = np.diagflat(phi)  # Diagonal phase shift matrix

    # Overall Channel Combining RIS reflection and Direct link
    hk = hr_k @ Phi_random @ G + hd_k
    hk = hk.T

    P_total = 10
    W = np.random.randn(M, K) + 1j * np.random.randn(M, K)  # Communication signal matrix
    normm = np.linalg.norm(W, 'fro')  # Frobenius norm
    W = np.sqrt(P_total) * W / normm  # Normalize
    # Calculate received signals and SINRs for each user
    rec_signal = np.zeros(K, dtype=complex)
    c = np.zeros(K, dtype=complex)
    g = np.zeros(K, dtype=complex)

    for _ in range(200):

        for k in range(K):
            rec_signal[k] = hk[:, k].conj().T @ W[:, k]  # Received signal for user k
            signal_power = np.abs(rec_signal[k])**2  # Signal power for user k

            other_signal_power = 0  # Initialize interference power
            for j in range(K):
                if j != k:
                    denum_c = hk[:, k].conj().T @ W[:, j]
                    other_signal_power += np.abs(denum_c)**2  # Accumulate interference power

            c[k] = signal_power / (other_signal_power + noise_power_linear)  # Calculate SINR for user k
            g[k] = (np.sqrt(1 + np.abs(c[k])) * rec_signal[k]) / (signal_power + other_signal_power + noise_power_linear)  # Calculate optimal combining coefficient for user k

        # CVX Optimization to Maximize the Objective Function
        Wc = cp.Variable((M, K), complex=True)  # Communication signal matrix
        objective = 0

        for k in range(K):
            total_power = 0
            for j in range(K):
                signal_j = hk[:, k].conj().T @ Wc[:, j]  # Signal for each user and radar
                total_power += cp.square(cp.abs(signal_j))  # Total power calculation

            # Modify the objective function to avoid complex values
            objective += 2 * np.sqrt(1 + np.abs(c[k])) * cp.real(cp.conj(g[k]) * hk[:, k].conj().T @ Wc[:, k]) - cp.square(cp.abs(g[k])) * total_power

        # Objective function: Maximize SINR-based utility
        prob = cp.Problem(cp.Maximize(objective), [cp.sum_squares(cp.vec(Wc)) <= P_total])
        prob.solve(solver=cp.MOSEK)

        W = Wc.value

    # Store results for each power level
    SNR = np.zeros(K)
    for k in range(K):
        rec_signal[k] = hk[:, k].conj().T @ W[:, k]  # Received signal for user k
        signal_power = np.abs(rec_signal[k])**2  # Signal power for user k

        other_signal_power = 0  # Initialize interference power
        for j in range(K):
            if j != k:
                denum_c = hk[:, k].conj().T @ W[:, j]
                other_signal_power += np.abs(denum_c)**2  # Accumulate interference power

        SNR[k] = signal_power / (other_signal_power + noise_power_linear)  # Calculate SINR for user k

    sum_rate[counter] = np.sum(np.log2(1 + SNR))
    counter += 1


plt.plot(RIS_elements_number, sum_rate)
plt.grid(True)
plt.xlabel('Power (dBm)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('Sum Rate vs. Power')

# Set limits correctly
plt.xlim(100, 200)  # Correct limits for the x-axis
plt.ylim(3, 18)  # Correct limits for the y-axis

# Correct ticks for better granularity
plt.xticks(range(100, 201, 20))  # Ticks from 100 to 200 every 20 units
plt.yticks(range(3, 19, 3))  # Ticks from 3 to 18 every 3 units

plt.show()
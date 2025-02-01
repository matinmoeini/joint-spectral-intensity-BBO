import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Constants
l = 10e-3
sigmapump = 0.1e-9
threshold = 0  # Example threshold for normalization
landapump_initial = 0.355  # Initial value of landapump
iterations = 150  # Number of iterations
landa1type2fix = 0.710
landa2type2fix = 0.710

# Define ranges for phi and theta
phi_values = np.linspace(0, 2 * np.pi, 100)  # X-axis values
theta_values = np.linspace(0, 30, 100) * np.pi / 180  # Y-axis values
phi, theta = np.meshgrid(phi_values, theta_values)

# Fixed theta value in radians
theta_fixed = 34.2 * (np.pi / 180)

# Prepare a grid for landa1type2 and landa2type2
landa1type2_values = np.linspace(0.69, 0.73, iterations)  # Adjust the range as needed
landa2type2_values = np.linspace(0.69, 0.73, iterations)  # Adjust the range as needed

# Create a 2D grid to store intensities
intensity_grid = np.zeros((iterations, iterations))

# Loop over landa1type2 and landa2type2
for i, landa1type2 in enumerate(landa1type2_values):
    for j, landa2type2 in enumerate(landa2type2_values):
        landapump = 1 / ((1 / landa1type2) + (1 / landa2type2))

        # Precomputed refractive indices
        no1type2 = np.sqrt(2.7359 + (0.01878 / (landa1type2 ** 2 - 0.01822)) - (0.01354 * (landa1type2 ** 2)))
        nopump = np.sqrt(2.7359 + (0.01878 / (landapump ** 2 - 0.01822)) - (0.01354 * (landapump ** 2)))
        nepump = np.sqrt(2.3753 + (0.01224 / (landapump ** 2 - 0.01667)) - (0.01516 * (landapump ** 2)))
        nethetapump_value = np.sqrt(1 / ((np.sin(theta_fixed) ** 2 / nepump ** 2) +
                                         (np.cos(theta_fixed) ** 2 / nopump ** 2)))

        no2type2 = np.sqrt(2.7359 + (0.01878 / (landa2type2 ** 2 - 0.01822)) - (0.01354 * (landa2type2 ** 2)))

        value = landa2type2 * no1type2 * np.sin(theta) / (landa1type2 * no2type2)
        value = np.clip(value, -1, 1)  # Ensure valid arcsin input
        thetaps = np.arcsin(value)

        eq2type2 = (nethetapump_value / (1e-6 * landapump)) - \
                   (no2type2 * np.sqrt(1 - value ** 2) / (1e-6 * landa2type2)) - \
                   (no1type2 * np.cos(theta) / (1e-6 * landa1type2))

        # Gaussian factor
        gaussian_factor = np.exp(-((landapump -landapump_initial)*1e-6)** 2 /
              (2 * sigmapump ** 2))

        phase_factor = np.sinc(eq2type2 * l / 2)
        exponential_factor = np.exp(-1j * eq2type2 * l / 2)

        # Intensity calculation: Use np.abs for complex number handling
        intensity = gaussian_factor * phase_factor * np.abs(exponential_factor)
        intensity=intensity**2
        intensity_grid[i, j] = np.sum(intensity)

# Normalize intensities
normalized_intensity = intensity_grid 

# Create a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(normalized_intensity, extent=[landa2type2_values[0], landa2type2_values[-1],
                                         landa1type2_values[0], landa1type2_values[-1]],
           origin='lower', cmap='gray', aspect='auto')
plt.colorbar(label="Intensity")
plt.xlabel("Signal Wavelength [µm]", fontsize=16)
plt.ylabel("Idler Wavelength [µm]", fontsize=16)
plt.title("Intensity Heatmap for Signal and Idler Wavelengths", fontsize=16)
plt.grid(True)
plt.show()

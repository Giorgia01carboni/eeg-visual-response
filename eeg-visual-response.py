import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1) Load the data. In Matlab, you may use 𝑐𝑠𝑣𝑟𝑒𝑎𝑑(‘𝑣𝑒𝑝.𝑐𝑠𝑣’)
# and store the epochs in the matrix 𝑉.
data = pd.read_csv("VEP.csv")

V = data.to_numpy()

# 2) Define the time vector 𝒕 and plot some EEG realizations (all is fine too).
epoch_length = len(V[0])
frequency = 350
delta_t = 1 / 350

time_vector = np.arange(0, epoch_length * delta_t, delta_t)

plt.plot(time_vector, V[0])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('EEG Realization')
plt.show()

# 3) Estimate the ensemble average for the process
ensemble_average = np.mean(V, axis=0)
#print(ensemble_average)

plt.plot(time_vector, ensemble_average)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Ensemble Average')
plt.show()

# 3.1) Variance to check if the process is stationary
variance = np.var(V, axis = 0)
plt.plot(time_vector, variance)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Variance')
plt.show()

# 4) From each realization stored in the matrix V, remove the ensemble average,
# and store the detrended realizations in the matrix V2
# V2 contains the detrended signal
V2 = V - ensemble_average
if len(V2.shape) != len(V.shape):
  print("Error in V2 shape")

plt.plot(time_vector, V2[0])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('De-trended EEG Realization')
plt.show()

# 4.1) Ensembled average of V2
ensemble_average_V2 = np.mean(V2, axis=0)
#print(ensemble_average_V2)
time_average_V2 = np.mean(V2, axis=1)
#print(time_average_V2)

epochs_vector = np.arange(1, len(V2) + 1)

plt.plot(time_vector, ensemble_average_V2, label='Ensemble Average of V2', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Ensemble Average of V2')
plt.show()

plt.plot(epochs_vector, time_average_V2, label='Temporal Average of V2', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Temporal Average of V2')
plt.show()

# 5) Compute the time variance of 𝑉2 and compare it with its ensemble variance.
time_variance = np.var(V2, axis=1)
plt.plot(epochs_vector * delta_t, time_variance, label='Time Variance of V2', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Time Variance of V2')
plt.show()
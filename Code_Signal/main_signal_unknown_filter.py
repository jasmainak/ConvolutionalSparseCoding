"""
This code is used to find a kernel and the corresponding codes for signal data
"""
import numpy as np
import matplotlib.pyplot as plt
import CSC_signal as CSC
import time

from mne import create_info
from mne.io import RawArray
from utils import load_data

start = time.clock()

eeg_signal, sfreq = load_data('simulated', noise_level=0, random_state=42)

n_signal = eeg_signal.shape[0]  # the number of signals
n_sample = eeg_signal.shape[1]  # dimension of each signal

# Normalize the data to be run
b = eeg_signal
b = ((b.T - np.mean(b, axis=1)) / np.std(b, axis=1)).T

# Define the parameters
# 128 kernels with size of 201
size_kernel = [2, 45]

ch_names = ['EEG%03d' % i for i in range(n_signal)]
info = create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = RawArray(eeg_signal * 1e-6, info)
raw.plot(scalings=dict(eeg='auto'), duration=300)

# Optim options
max_it = 30  # the number of iterations
tol = np.float64(1e-3)  # the stop threshold for the algorithm

# RUN THE ALGORITHM
[d, z, Dz, list_obj_val, list_obj_val_filter, list_obj_val_z, reconstr_err] = \
    CSC.learn_conv_sparse_coder(b, size_kernel, max_it, tol, random_state=42)

plt.figure()
plt.plot(d[0, :])
plt.plot(d[1, :])
plt.show()

end = time.clock()

print end - start

"""Transforms raw simulation data into a shape that can be used for machine learning."""
import h5py
import numpy as np
import os
import random

SIMULATION_NAME = '48_48_32_1500_0.71_0.01_0.3_1000.2'

data_dir = os.path.dirname(os.path.realpath(__file__))


# get simulation data
sim_file = h5py.File(
    os.path.join(data_dir, '..', 'simulation', '3d', 'data', SIMULATION_NAME, 'sim.h5'), 
    'r')
sim_data = sim_file['data']
h, d, w, _, N = sim_data.shape

# standardize data
mean = np.mean(sim_data, axis=-1, keepdims=True)
std = np.std(sim_data, axis=-1, keepdims=True)
sim_data = np.divide((sim_data - mean), std, out=np.zeros_like(sim_data), where=std!=0)


# train/test split
N_train = int(0.8*N)
N_test = N-N_train

tf_h5 = h5py.File(os.path.join(data_dir, f'{SIMULATION_NAME}.h5'), 'w')
train_data = tf_h5.create_dataset("train", (N_train, w, d, h, 4), chunks=(1, w, d, h, 1))
test_data = tf_h5.create_dataset("test", (N_test, w, d, h, 4), chunks=(1, w, d, h, 1))
train_data.attrs['N'] = N_train
test_data.attrs['N'] = N_test


indices = list(range(N))
random.shuffle(indices)
train_indices = indices[:N_train]
test_indices = indices[-N_test:]

for i, sample in zip(range(N_train), train_indices):
    train_data[i, :, :, :, :] = np.transpose(sim_data[:, :, :, :, sample], [2, 1, 0, 3])
    
    if i%10==0: print(f'train set: {i}/{N_train}')
    
    
for i, sample in zip(range(N_test), test_indices):
    test_data[i, :, :, :, :] = np.transpose(sim_data[:, :, :, :, sample], [2, 1, 0, 3])
    
    if i%10==0: print(f'test set: {i}/{N_test}')

# save mean and std tensor to be able to reverse standardization
mean_data = tf_h5.create_dataset("mean", (w, d, h, 4), chunks=(w, d, h, 1))
mean_data[:, :, :, :] = np.transpose(mean[:, :, :, :, 0], [2, 1, 0, 3])
std_data = tf_h5.create_dataset("std", (w, d, h, 4), chunks=(w, d, h, 1))
std_data[:, :, :, :] = np.transpose(std[:, :, :, :, 0], [2, 1, 0, 3])

sim_file.close()

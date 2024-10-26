import h5py
import numpy as np
import os

SIMULATION_NAME = '48_48_32_2000_0.71_0.01_0.3_1000.2'

data_dir = os.path.dirname(os.path.realpath(__file__))

sim_h5 = h5py.File(
    os.path.join(data_dir, '..', 'simulation', '3d', 'data', SIMULATION_NAME, 'sim.h5'), 
    'r')
sim_temp = sim_h5['temperature']
sim_vel = sim_h5['velocity']
h, d, w, N = sim_temp.shape

N_train = int(0.8*N)
N_test = N-N_train

tf_h5 = h5py.File(os.path.join(data_dir, f'{SIMULATION_NAME}.h5'), 'w')

train_data = tf_h5.create_dataset("train", (N_train, w, d, h, 4), chunks=(1, w, d, h, 1))
test_data = tf_h5.create_dataset("test", (N_test, w, d, h, 4), chunks=(1, w, d, h, 1))


train_data.attrs['N'] = N_train
test_data.attrs['N'] = N_test

import random
indices = list(range(N))
random.shuffle(indices)
train_indices = indices[:N_train]
train_indices = indices[-N_test:]

for i, sample in zip(range(N_train), train_indices):
    train_data[i, :, :, :, 0] = np.transpose(sim_temp[:, :, :, sample])
    train_data[i, :, :, :, 1:] = np.transpose(sim_vel[:, :, :, :, sample], [2, 1, 0, 3])
    
    if i%10==0: print(f'train set: {i}/{N_train}')
    
    
for i, sample in zip(range(N_test), train_indices):
    test_data[i, :, :, :, 0] = np.transpose(sim_temp[:, :, :, sample])
    test_data[i, :, :, :, 1:] = np.transpose(sim_vel[:, :, :, :, sample], [2, 1, 0, 3])
    
    if i%10==0: print(f'test set: {i}/{N_test}')
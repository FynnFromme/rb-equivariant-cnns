import h5py
import numpy as np
import os

SIMULATION_NAME = '96_96_64_10000.0_0.71_0.03_0.3_1000.2'

data_dir = os.path.dirname(os.path.realpath(__file__))

sim_h5 = h5py.File(
    os.path.join(data_dir, '..', 'simulation', '3d', 'data', SIMULATION_NAME, 'sim.h5'), 
    'r')
sim_temp = sim_h5['temperature']
sim_vel = sim_h5['velocity']
h, d, w, N = sim_temp.shape


tf_h5 = h5py.File(os.path.join(data_dir, f'{SIMULATION_NAME}.h5'), 'w')

tf_data = tf_h5.create_dataset("data", (N, w, d, h, 4), chunks=(1, w, d, h, 1))


for i in range(N):
    tf_data[i, :, :, :, 0] = np.transpose(sim_temp[:, :, :, 1])
    tf_data[i, :, :, :, 1:] = np.transpose(sim_vel[:, :, :, :, 1], [2, 1, 0, 3])
    
    if i%10==0: print(f'{i}/{N}')
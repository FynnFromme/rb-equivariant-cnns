"""Transforms raw simulation data into a shape that can be used for machine learning."""
import h5py
import numpy as np
import os
import random
import re
from sklearn.preprocessing import StandardScaler

SIMULATION_NAME = '48_48_32_10000_0.71_0.01_0.3_300'
FIRST_SNAPSHOT = 0 # to cut out the initial phase of the simulations

random.seed(SIMULATION_NAME)

DATA_DIR = os.path.dirname(os.path.realpath(__file__))


def get_simulation_data(simulation_name):
    """Data shape: (height, depth, width, channels, snapshots)"""
    simulation_dir = os.path.join(DATA_DIR, '..', 'simulation', '3d', 'data', simulation_name)
    sim_filenames = filter(lambda fn: re.match('sim[0-9]+.h5', fn), os.listdir(simulation_dir))
    sim_files = [h5py.File(os.path.join(simulation_dir, filename), 'r') for filename in sim_filenames]
    sim_data = [sim_file['data'] for sim_file in sim_files]
    return sim_data


def prepare_data(sim_data):
    """Can be used to remove/combine channels"""
    return sim_data





def standardize_data(sim_data):
    # make sure to load arrays sequentially to save memory
    scaler = StandardScaler()
    
    
    
    
    print(0, a)        
    mean2 = np.mean(sim_data, axis=(0,-1))[..., np.newaxis] # over all snapshots accross all initializations
    print(1, time.time()-a)
    mean = np.mean([np.mean(init, axis=-1, keepdims=True) for init in sim_data], axis=0)
    summed_squared_diffs = np.zeros_like(mean)
    for init in sim_data:
        summed_squared_diffs += np.sum((init-mean)**2, axis=-1, keepdims=True)
    num_snapshots = sim_data[0].shape[-1]*len(sim_data)
    std = np.sqrt(summed_squared_diffs/num_snapshots)
    print(2, time.time()-a)
    print(3, time.time()-a)
    std2 = np.std(sim_data, axis=(0,-1))[..., np.newaxis] # over all snapshots accross all initializations
    print(4, time.time()-a)
    
    print(np.allclose(mean, mean2))
    print(np.allclose(std, std2))
    print(5, time.time()-a)
    standardized_sim_data = [np.divide((sim_data_i - mean), std, out=np.zeros_like(sim_data_i), where=std!=0)
                for sim_data_i in sim_data]
    print(6, time.time()-a)
    return standardized_sim_data


sim_data = get_simulation_data(SIMULATION_NAME)
standardize_data(sim_data)

def split_random_initializations(sim_data, p_train, p_valid, p_test):
    num_inits = len(sim_data)
    num_train_inits = int(p_train*num_inits)
    num_valid_inits = int(p_valid*num_inits)
    num_test_inits = num_inits - (num_train_inits+num_valid_inits)
    
    indices = random.sample(range(num_inits), num_inits)
    train_data = np.concatenate([sim_data[i] for i in indices[:num_train_inits]], axis=-1)
    valid_data = np.concatenate([sim_data[i] for i in indices[num_train_inits:num_train_inits+num_valid_inits]], axis=-1)
    test_data = np.concatenate([sim_data[i] for i in indices[-num_test_inits:]], axis=-1)
    
    return train_data, valid_data, test_data
    
    
def create_h5_datasets(simulation_name, N_train, N_valid, N_test, dims):
    tf_h5 = h5py.File(os.path.join(DATA_DIR, f'{simulation_name}.h5'), 'w')
    
    train_data = tf_h5.create_dataset("train", (N_train, *dims, 4), chunks=(1, *dims, 1))
    valid_data = tf_h5.create_dataset("valid", (N_valid, *dims, 4), chunks=(1, *dims, 1))
    test_data = tf_h5.create_dataset("test", (N_test, *dims, 4), chunks=(1, *dims, 1))
    
    train_data.attrs['N'] = N_train
    valid_data.attrs['N'] = N_valid
    test_data.attrs['N'] = N_test

# h, d, w, _, N = sim_data.shape
# N = N-FIRST_SNAPSHOT

# new shape: N, w, d, h, channels(t,u1,u2,u3)
train_data[:, :, :, :, :] = np.transpose(sim_data[:, :, :, :, train_indices], [4, 2, 1, 0, 3])
valid_data[:, :, :, :, :] = np.transpose(sim_data[:, :, :, :, valid_indices], [4, 2, 1, 0, 3])
test_data[:, :, :, :, :] = np.transpose(sim_data[:, :, :, :, test_indices], [4, 2, 1, 0, 3])

# save mean and std tensor to be able to reverse standardization
mean_data = tf_h5.create_dataset("mean", (w, d, h, 4), chunks=(w, d, h, 4))
mean_data[:, :, :, :] = np.transpose(mean[:, :, :, :, 0], [2, 1, 0, 3])
std_data = tf_h5.create_dataset("std", (w, d, h, 4), chunks=(w, d, h, 4))
std_data[:, :, :, :] = np.transpose(std[:, :, :, :, 0], [2, 1, 0, 3])

sim_file.close()

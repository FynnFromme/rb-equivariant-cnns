"""Transforms raw simulation data into a shape that can be used for machine learning."""
import h5py
import numpy as np
import os
import random
import re
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')

# TODO: global first_snapshot or select first_snapshot per file manually???

def prepare_data(simulation_name, use_heat_flux=False, p_train=0.6, p_valid=0.2, p_test=0.2, first_snapshot=0):
    random.seed(simulation_name)
    
    channels = 1 if use_heat_flux else 4
    
    files = simulation_files(simulation_name)
    snapshots, w, d, h, sim_channels = data_shape(files)
    
    if use_heat_flux:
        temp_means = compute_temp_means(files) # used to calculate heat flux later
    else:
        temp_means = None
        
    train_files, valid_files, test_files = split_random_initializations(files, p_train, p_valid, p_test)
    
    snapshots_per_file = snapshots-first_snapshot
    N_train = snapshots_per_file*len(train_files)
    N_valid = snapshots_per_file*len(valid_files)
    N_test = snapshots_per_file*len(test_files)
    datafile = create_h5_datasets(simulation_name, N_train, N_valid, N_test, dims=(w, d, h), channels=channels)
    
    scaler = fit_scaler(files, first_snapshot, use_heat_flux, temp_means) # can be used to standardize data
    save_scaler(scaler, datafile, dims=(w, d, h), channels=channels) # to be able to unstandardize data later (e.g. for visualization)
    
    write_data(datafile, scaler, train_files, valid_files, test_files, first_snapshot, use_heat_flux, temp_means)
    
    datafile.close()
    

def compute_heatflux(sim_data, temp_mean):
    """Computes local convective heat flux"""
    temp = sim_data[:, :, :, :, [0]]
    vel_z = sim_data[:, :, :, :, [3]]
    return vel_z * (temp-temp_mean)


def compute_temp_means(files):
    temp_means = {}
    for fn in files:
        batch_means = []
        batch_sizes = [] # last batch might has different size
        for batch in read_data(fn, 64, first_snapshot=0, use_heat_flux=False):
            batch_means.append(batch[:, :, :, :, 0].mean())
            batch_sizes.append(batch.shape[0])
        temp_means[fn] = np.average(batch_means, weights=batch_sizes) # weight according to batch sizes
    return temp_means
        

def data_shape(files):
    """Shape of simulation data: (height, depth, width, channels, snapshots)
       Returned shape: (snapshots, width, depth, height, channels)"""
    with h5py.File(files[0], 'r') as hf:
        h, d, w, c, n = hf['data'].shape
        return (n, w, d, h, c)


def simulation_files(simulation_name):
    simulation_dir = os.path.join(DATA_DIR, '..', '..', 'simulation', '3d', 'data', simulation_name)
    sim_names = filter(lambda fn: re.match('sim[0-9]+', fn), os.listdir(simulation_dir))
    sim_paths = [os.path.join(simulation_dir, sim_name, 'sim.h5') for sim_name in sim_names]
    return sim_paths


def read_data(filename, batch_size, first_snapshot, use_heat_flux, temp_means=None):
    """Shape of simulation data: (height, depth, width, channels, snapshots)
       Shape of yielded data: (batch_size, width, depth, height, channels)"""
    with h5py.File(filename, 'r') as hf:
        snapshots = hf['data']
        N = snapshots.shape[-1]
        
        for i in range(first_snapshot, N, batch_size):
            batch = snapshots[:, :, :, :, i:i+batch_size].transpose([4, 2, 1, 0, 3])
            if use_heat_flux:
                batch = compute_heatflux(batch, temp_means[filename])
            yield batch


def fit_scaler(sim_filenames, first_snapshot, use_heat_flux, temp_means):
    print('fitting scaler for standardization...')
    
    scaler = StandardScaler()
    
    batch_size = 64
    # make sure to load arrays sequentially to save memory
    for i_f, filename in enumerate(sim_filenames):
        for batch in read_data(filename, batch_size, first_snapshot, use_heat_flux, temp_means):
            scaler.partial_fit(batch.reshape((batch.shape[0], -1)))
        print(f'-> fitted scaler to simulation file {i_f+1}/{len(sim_filenames)}')
        
    return scaler


def save_scaler(scaler, datafile, dims, channels):
    # save mean and std tensor to be able to reverse standardization
    mean_data = datafile.create_dataset("mean", (*dims, channels), chunks=(*dims, 1))
    std_data = datafile.create_dataset("std", (*dims, channels), chunks=(*dims, 1))    
    mean_data[:, :, :, :] = scaler.mean_.reshape((*dims, channels))
    std_data[:, :, :, :] = scaler.scale_.reshape((*dims, channels))


def split_random_initializations(sim_files, p_train, p_valid, p_test):
    print('splitting data...')
    num_inits = len(sim_files)
    num_train_inits = int(p_train*num_inits)
    num_valid_inits = int(p_valid*num_inits)
    num_test_inits = num_inits - (num_train_inits+num_valid_inits)
    
    indices = random.sample(range(num_inits), num_inits)
    train_files = [sim_files[i] for i in indices[:num_train_inits]]
    valid_files = [sim_files[i] for i in indices[num_train_inits:num_train_inits+num_valid_inits]]
    test_files = [sim_files[i] for i in indices[-num_test_inits:]]
    
    print(f'-> {num_train_inits} train initializations')
    for fn in train_files:
        print(f'\t->{fn}')
    print(f'-> {num_valid_inits} validation initializations')
    for fn in valid_files:
        print(f'\t->{fn}')
    print(f'-> {num_test_inits} test initializations')
    for fn in test_files:
        print(f'\t->{fn}')
    
    return train_files, valid_files, test_files
    
    
def create_h5_datasets(simulation_name, N_train, N_valid, N_test, dims, channels):
    datafile = h5py.File(os.path.join(DATA_DIR, f'{simulation_name}.h5'), 'w')
    
    train_data = datafile.create_dataset("train", (N_train, *dims, channels), chunks=(1, *dims, 1))
    valid_data = datafile.create_dataset("valid", (N_valid, *dims, channels), chunks=(1, *dims, 1))
    test_data = datafile.create_dataset("test", (N_test, *dims, channels), chunks=(1, *dims, 1))
    
    train_data.attrs['N'] = N_train
    valid_data.attrs['N'] = N_valid
    test_data.attrs['N'] = N_test
    
    return datafile


def write_data(datafile, scaler, train_files, valid_files, test_files, first_snapshot, use_heat_flux, temp_means):
    batch_size = 64
    
    print('writing files...')
    for dataset_name, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        dataset = datafile[dataset_name]
        next_index = 0
        for i, filename in enumerate(files):
            for batch in read_data(filename, batch_size, first_snapshot, use_heat_flux, temp_means):
                scaled_batch = scale_data(scaler, batch)
                
                batch_snaps = scaled_batch.shape[0]
                dataset[next_index:next_index+batch_snaps, :, :, :, :] = scaled_batch
                next_index = next_index+batch_snaps
                
            print(f'-> written {i+1}/{len(files)} {dataset_name} files')


def scale_data(scaler, data):
    shape = data.shape
    return scaler.transform(data.reshape((shape[0], -1))).reshape(shape)


if __name__ == '__main__':
    prepare_data('48_48_32_5000_0.7_0.01_0.3_300', use_heat_flux=False, 
                 p_train=0.6, p_valid=0.2, p_test=0.2, first_snapshot=200)
    
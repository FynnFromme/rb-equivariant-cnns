"""Transforms raw simulation data into a shape that can be used for machine learning."""
import h5py
import numpy as np
import os
import random
import re
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')

# TODO: global first_snapshot or select first_snapshot per file manually???

class DataPreparation:
    def __init__(self, 
                 out_shape='nhcwd', # tf: 'nwdhc'
                 transformation=lambda batch, precomputed: batch,
                 transform_precomputation=lambda files: defaultdict(lambda:None),
                 out_channels=4,
                 p_train=0.6, 
                 p_valid=0.2, 
                 p_test=0.2, 
                 first_snapshot=0,
                 step=1
                 ):
        
        out_shape = out_shape.lower()
        assert len(out_shape) == 5 and set(out_shape.lower()) == set('nwdhc')
        self.dim2index = {dim : out_shape.index(dim) for dim in 'nwdhc'}
        self.transpose2outshape = ['nwdhc'.index(dim) for dim in out_shape]
        self.transpose2outshape_without_n = ['wdhc'.index(dim) for dim in out_shape.replace('n','')]
        
        
        self.transformation = transformation
        self.transform_precomputation = transform_precomputation
        self.out_channels = out_channels
        
        self.p_train = p_train
        self.p_valid = p_valid
        self.p_test = p_test
        
        self.first_snapshot = first_snapshot
        self.step = step
        
    def prepare_data(self, simulation_name):
        random.seed(simulation_name)
    
        files = self._simulation_files(simulation_name)
        snapshots, *dims, sim_channels = self._data_shape(files)
        
        transform_args = self.transform_precomputation(files)
            
        train_files, valid_files, test_files = self._split_random_initializations(files)
        
        snapshots_per_file = (snapshots-self.first_snapshot) // self.step
        N_train = snapshots_per_file*len(train_files)
        N_valid = snapshots_per_file*len(valid_files)
        N_test = snapshots_per_file*len(test_files)
        datafile = self._create_h5_datasets(simulation_name, N_train, N_valid, N_test, dims)
        
        scaler = self._fit_scaler(files, transform_args) # can be used to standardize data
        self._save_scaler(scaler, datafile, dims) # to be able to unstandardize data later (e.g. for visualization)
        
        self._write_data(datafile, scaler, train_files, valid_files, test_files, transform_args)
        
        datafile.close()
        

    def _compute_heatflux(self, sim_data, temp_mean):
        """Computes local convective heat flux"""
        temp = sim_data[:, :, :, :, [0]]
        vel_z = sim_data[:, :, :, :, [3]]
        return vel_z * (temp-temp_mean)


    def _compute_temp_means(self, files):
        temp_means = {}
        for fn in files:
            batch_means = []
            batch_sizes = [] # last batch might has different size
            for batch in self._read_data(fn, 64, first_snapshot=0, use_heat_flux=False):
                batch_means.append(batch[:, :, :, :, 0].mean())
                batch_sizes.append(batch.shape[0])
            temp_means[fn] = np.average(batch_means, weights=batch_sizes) # weight according to batch sizes
        return temp_means
            

    def _data_shape(self, files):
        """Shape of simulation data: (height, depth, width, channels, snapshots)
        Returned shape: (snapshots, width, depth, height, channels)"""
        with h5py.File(files[0], 'r') as hf:
            h, d, w, c, n = hf['data'].shape
            
        return (n, w, d, h, c)


    def _simulation_files(self, simulation_name):
        simulation_dir = os.path.join(DATA_DIR, '..', '..', 'simulation', '3d', 'data', simulation_name)
        sim_names = filter(lambda fn: re.match('sim[0-9]+', fn), os.listdir(simulation_dir))
        sim_paths = [os.path.join(simulation_dir, sim_name, 'sim.h5') for sim_name in sim_names]
        return sim_paths


    def _read_data(self, filename, batch_size, transform_args):
        """Shape of simulation data: (height, depth, width, channels, snapshots)
        Shape of yielded data: (batch_size, width, depth, height, channels)"""
        with h5py.File(filename, 'r') as hf:
            snapshots = hf['data']
            N = snapshots.shape[-1]
            
            for i in range(self.first_snapshot, N, self.step*batch_size):
                batch = snapshots[:, :, :, :, i:i+self.step*batch_size:self.step].transpose([4, 2, 1, 0, 3])
                batch = self.transformation(batch, transform_args[filename])
                yield batch


    def _fit_scaler(self, sim_filenames, transform_args):
        print('fitting scaler for standardization...')
        
        scaler = StandardScaler()
        
        batch_size = 64
        # make sure to load arrays sequentially to save memory
        for i_f, filename in enumerate(sim_filenames):
            for batch in self._read_data(filename, batch_size, transform_args):
                batch = self.transformation(batch, transform_args)
                scaler.partial_fit(batch.reshape((batch.shape[0], -1)))
            print(f'-> fitted scaler to simulation file {i_f+1}/{len(sim_filenames)}')
            
        return scaler


    def _save_scaler(self, scaler, datafile, dims):
        # save mean and std tensor to be able to reverse standardization
        
        shape = [0]*5
        shape[self.dim2index['w']] = dims[0]
        shape[self.dim2index['d']] = dims[1]
        shape[self.dim2index['h']] = dims[2]
        shape[self.dim2index['c']] = self.out_channels
        shape.pop(self.dim2index['n'])
        
        chunk_shape = [1]*5
        chunk_shape[self.dim2index['w']] = dims[0]
        chunk_shape[self.dim2index['d']] = dims[1]
        chunk_shape[self.dim2index['h']] = dims[2]
        chunk_shape.pop(self.dim2index['n'])
        chunk_shape = tuple(chunk_shape)
        
        mean_data = datafile.create_dataset("mean", shape, chunks=chunk_shape)
        std_data = datafile.create_dataset("std", shape, chunks=chunk_shape)    
        mean_data[:, :, :, :] = scaler.mean_.reshape(*dims, self.out_channels).transpose(self.transpose2outshape_without_n)
        std_data[:, :, :, :] = scaler.scale_.reshape(*dims, self.out_channels).transpose(self.transpose2outshape_without_n)


    def _split_random_initializations(self, sim_files):
        print('splitting data...')
        num_inits = len(sim_files)
        num_train_inits = int(self.p_train*num_inits)
        num_valid_inits = int(self.p_valid*num_inits)
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
        
        
    def _create_h5_datasets(self, simulation_name, N_train, N_valid, N_test, dims):
        datafile = h5py.File(os.path.join(DATA_DIR, f'{simulation_name}.h5'), 'w')
        
        shape = [0]*5
        shape[self.dim2index['w']] = dims[0]
        shape[self.dim2index['d']] = dims[1]
        shape[self.dim2index['h']] = dims[2]
        shape[self.dim2index['c']] = self.out_channels
        
        chunk_shape = [1]*5
        chunk_shape[self.dim2index['w']] = dims[0]
        chunk_shape[self.dim2index['d']] = dims[1]
        chunk_shape[self.dim2index['h']] = dims[2]
        chunk_shape = tuple(chunk_shape)
        
        shape[self.dim2index['n']] = N_train
        train_data = datafile.create_dataset("train", shape, chunks=chunk_shape)
        shape[self.dim2index['n']] = N_valid
        valid_data = datafile.create_dataset("valid", shape, chunks=chunk_shape)
        shape[self.dim2index['n']] = N_test
        test_data = datafile.create_dataset("test", shape, chunks=chunk_shape)
        
        train_data.attrs['N'] = N_train
        valid_data.attrs['N'] = N_valid
        test_data.attrs['N'] = N_test
        
        return datafile


    def _write_data(self, datafile, scaler, train_files, valid_files, test_files, transform_args):
        batch_size = 64
        
        print('writing files...')
        for dataset_name, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
            dataset = datafile[dataset_name]
            next_index = 0
            for i, filename in enumerate(files):
                for batch in self._read_data(filename, batch_size, transform_args):
                    scaled_batch = self._scale_data(scaler, batch)
                    
                    batch_snaps = scaled_batch.shape[0]
                    
                    snap_slice = [slice(0, None)] * 5
                    snap_slice[self.dim2index['n']] = slice(next_index, next_index+batch_snaps)
                    dataset[*snap_slice] = scaled_batch.transpose(*self.transpose2outshape)
                    next_index = next_index+batch_snaps
                    
                print(f'-> written {i+1}/{len(files)} {dataset_name} files')


    def _scale_data(self, scaler, data):
        shape = data.shape
        return scaler.transform(data.reshape((shape[0], -1))).reshape(shape)


if __name__ == '__main__':
    DataPreparation(
                out_shape='nhcwd', # required shape for steerable pytorch implementation 
                p_train=0.6, p_valid=0.2, p_test=0.2, 
                first_snapshot=200,
                step=1 # every shapshot
                    ).prepare_data('48_48_32_2500_0.7_0.01_0.3_300')
    
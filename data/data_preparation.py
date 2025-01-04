"""
Transforms raw simulation data into a shape that can be used for machine learning and splits it
into train, validation and test datasets.
"""

import h5py
import numpy as np
import os
import random
import re
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import Callable, Any, Generator

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
SIM_DATA_DIR = os.path.join(DATA_DIR, '..', '..', 'simulation', '3d', 'data')

class DataPreparation:
    def __init__(self, 
                 out_shape: str ='nhcwd', # tf: 'nwdhc'
                 transformation: Callable[[np.ndarray, Any], np.ndarray] = lambda batch, precomputed: batch,
                 transform_precomputation: Callable[[list[str]], dict] = lambda files: defaultdict(lambda:None),
                 out_channels: int = 4,
                 p_train: float = 0.6, 
                 p_valid: float = 0.2, 
                 p_test: float = 0.2, 
                 first_snapshot: int = 0,
                 step: int = 1):
        """Initializes a data preparation object that transforms simulation data into
        a format that is suited for machine learning.

        Args:
            out_shape (str, optional): The order of the dimensions to output (n: samples,
                c: channels, w: width, d: depth, h: height). Defaults to 'nhcwd'.
            transformation (Callable, optional): Transformation applied to the simulation data.
                For instance transforming the rb data into heatflux via `DataPreparation.compute_heatflux`.
                The second argument is a precomputed value for the corresponding simulation (e.g. the precomputed
                temperature mean via `DataPreparation.compute_temp_means`)
            transform_precomputation (Callable, optional): A function that performs precomputation for every
                simulation file and stores the result in a dictionary. This for instance could be used to precompute
                the temperature mean for computing the heat flux.
            out_channels (int, optional): The number of output channels of the transformation. Defaults to 4.
            p_train (float, optional): The fraction of training data. Defaults to 0.6.
            p_valid (float, optional): The fraction of validation data. Defaults to 0.2.
            p_test (float, optional): The fraction of testing data. Defaults to 0.2.
            first_snapshot (int, optional): The first snapshot of the simulation data used for ML. Defaults to 0.
            step (int, optional): Can be used to only take every for example 2nd snapshot. Defaults to 1.
        """
        # precomputed mappings for the output dimension order
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
        
    def prepare_data(self, simulation_name: str):
        """Performs the actual data preparation:
        - Splitting into train, validation, test sets
        - Applying transformation
        - Standardize to zero mean and unit variance
        
        The results are stored in the datasets directory as a .h5 file. The file includes 3 datasets
        ("train", "valid", "test") that include the data. It also includes the datasets "mean" and 
        "variance" in order to be able to unstandardize the data later.

        Args:
            simulation_name (str): The name of the simulation.
        """
        random.seed(simulation_name)
    
        files = self._simulation_files(simulation_name)
        snapshots, *dims, sim_channels = self._data_shape(files)
        
        transform_args = self.transform_precomputation(files) # precomputed information required for transformation
            
        train_files, valid_files, test_files = self._split_random_initializations(files)
        
        snapshots_per_file = (snapshots-self.first_snapshot) // self.step
        N_train = snapshots_per_file*len(train_files)
        N_valid = snapshots_per_file*len(valid_files)
        N_test = snapshots_per_file*len(test_files)
        
        datafile = self._create_h5_datasets(simulation_name, N_train, N_valid, N_test, dims) # output h5 file
        
        scaler = self._fit_scaler(files, transform_args) # can be used to standardize data
        self._save_scaler(scaler, datafile, dims) # to be able to unstandardize data later (e.g. for visualization)
        
        self._write_data(datafile, scaler, train_files, valid_files, test_files, transform_args)
        
        datafile.close()
        

    def compute_heatflux(self, sim_data: np.ndarray, temp_mean: float) -> np.ndarray:
        """Computes local convective heat flux.

        Args:
            sim_data (np.ndarray): The simulation data of shape nwdhc.
            temp_mean (float): The temperature mean of the whole simulation.

        Returns:
            np.ndarray: The local convective heat flux of the data of shape nwdhc with one channel.
        """
        temp = sim_data[:, :, :, :, [0]]
        vel_z = sim_data[:, :, :, :, [3]]
        return vel_z * (temp-temp_mean)


    def compute_temp_means(self, files: list[str]) -> dict[str, float]:
        """Precomputes the temperature mean (accross space and time) of every simulation.

        Args:
            files (list[str]): A list of all simulation file names.

        Returns:
            dict[str, float]: The mean temperature of each simulation.
        """
        temp_means = {}
        for fn in files:
            batch_means = []
            batch_sizes = [] # last batch might has different size
            
            for batch in self._read_data(fn, batch_size=64, transform_args=None, transform=False):
                batch_means.append(batch[:, :, :, :, 0].mean())
                batch_sizes.append(batch.shape[0])
                
            temp_means[fn] = np.average(batch_means, weights=batch_sizes) # weight according to batch sizes
            
        return temp_means
            

    def _data_shape(self, files: list[str]) -> tuple:
        """Reads the shape of the simulation data, assuming the simulation data has the shape hdwcn.
        The shape is returned as a tuple in the order that corresponds to the shape returned by '_read_data': 
        (samples(n), width(w), depth(d), height(h), channels(c))

        Args:
            files (list[str]): The list of all simulation file names.

        Returns:
            tuple: (samples(n), width(w), depth(d), height(h), channels(c))
        """
        with h5py.File(files[0], 'r') as hf:
            h, d, w, c, n = hf['data'].shape
            
        return (n, w, d, h, c)


    def _simulation_files(self, simulation_name: str) -> list[str]:
        """Comptues the paths to all simulation files corresponding to the simulation name.

        Args:
            simulation_name (str): The name of the simulation.

        Returns:
            list[str]: The list of paths to all simulation files.
        """
        simulation_dir = os.path.join(SIM_DATA_DIR, simulation_name)
        sim_names = filter(lambda fn: re.match('sim[0-9]+', fn), os.listdir(simulation_dir))
        sim_paths = [os.path.join(simulation_dir, sim_name, 'sim.h5') for sim_name in sim_names]
        return sim_paths


    def _read_data(self, filename: str, batch_size: int, transform_args: Any, 
                   transform: bool = True) -> Generator[np.ndarray, None, None]:
        """Yields the transformed simulation data of the given simulation file batch-wise.
        Assuming the simulation data has the shape hdwcn, the transformed data is yielded in shape
        (samples(n), width(w), depth(d), height(h), channels(c))
        

        Args:
            filename (str): The filename of the simulation.
            batch_size (int): The size of each batch.
            transform_args (Any): Precomputed information required for transformation.
            transform (bool, optional): Whether to transform the data. Defaults to True.

        Yields:
            np.ndarray: A transformed batch of the simulation data.
        """
        with h5py.File(filename, 'r') as hf:
            snapshots = hf['data']
            N = snapshots.shape[-1]
            
            for i in range(self.first_snapshot, N, self.step*batch_size):
                batch = snapshots[:, :, :, :, i:i+self.step*batch_size:self.step]
                batch = batch.transpose([4, 2, 1, 0, 3]) # hdwcn -> nwdhc
                if transform:
                    batch = self.transformation(batch, transform_args)
                yield batch


    def _fit_scaler(self, sim_filenames: list[str], transform_args: dict[str, Any]) -> StandardScaler:
        """Fits a sklearn.StandardScaler to the whole simulation data.

        Args:
            sim_filenames (list[str]): The list of simulation files.
            transform_args (dict[str, Any]): The precomputed transformation arguments for every simulation.

        Returns:
            StandardScaler: The fitted StandardScaler.
        """
        print('fitting scaler for standardization...')
        
        scaler = StandardScaler()
        
        batch_size = 64
        # make sure to load arrays sequentially to save memory
        for i, filename in enumerate(sim_filenames):
            for batch in self._read_data(filename, batch_size, transform_args[filename]):
                scaler.partial_fit(batch.reshape((batch.shape[0], -1)))
                
            print(f'-> fitted scaler to simulation file {i+1}/{len(sim_filenames)}')
            
        return scaler


    def _save_scaler(self, scaler: StandardScaler, datafile: h5py.File, dims: tuple):
        """Saves mean and standard deviation of the simulation data to be able to reverse standardization.
        This is saved in the `datafile` as a "mean" and "std" dataset.

        Args:
            scaler (StandardScaler): The fitted scaler that contains the mean and standard deviation of the data.
            datafile (h5py.File): The output file.
            dims (tuple): The spatial dimensions of the simulation data.
        """
        
        # note: here we work on the output dimension order
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
        
        mean = scaler.mean_
        mean = mean.reshape(*dims, self.out_channels) # since scaler works on flattened features
        mean = mean.transpose(self.transpose2outshape_without_n) # use output dim order
        mean_data[:, :, :, :] = mean
        
        std = scaler.scale_
        std = std.reshape(*dims, self.out_channels) # since scaler works on flattened features
        std = std.transpose(self.transpose2outshape_without_n) # use output dim order
        std_data[:, :, :, :] = std


    def _split_random_initializations(self, sim_files: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Randomly splits the simulation files into train, validation and test sets.
        
        Note that simulations are splitted as a whole rather than snapshot wise to make the sets as 
        indepenedent as possible.

        Args:
            sim_files (list[str]): The simulation files.

        Returns:
            tuple[list[str], list[str], list[str]]: The train, validation and test sets.
        """
        
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
        print(*train_files, sep='\n')
        print(f'-> {num_valid_inits} validation initializations')
        print(*valid_files, sep='\n')
        print(f'-> {num_test_inits} test initializations')
        print(*test_files, sep='\n')
        
        return train_files, valid_files, test_files
        
        
    def _create_h5_datasets(self, simulation_name: str, N_train: int, N_valid: int, 
                            N_test: int, dims: tuple) -> h5py.File:
        """Creates a .h5 file for the output and initializes empty datasets for the train, validation
        and test sets.

        Args:
            simulation_name (str): The name of the simulation
            N_train (int): The number of training snapshots.
            N_valid (int): The number of validaiton snapshots.
            N_test (int): The number of testing snapshots.
            dims (tuple): The spatial dimensions of the simulation data.

        Returns:
            h5py.File: The output .h5 file.
        """
        datafile = h5py.File(os.path.join(DATA_DIR, f'{simulation_name}.h5'), 'w')
        
        # note: here we work on the output dimension order
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


    def _write_data(self, datafile: h5py.File, scaler: StandardScaler, train_files: list[str], 
                    valid_files: list[str], test_files: list[str], transform_args: dict[str, Any]):
        """Writes the transformed and standardized simulation data into the output .h5 file.

        Args:
            datafile (h5py.File): The output file.
            scaler (StandardScaler): The fitted scaler to standardize the data.
            train_files (list[str]): The list of all train simulation files.
            valid_files (list[str]): The list of all validation simulation files.
            test_files (list[str]): The list of all test simulation files.
            transform_args (dict[str, Any]): The precomputed arguments for each simulation file required for
                transformation of the simulation data. 
        """
        
        print('writing files...')
        
        batch_size = 64 # number of snapshots to load at once
        
        for dataset_name, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
            dataset = datafile[dataset_name] # output gets stored in this dataset
            
            next_index = 0
            for i, filename in enumerate(files):
                # read data sequentially to save memory
                for batch in self._read_data(filename, batch_size, transform_args[filename]):
                    scaled_batch = self._standardize_batch(scaler, batch)
                    
                    batch_snaps = scaled_batch.shape[0]
                    
                    # save data in output dimension order
                    snap_slice = [slice(0, None)] * 5
                    snap_slice[self.dim2index['n']] = slice(next_index, next_index+batch_snaps)
                    dataset[*snap_slice] = scaled_batch.transpose(*self.transpose2outshape)
                    
                    next_index = next_index+batch_snaps
                    
                print(f'-> written {i+1}/{len(files)} {dataset_name} files')


    def _standardize_batch(self, scaler: StandardScaler, batch: np.ndarray) -> np.ndarray:
        """Standardizes the batch using the fitted scaler.

        Args:
            scaler (StandardScaler): The fitted scaler to standardize the data.
            batch (np.ndarray): The transformed simulation data.

        Returns:
            np.ndarray: The standardized simulation data.
        """
        shape = batch.shape
        
        # scaler works on flattened features
        flat_batch = batch.reshape((shape[0], -1))
        scaled_flat_batch = scaler.transform(flat_batch)
        scaled_batch = scaled_flat_batch.reshape(shape)
        
        return scaled_batch


if __name__ == '__main__':
    prep = DataPreparation(
                out_shape='nhcwd', # required shape for steerable pytorch implementation 
                p_train=0.6, p_valid=0.2, p_test=0.2, 
                first_snapshot=200,
                step=1) # every shapshot
    
    prep.prepare_data('x48_y48_z32_Ra2500_Pr0.7_t0.01_snap0.125_dur300')
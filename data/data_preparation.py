"""
Transforms raw simulation data into a shape that can be used for machine learning and splits it
into standardized train, validation and test datasets.
"""

import h5py
import numpy as np
import os
import random
import re
from sklearn.preprocessing import StandardScaler
from typing import Generator
from argparse import ArgumentParser

class DataPreparation:
    def __init__(self,
                 p_train: float = 0.6, 
                 p_valid: float = 0.2, 
                 p_test: float = 0.2, 
                 first_snapshot: int = 0,
                 step: int = 1,
                 batch_size: int = 64):
        """Initializes a data preparation object that transforms simulation data into
        a format that is suited for machine learning.

        Args:
            p_train (float, optional): The fraction of training data. Defaults to 0.6.
            p_valid (float, optional): The fraction of validation data. Defaults to 0.2.
            p_test (float, optional): The fraction of testing data. Defaults to 0.2.
            first_snapshot (int, optional): The first snapshot of the simulation data used for ML. Defaults to 0.
            step (int, optional): Can be used to only take every for example 2nd snapshot. Defaults to 1.
            batch_size (int, optional): The number of samples loaded into memory at once. Defaults to 64.
        """
        
        self.p_train = p_train
        self.p_valid = p_valid
        self.p_test = p_test
        
        self.first_snapshot = first_snapshot
        self.step = step
        self.batch_size = batch_size
        
        
    def prepare_data(self, data_dir: str, sim_dir: str, simulation_name: str):
        """Performs the actual data preparation:
        - Splitting into train, validation, test sets
        - Standardize to zero mean and unit variance
        
        The results are stored in the datasets directory as a .h5 file. The file includes 3 datasets
        ("train", "valid", "test") that include the data. It also includes the datasets "mean" and 
        "variance" in order to be able to unstandardize the data later.
        
        The data has the shape [samples, width, depth, height, channels].

        Args:
            data_dir (str): The path of the directory to store the dataset in.
            sim_dir (str): The path of the directory, where the simulations are stored in.
            simulation_name (str): The name of the simulation.
        """
        random.seed(simulation_name)
    
        files = self._simulation_files(sim_dir, simulation_name)
        snapshots, *dims, sim_channels = self._data_shape(files)
            
        train_files, valid_files, test_files = self._split_random_initializations(files)
        
        snapshots_per_file = (snapshots-self.first_snapshot) // self.step
        N_train = snapshots_per_file*len(train_files)
        N_valid = snapshots_per_file*len(valid_files)
        N_test = snapshots_per_file*len(test_files)
        
        datafile = self._create_h5_datasets(data_dir, simulation_name, N_train, N_valid, 
                                            N_test, snapshots_per_file, dims) # output h5 file
        
        scaler = self._fit_scaler(files) # can be used to standardize data
        self._save_scaler(scaler, datafile, dims) # to be able to unstandardize data later (e.g. for visualization)
        
        self._write_data(datafile, scaler, train_files, valid_files, test_files)
        
        datafile.close()
            

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


    def _simulation_files(self, sim_dir: str, simulation_name: str) -> list[str]:
        """Comptues the paths to all simulation files corresponding to the simulation name.

        Args:
            sim_dir (str): The path of the directory, where the simulations are stored in.
            simulation_name (str): The name of the simulation.

        Returns:
            list[str]: The list of paths to all simulation files.
        """
        simulation_dir = os.path.join(sim_dir, simulation_name)
        sim_names = filter(lambda fn: re.match('sim[0-9]+', fn), os.listdir(simulation_dir))
        sim_paths = [os.path.join(simulation_dir, sim_name, 'sim.h5') for sim_name in sim_names]
        return sim_paths


    def _read_data(self, filename: str) -> Generator[np.ndarray, None, None]:
        """Yields the simulation data of the given simulation file batch-wise.
        Assuming the simulation data has the shape hdwcn, the data is yielded in shape
        (samples(n), width(w), depth(d), height(h), channels(c))
        

        Args:
            filename (str): The filename of the simulation.

        Yields:
            np.ndarray: A batch of the simulation data.
        """
        with h5py.File(filename, 'r') as hf:
            snapshots = hf['data']
            N = snapshots.shape[-1]
            
            for i in range(self.first_snapshot, N, self.step*self.batch_size):
                batch = snapshots[:, :, :, :, i:i+self.step*self.batch_size:self.step]
                batch = batch.transpose([4, 2, 1, 0, 3]) # hdwcn -> nwdhc
                yield batch


    def _fit_scaler(self, sim_filenames: list[str]) -> StandardScaler:
        """Fits a sklearn.StandardScaler to the whole simulation data.

        Args:
            sim_filenames (list[str]): The list of simulation files.

        Returns:
            StandardScaler: The fitted StandardScaler.
        """
        print('fitting scaler for standardization...')
        
        scaler = StandardScaler()
        
        # make sure to load arrays sequentially to save memory
        for i, filename in enumerate(sim_filenames):
            for batch in self._read_data(filename):
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
        shape = (*dims, 4)
        chunk_shape = (*dims, 1)
        
        mean_data = datafile.create_dataset("mean", shape, chunks=chunk_shape)
        std_data = datafile.create_dataset("std", shape, chunks=chunk_shape)  
        
        mean = scaler.mean_
        mean = mean.reshape(*dims, 4) # since scaler works on flattened features
        mean_data[:, :, :, :] = mean
        
        std = scaler.scale_
        std = std.reshape(*dims, 4) # since scaler works on flattened features
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
        
        
    def _create_h5_datasets(self, data_dir: str, simulation_name: str, N_train: int, N_valid: int, 
                            N_test: int, snapshots_per_file: int, dims: tuple) -> h5py.File:
        """Creates a .h5 file for the output and initializes empty datasets for the train, validation
        and test sets.

        Args:
            data_dir (str): The path of the directory to store the dataset in.
            simulation_name (str): The name of the simulation.
            N_train (int): The number of training snapshots.
            N_valid (int): The number of validaiton snapshots.
            N_test (int): The number of testing snapshots.
            snapshots_per_file (int): The number of snapshots per file.
            dims (tuple): The spatial dimensions of the simulation data.

        Returns:
            h5py.File: The output .h5 file.
        """
        datafile = h5py.File(os.path.join(data_dir, f'{simulation_name}.h5'), 'w')
        
        samples_shape = (*dims, 4)
        chunk_shape = (1, *dims, 1)
        
        train_data = datafile.create_dataset('train', (N_train, *samples_shape), chunks=chunk_shape)
        valid_data = datafile.create_dataset('valid', (N_valid, *samples_shape), chunks=chunk_shape)
        test_data = datafile.create_dataset('test', (N_test, *samples_shape), chunks=chunk_shape)
        
        train_data.attrs['N'] = N_train
        valid_data.attrs['N'] = N_valid
        test_data.attrs['N'] = N_test
        train_data.attrs['N_per_sim'] = snapshots_per_file
        valid_data.attrs['N_per_sim'] = snapshots_per_file
        test_data.attrs['N_per_sim'] = snapshots_per_file
        
        return datafile


    def _save_attrs(self, datafile, train_files, valid_files, test_files, snapshots_per_file):
        datafile['train'].attrs['N'] = snapshots_per_file*len(train_files)
        datafile['valid'].attrs['N'] = snapshots_per_file*len(valid_files)
        datafile['test'].attrs['N'] = snapshots_per_file*len(test_files)
        
        datafile['train'].attrs['files'] = train_files
        datafile['valid'].attrs['files'] = valid_files
        datafile['test'].attrs['files'] = test_files
        
        datafile['train'].attrs['N_per_sim'] = snapshots_per_file
        datafile['valid'].attrs['N_per_sim'] = snapshots_per_file
        datafile['test'].attrs['N_per_sim'] = snapshots_per_file


    def _write_data(self, datafile: h5py.File, scaler: StandardScaler, train_files: list[str], 
                    valid_files: list[str], test_files: list[str]):
        """Writes the standardized simulation data into the output .h5 file.

        Args:
            datafile (h5py.File): The output file.
            scaler (StandardScaler): The fitted scaler to standardize the data.
            train_files (list[str]): The list of all train simulation files.
            valid_files (list[str]): The list of all validation simulation files.
            test_files (list[str]): The list of all test simulation files.
        """
        
        print('writing files...')
        
        for dataset_name, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
            dataset = datafile[dataset_name] # output gets stored in this dataset
            
            next_index = 0
            for i, filename in enumerate(files):
                # read data sequentially to save memory
                for batch in self._read_data(filename):
                    scaled_batch = self._standardize_batch(scaler, batch)
                    
                    batch_snaps = scaled_batch.shape[0]
                    dataset[next_index:next_index+batch_snaps, ...] = scaled_batch
                    
                    next_index += batch_snaps
                    
                print(f'-> written {i+1}/{len(files)} {dataset_name} files')


    def _standardize_batch(self, scaler: StandardScaler, batch: np.ndarray) -> np.ndarray:
        """Standardizes the batch using the fitted scaler.

        Args:
            scaler (StandardScaler): The fitted scaler to standardize the data.
            batch (np.ndarray): The simulation data.

        Returns:
            np.ndarray: The standardized simulation data.
        """
        shape = batch.shape
        
        # scaler works on flattened features
        flat_batch = batch.reshape((shape[0], -1))
        scaled_flat_batch = scaler.transform(flat_batch)
        scaled_batch = scaled_flat_batch.reshape(shape)
        
        return scaled_batch


def parse_arguments():
    parser = ArgumentParser(description="""Transforms raw 3D Rayleigh-Bénard simulation data into a shape that can be 
                            used for machine learning and splits it into standardized train, validation 
                            and test datasets.
                            
                            The results are stored in the datasets directory as a .h5 file. The file includes 
                            3 datasets ("train", "valid", "test") that include the data. It also includes the 
                            datasets "mean" and "variance" in order to be able to unstandardize the data later.
        
                            Each dataset has the shape [samples, width, depth, height, channels].""")
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    default_data_dir = os.path.join(current_dir, 'datasets')
    default_sim_dir = os.path.join(current_dir, '..', 'simulation', '3d', 'data')
    
    parser.add_argument('sim_name', type=str, default=default_sim_dir,
                        help='The name of the simulation to transfer into a ML dataset.')
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help='The path of the directory to store the dataset in.')
    parser.add_argument('--sim_dir', type=str, default=default_sim_dir,
                        help='The path of the directory, where the simulations are stored in.\
                            The path should *not* include the simulation name.')
    parser.add_argument('--p_train', type=float, default=0.6,
                        help='The fraction of the simulations used for training.')
    parser.add_argument('--p_valid', type=float, default=0.2,
                        help='The fraction of the simulations used for validation.')
    parser.add_argument('--p_test', type=float, default=0.2,
                        help='The fraction of the simulations used for testing.')
    parser.add_argument('--first_snapshot', type=int, default=200,
                        help='The number of initial snapshots of each simulation to discard \
                            when creating the dataset.')
    parser.add_argument('--step', type=int, default=1,
                        help="If step>1, only every i'th snapshot is included in the dataset.")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(args.data_dir, exist_ok=True)
    prep = DataPreparation(
                p_train=args.p_train, p_valid=args.p_valid, p_test=args.p_test, 
                first_snapshot=args.first_snapshot,
                step=args.step)
    
    prep.prepare_data(args.data_dir, args.sim_dir, args.sim_name)
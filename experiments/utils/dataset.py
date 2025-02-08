import math
import numpy as np
import random

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

import h5py

from typing import Generator


class RBDataset(IterableDataset):
    def __init__(self, 
                 sim_file: str, 
                 dataset: str,
                 device: str = None,
                 samples: int = -1,
                 shuffle: bool = True,
                 slice_start: int = 0,
                 slice_end: int = -1):
        super().__init__()
        
        self.sim_file = sim_file
        self.dataset = dataset
        
        self.device = device
        self.shuffle = shuffle
        self.slice_start = slice_start
        self.slice_end = slice_end
        
        self.mean, self.std = standardization_params(sim_file)
        self.snapshot_shape = snapshot_shape(sim_file, dataset)
        self.snaps_per_sim = num_samples_per_sim(sim_file, dataset)
        self.snaps_in_dataset = num_samples(sim_file, dataset) # total number of samples in the dataset (regardless of current slice)
        self.num_simulations = self.snaps_in_dataset // self.snaps_per_sim
        
        slice_end = min(slice_end, self.snaps_in_dataset) if slice_end != -1 else self.snaps_in_dataset
        
        self.num_samples = min(slice_end-slice_start, samples) if samples != -1 else slice_end-slice_start


    def generator(self, start: int = 0, end: int = -1):
        end = min(self.num_samples, end) if end != -1 else self.num_samples
        start += self.slice_start
        end += self.slice_start
            
        with h5py.File(self.sim_file, 'r') as hf:
            snapshots = hf[self.dataset]
            
            indices = list(range(start, end))
            if self.shuffle:
                random.shuffle(indices)

            for i in indices:
                # (snap, snap) pairs for autoencoder
                snap = snapshots[i]
                snap = torch.Tensor(snap)
                
                if self.device is not None:
                    snap = snap.to(self.device)
                
                yield snap, snap
    
    
    def iterate_simulations(self) -> Generator['RBDataset', None, None]:
        """Yields datasets that are sliced to the data windows of the respective simulations."""
        sim_start_indices = range(0, self.snaps_in_dataset, self.snaps_per_sim)
        sim_end_indices = range(self.snaps_per_sim, self.snaps_in_dataset+1, self.snaps_per_sim)
        
        for start, end in zip(sim_start_indices, sim_end_indices):
            yield RBDataset(self.sim_file, self.dataset, self.device, samples=-1,
                             shuffle=self.shuffle, slice_start=start, slice_end=end)
    
                
    def standardize_batch(self, batch: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            raise Exception('Dataset does not contain standardization parameters')
        
        assert len(batch.shape) == 5
        
        assert tuple(batch.shape[-4:]) == self.snapshot_shape
        
        return (batch-self.mean) / self.std
    
    
    def de_standardize_batch(self, batch: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            raise Exception('Dataset does not contain standardization parameters')
        
        assert len(batch.shape) == 5
        
        assert tuple(batch.shape[-4:]) == self.snapshot_shape
        
        return batch * self.std + self.mean
                
    
    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)
                
                
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.num_samples
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.num_samples) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_samples)
        return self.generator(start=iter_start, end=iter_end)
    
    
class RBForecastDataset(RBDataset):
    def __init__(self, 
                 sim_file: str, 
                 dataset: str, 
                 warmup_seq_length: int,
                 forecast_seq_length: int,
                 forecast_warmup: bool = False,
                 device: str = None,
                 samples: int = -1,
                 shuffle: bool = True,
                 slice_start: int = 0,
                 slice_end: int = -1,
                 *args, **kwargs):
        super().__init__(sim_file=sim_file, dataset=dataset, device=device, samples=samples, shuffle=shuffle, 
                         slice_start=slice_start, slice_end=slice_end, *args, **kwargs)
        
        self.warmup_seq_length = warmup_seq_length
        self.forecast_seq_length = forecast_seq_length
        self.forecast_warmup = forecast_warmup
        
        # during forecasting the number of samples differs since there is no forecasting across simulation boundaries
        self.num_sampels = len(self.compute_forecasting_indices(start=0, end=slice_start+self.num_samples))


    def generator(self, start: int = 0, end: int = -1):
        end = min(self.num_samples, end) if end != -1 else self.num_samples
        start += self.slice_start
        end += self.slice_start
            
        with h5py.File(self.sim_file, 'r') as hf:
            snapshots = hf[self.dataset]
            
            indices = self.compute_forecasting_indices(start, end)
            if self.shuffle:
                random.shuffle(indices)

            for i in indices:
                # (warmup_seq, forecast_seq) pairs for training a forecasting model
                x = snapshots[i-self.warmup_seq_length:i]
                y = snapshots[i-self.warmup_seq_length+1:i+self.forecast_seq_length] if self.forecast_warmup else snapshots[i:i+self.forecast_seq_length]
                
                x = torch.Tensor(x)
                y = torch.Tensor(y)
                
                if self.device is not None:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                yield x, y
                
    
    def compute_forecasting_indices(self, start: int, end: int):
        # make sure that there are no forecasting samples across simulation boundaries
        sim_start_indices = range(0, self.snaps_in_dataset, self.snaps_per_sim)
        sim_end_indices = range(self.snaps_per_sim, self.snaps_in_dataset+1, self.snaps_per_sim)
        
        indices = []
        for sim_start, sim_end in zip(sim_start_indices, sim_end_indices):
            if sim_end <= start:
                continue # whole simulation is in front of selected range
            if sim_start >= end:
                break # whole simulation (and following ones) are after selected range
            sim_end = min(sim_end, end)
            sim_start = max(sim_start, start)
            
            sim_indices = list(range(sim_start+self.warmup_seq_length, sim_end-self.forecast_seq_length+1))
            indices.extend(sim_indices)
            
        return indices
    
    
    def iterate_simulations(self) -> Generator['RBForecastDataset', None, None]:
        """Yields datasets that are sliced to the data windows of the respective simulations."""
        sim_start_indices = range(0, self.snaps_in_dataset, self.snaps_per_sim)
        sim_end_indices = range(self.snaps_per_sim, self.snaps_in_dataset+1, self.snaps_per_sim)
        
        for start, end in zip(sim_start_indices, sim_end_indices):
            yield RBForecastDataset(self.sim_file, self.dataset, self.warmup_seq_length, self.forecast_seq_length, 
                                    self.forecast_warmup, self.device, samples=-1, shuffle=self.shuffle, 
                                    slice_start=start, slice_end=end)



def num_samples(sim_file: str, datasets: str | list[str]) -> int:
    single_dataset = type(datasets) is str
    if single_dataset:
        datasets = [datasets]
        
    with h5py.File(sim_file, 'r') as hf:
        samples = [hf[dataset].shape[0] for dataset in datasets]
        
    return samples[0] if single_dataset else samples


def num_samples_per_sim(sim_file: str, dataset: str) -> int:        
    with h5py.File(sim_file, 'r') as hf:
        samples_per_sim = hf[dataset].attrs['N_per_sim']
        
    return samples_per_sim


def standardization_params(sim_file: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(sim_file, 'r') as hf:
        if 'mean' in hf.keys() and 'std' in hf.keys():
            mean = np.array(hf['mean'])
            std = np.array(hf['std'])
        else:
            mean, std = None, None
    return mean, std


def snapshot_shape(sim_file: str, dataset: str) -> tuple:
    with h5py.File(sim_file, 'r') as hf:
        snapshot = hf[dataset][0]
    
    return snapshot.shape
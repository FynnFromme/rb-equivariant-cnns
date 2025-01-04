import math
import numpy as np
import random

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

import h5py

class DataReader(IterableDataset):
    def __init__(self, 
                 sim_file: str, 
                 dataset: str, 
                 device: str = None,
                 samples: int = -1,
                 shuffle: bool = True,
                 torch_format: bool = True):
        super().__init__()
        
        self.sim_file = sim_file
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.torch_format = torch_format
        
        self.mean, self.std = standardization_params(sim_file)
        self.snapshot_shape = snapshot_shape(sim_file, dataset)
        self.num_samples = num_samples(sim_file, dataset)
        if samples > 0: 
            self.num_samples = min(self.num_samples, samples)


    def generator(self, start: int = 0, end: int = None, batch_size: int = 1):
        if end == None:
            end = self.num_samples
            
        with h5py.File(self.sim_file, 'r') as hf:
            snapshots = hf[self.dataset]
            
            indices = list(range(start, end))
            if self.shuffle:
                random.shuffle(indices)

            for batch_indices in to_batches(indices, batch_size):
                batch = snapshots[batch_indices]
                
                if self.torch_format:
                    batch = self.sim_to_torch_format(batch)
                    
                batch = torch.Tensor(batch)
                
                if self.device is not None:
                    batch = batch.to(self.device)
                
                yield batch, batch # input equals output for autoencoder
                
    
    def sim_to_torch_format(self, data: Tensor) -> Tensor:
        assert len(data.shape) in (4, 5)
        
        h, c, w, d = self.snapshot_shape
        assert tuple(data.shape[-4:]) == (h, c, w, d)
        
        if len(data.shape) == 4: # single snapshot
            data = data.reshape(h*c, w, d)
        else: # batch
            data = data.reshape(-1, h*c, w, d)
            
        return data
    
    
    def torch_to_sim_format(self, data: Tensor) -> Tensor:
        assert len(data.shape) in (3, 4)
        
        h, c, w, d = self.snapshot_shape
        assert tuple(data.shape[-3:]) == (h*c, w, d)
        
        if len(data.shape) == 3: # single snapshot
            data = data.reshape(h, c, w, d)
        else: # batch
            data = data.reshape(-1, h, c, w, d)
            
        return data
    
                
    def standardize_batch(self, batch: Tensor) -> Tensor:
        assert len(batch.shape) == 5
        
        h, c, w, d = self.snapshot_shape
        assert tuple(batch.shape[-4:]) == (h, c, w, d)
        
        return (batch-self.mean) / self.std
    
    
    def de_standardize_batch(self, batch: Tensor) -> Tensor:
        assert len(batch.shape) == 5
        
        h, c, w, d = self.snapshot_shape
        assert tuple(batch.shape[-4:]) == (h, c, w, d)
        
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
        return self.generator(start=iter_start, end=iter_end, batch_size=1)
    
    
def num_samples(sim_file: str, datasets: str | list[str]) -> int:
    single_dataset = type(datasets) is str
    if single_dataset:
        datasets = [datasets]
        
    with h5py.File(sim_file, 'r') as hf:
        samples = [hf[dataset].shape[0] for dataset in datasets]
        
    return samples[0] if single_dataset else samples


def standardization_params(sim_file: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(sim_file, 'r') as hf:
        mean = np.array(hf['mean'])
        std = np.array(hf['std'])
    return mean, std


def snapshot_shape(sim_file: str, dataset: str) -> tuple:
    with h5py.File(sim_file, 'r') as hf:
        snapshot = hf[dataset][0]
    
    return snapshot.shape


def to_batches(iterable, size=1):
    if size == 1:
        for x in iterable:
            yield x
    else:
        l = len(iterable)
        for ndx in range(0, l, size):
            yield iterable[ndx:min(ndx + size, l)]
import torch
import numpy as np
import math

from utils import dataset
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import h5py
import os
from pathlib import Path


def compute_latent_dataset(autoencoder, latent_file, sim_file, device, batch_size):   
    N_train, N_valid, N_test = dataset.num_samples(sim_file, ['train', 'valid', 'test'])
    snapshots_per_file = dataset.num_samples_per_sim(sim_file, 'train')
    
    h5file = _create_h5_datasets(latent_file, autoencoder.latent_shape, N_train, N_valid, N_test, snapshots_per_file)

    # compute latent representations and store in file
    for ds_name in ['train', 'valid', 'test']:
        rb_dataset = dataset.RBDataset(sim_file, ds_name, device=device, shuffle=False)
        rb_loader = DataLoader(rb_dataset, batch_size=batch_size, num_workers=0, drop_last=False)
        
        latent_representations = _encode(autoencoder, rb_loader, rb_dataset.num_samples, batch_size)
        
        latent_dataset = h5file[ds_name]
        
        next_index = 0
        for latent in latent_representations:
            batch_snaps = len(latent)
            latent_dataset[next_index:next_index+batch_snaps, ...] = latent.cpu().detach().numpy()
                        
            next_index += batch_snaps
            
    h5file.close()
    
def _encode(autoencoder: torch.nn.Module, loader: DataLoader,
            samples: int = None, batch_size: int = None):
    batches = None
    if samples is not None and batch_size is not None: 
        batches = math.ceil(samples/batch_size)
        
    autoencoder.eval()
    with torch.no_grad():
        pbar = tqdm(loader, total=batches, desc='encoding rb', unit='batch')
        for inputs, outputs in pbar:
            latent = autoencoder.encode(inputs)
            yield latent


def _create_h5_datasets(file: str, latent_shape: tuple, N_train: int, N_valid: int, 
                        N_test: int, snapshots_per_file: int) -> h5py.File:
    directory = Path(file).parent
    directory.mkdir(parents=True, exist_ok=True)
    
    datafile = h5py.File(file, 'w')
    
    chunk_shape = (1, *latent_shape[:3], 1)
    
    train_data = datafile.create_dataset('train', (N_train, *latent_shape), chunks=chunk_shape)
    valid_data = datafile.create_dataset('valid', (N_valid, *latent_shape), chunks=chunk_shape)
    test_data = datafile.create_dataset('test', (N_test, *latent_shape), chunks=chunk_shape)
    
    train_data.attrs['N'] = N_train
    valid_data.attrs['N'] = N_valid
    test_data.attrs['N'] = N_test
    train_data.attrs['N_per_sim'] = snapshots_per_file
    valid_data.attrs['N_per_sim'] = snapshots_per_file
    test_data.attrs['N_per_sim'] = snapshots_per_file
    
    return datafile
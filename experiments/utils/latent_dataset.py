import torch
import math

from utils import dataset
from utils.dataset import RBDataset
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import h5py
from pathlib import Path

from utils.data_augmentation import DataAugmentation


def compute_latent_dataset(autoencoder, latent_file, sim_file, device, batch_size, 
                           data_augmentation: DataAugmentation = None):   
    N_train, N_valid, N_test = dataset.num_samples(sim_file, ['train', 'valid', 'test'])
    snapshots_per_file = dataset.num_samples_per_sim(sim_file, 'train')
    
    num_augmentations = len(data_augmentation.get_transformations()) if data_augmentation else 1
    
    h5file = _create_h5_datasets(latent_file, 
                                 autoencoder.latent_shape, 
                                 N_train, 
                                 N_valid, 
                                 N_test, 
                                 snapshots_per_file,
                                 num_augmentations)

    # compute latent representations and store in file
    for ds_name in ['train', 'valid', 'test']:
        rb_dataset = dataset.RBDataset(sim_file, ds_name, device=device, shuffle=False)
        
        ds_augmentation = data_augmentation if ds_name == 'train' else None
        latent_representations = _encode(autoencoder, rb_dataset, rb_dataset.num_samples, batch_size, ds_augmentation)
        
        latent_dataset = h5file[ds_name]
        
        next_index = 0
        for latent in latent_representations:
            batch_snaps = len(latent)
            latent_dataset[next_index:next_index+batch_snaps, ...] = latent.cpu().detach().numpy()
                        
            next_index += batch_snaps
            
    h5file.close()
    
def _encode(autoencoder: torch.nn.Module, rb_dataset: RBDataset, samples: int = None, 
            batch_size: int = None, data_augmentation: DataAugmentation = None):
    batches = None
    if samples is not None and batch_size is not None: 
        num_augmentations = len(data_augmentation.get_transformations()) if data_augmentation else 1
        batches = math.ceil((samples/rb_dataset.num_simulations)/batch_size) * num_augmentations * rb_dataset.num_simulations
        
    simulations = rb_dataset.iterate_simulations()
    augmentations = data_augmentation.get_transformations() if data_augmentation else [None]
    
    autoencoder.eval()
    with torch.no_grad():
        pbar = tqdm(total=batches, desc='encoding rb', unit='batch')
        
        for sim_dataset in simulations:
            sim_loader = DataLoader(sim_dataset, batch_size=batch_size, num_workers=0, drop_last=False)
            for augmentation in augmentations:
                for inputs, outputs in sim_loader:
                    if data_augmentation:
                        inputs = data_augmentation.transform(inputs, augmentation)
                    latent = autoencoder.encode(inputs)
                    pbar.update(1)
                    yield latent


def _create_h5_datasets(file: str, latent_shape: tuple, N_train: int, N_valid: int, 
                        N_test: int, snapshots_per_file: int, augmentations: int = 1) -> h5py.File:
    directory = Path(file).parent
    directory.mkdir(parents=True, exist_ok=True)
    
    datafile = h5py.File(file, 'w')
    
    chunk_shape = (1, *latent_shape[:3], 1)
    
    train_data = datafile.create_dataset('train', (N_train*augmentations, *latent_shape), chunks=chunk_shape)
    valid_data = datafile.create_dataset('valid', (N_valid, *latent_shape), chunks=chunk_shape)
    test_data = datafile.create_dataset('test', (N_test, *latent_shape), chunks=chunk_shape)
    
    train_data.attrs['augmentations'] = augmentations
    train_data.attrs['N'] = N_train*augmentations
    valid_data.attrs['N'] = N_valid
    test_data.attrs['N'] = N_test
    train_data.attrs['N_per_sim'] = snapshots_per_file
    valid_data.attrs['N_per_sim'] = snapshots_per_file
    test_data.attrs['N_per_sim'] = snapshots_per_file
    
    return datafile
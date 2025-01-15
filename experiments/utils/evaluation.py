import math
import torch
from torch.utils.data import DataLoader
from utils.data_reader import DataReader
from tqdm.auto import tqdm
from collections import Iterable

import os

def compute_loss(model: torch.nn.Module, test_loader: DataLoader, loss_fns, 
                      samples: int = None, batch_size: int = None):
    model.eval()

    multiple_losses = isinstance(loss_fns, Iterable)
    if not multiple_losses:
        loss_fns = [loss_fns]
        
    batches = None
    if samples is not None and batch_size is not None: 
        batches = math.ceil(samples/batch_size)

    running_losses = [0]*len(loss_fns)
    with torch.no_grad():
        for i, (inputs, outputs) in tqdm(enumerate(test_loader, 1), total=batches, desc='computing loss', unit='batch'):
            predictions = model(inputs)
            
            for loss_nr, loss_fn in enumerate(loss_fns):
                loss = loss_fn(outputs, predictions).item()
                running_losses[loss_nr] += loss
    
    avg_losses = [running_loss / i for running_loss in running_losses]

    return avg_losses if multiple_losses else avg_losses[0]


def compute_latent_sensitivity(model: torch.nn.Module, dataset: DataReader, samples: int = None, 
                               save_dir: str = None, filename: str = None):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)

    model.eval()
    aggregated_jacobian = None
    aggregated_jacobian_abs = None
    
    n = 0
    try:
        for batch, _ in tqdm(dataloader, total=samples):
            # Compute the Jacobian for the current batch
            jacobian_batch = torch.autograd.functional.jacobian(model.encode, batch, vectorize=True)
            
            # remove batch=1 dimension
            jacobian_batch = torch.squeeze(jacobian_batch)

            # Initialize the aggregator if not done yet
            if aggregated_jacobian is None:
                aggregated_jacobian = torch.zeros_like(jacobian_batch)
                aggregated_jacobian_abs = torch.zeros_like(jacobian_batch)

            # Sum the absolute Jacobian values
            aggregated_jacobian += jacobian_batch
            aggregated_jacobian_abs += jacobian_batch.abs()
            
            n += 1
            
            if samples is not None and n >= samples:
                break
    except KeyboardInterrupt as e:
        raise e
    finally:
        # Average over all test samples
        average_jacobian = aggregated_jacobian.cpu().detach().numpy() / n
        average_jacobian_abs = aggregated_jacobian_abs.cpu().detach().numpy() / n
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if filename is None:
                filename = 'latent_sensitivity'
            torch.save({'avg_sensitivity': average_jacobian,
                        'avg_abs_sensitivity': average_jacobian_abs,
                        'n': n},
                        os.path.join(save_dir, filename+'.pt'))

    return average_jacobian, average_jacobian_abs


def load_latent_sensitivity(save_dir: str, filename: str):
    sensitivity_data = torch.load(os.path.join(save_dir, filename+'.pt'))
    return sensitivity_data['avg_sensitivity'], sensitivity_data['avg_abs_sensitivity'], sensitivity_data['n']

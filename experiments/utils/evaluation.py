import math
import torch
from torch.utils.data import DataLoader
from .data_reader import DataReader

import os
from itertools import product
from tqdm.auto import tqdm
from collections.abc import Iterable

def compute_predictions(model: torch.nn.Module, loader: DataLoader,
                        samples: int = None, batch_size: int = None):
    batches = None
    if samples is not None and batch_size is not None: 
        batches = math.ceil(samples/batch_size)
        
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, total=batches, desc='computing loss', unit='batch')
        for inputs, outputs in pbar:
            predictions = model(inputs)
            yield inputs, predictions


def compute_loss(model: torch.nn.Module, test_loader: DataLoader, loss_fns, 
                 samples: int = None, batch_size: int = None):
    multiple_losses = isinstance(loss_fns, Iterable)
    if not multiple_losses: loss_fns = [loss_fns]
        
    predictions = compute_predictions(model, test_loader, samples, batch_size)
    
    running_losses = [0]*len(loss_fns)
    for batch_nr, (outputs, predictions) in enumerate(predictions, 1):
        for loss_nr, loss_fn in enumerate(loss_fns):
            loss = loss_fn(outputs, predictions).item()
            running_losses[loss_nr] += loss
    
    avg_losses = [running_loss / batch_nr for running_loss in running_losses]

    return avg_losses if multiple_losses else avg_losses[0]


def compute_loss_per_channel(model: torch.nn.Module, test_loader: DataLoader, loss_fns, 
                             axis: int, samples: int = None, batch_size: int = None):
    multiple_losses = isinstance(loss_fns, Iterable)
    if not multiple_losses: loss_fns = [loss_fns]
        
    predictions = compute_predictions(model, test_loader, samples, batch_size)
    
    running_losses = None
    for batch_nr, (outputs, predictions) in enumerate(predictions, 1):
        if running_losses is None:
            running_losses = [[0]*outputs.shape[axis] for _ in loss_fns]
            
        for loss_nr, loss_fn in enumerate(loss_fns):
            for channel in range(outputs.shape[axis]):
                loss = loss_fn(outputs.select(axis, channel), 
                               predictions.select(axis, channel)).item()
                running_losses[loss_nr][channel] += loss
    
    avg_losses = running_losses.copy()
    for loss_nr in range(len(loss_fns)):
        for channel in range(outputs.shape[axis]):
            avg_losses[loss_nr][channel] = running_losses[loss_nr][channel] / batch_nr

    return avg_losses if multiple_losses else avg_losses[0]


def compute_loss_per_channel(model: torch.nn.Module, test_loader: DataLoader, loss_fns, 
                             axes: int | list[int], samples: int = None, batch_size: int = None):
    if type(axes) is int: axes = [axes]
    
    multiple_losses = isinstance(loss_fns, Iterable)
    if not multiple_losses: loss_fns = [loss_fns]
        
    predictions = compute_predictions(model, test_loader, samples, batch_size)
    
    running_losses = None
    ax_indices = None
    ax_sizes = None
    for batch_nr, (outputs, predictions) in enumerate(predictions, 1):
        if ax_sizes is None:
            ax_sizes = [outputs.shape[ax] for ax in axes]
            running_losses = torch.zeros((len(loss_fns), *ax_sizes))
            
        for loss_nr, loss_fn in enumerate(loss_fns):
            for ax_index in product(*(range(d) for d in ax_sizes)):
                index_tuple = [slice(None)] * outputs.ndim  # Default to [:, :, ...]
                for d, i in zip(axes, ax_index):
                    index_tuple[d] = i  # Replace with specific indices
                    
                loss = loss_fn(outputs[tuple(index_tuple)], 
                               predictions[tuple(index_tuple)]).item()
                running_losses[(loss_nr, *ax_index)] += loss
    
    avg_losses = running_losses / batch_nr

    avg_losses = avg_losses.tolist()
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

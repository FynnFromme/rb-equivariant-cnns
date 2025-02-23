import math
import torch
from torch.utils.data import DataLoader
from .dataset import RBDataset

import os
from itertools import product
from tqdm.auto import tqdm
from collections.abc import Iterable

def compute_predictions(model: torch.nn.Module, loader: DataLoader,
                        samples: int = None, batch_size: int = None,
                        model_forward_kwargs: dict = {}):
    batches = None
    if samples is not None and batch_size is not None: 
        batches = math.ceil(samples/batch_size)
        
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, total=batches, desc='computing loss', unit='batch')
        for inputs, outputs in pbar:
            predictions = model(inputs, **model_forward_kwargs)
            yield outputs, predictions


def compute_loss(model: torch.nn.Module, test_loader: DataLoader, loss_fns, 
                 samples: int = None, batch_size: int = None,
                 model_forward_kwargs: dict = {}):
    multiple_losses = isinstance(loss_fns, Iterable)
    if not multiple_losses: loss_fns = [loss_fns]
        
    predictions = compute_predictions(model, test_loader, samples, batch_size, model_forward_kwargs)
    
    running_losses = [0]*len(loss_fns)
    n = 0
    for batch_nr, (outputs, predictions) in enumerate(predictions, 1):
        n += batch_size
        for loss_nr, loss_fn in enumerate(loss_fns):
            batch_size = outputs.size(0)
            loss = loss_fn(outputs, predictions).item()*batch_size
            running_losses[loss_nr] += loss

    avg_losses = [running_loss / n for running_loss in running_losses]

    return avg_losses if multiple_losses else avg_losses[0]


def compute_loss_per_channel(model: torch.nn.Module, test_loader: DataLoader, loss_fns, 
                             axes: int | list[int], samples: int = None, batch_size: int = None,
                             model_forward_kwargs: dict = {}):
    if type(axes) is int: axes = [axes]
    
    multiple_losses = isinstance(loss_fns, Iterable)
    if not multiple_losses: loss_fns = [loss_fns]
        
    predictions = compute_predictions(model, test_loader, samples, batch_size, model_forward_kwargs)
    
    running_losses = None
    ax_indices = None
    ax_sizes = None
    n = 0
    for batch_nr, (outputs, predictions) in enumerate(predictions, 1):
        batch_size = outputs.size(0)
        n += batch_size
        if ax_sizes is None:
            ax_sizes = [outputs.shape[ax] for ax in axes]
            running_losses = torch.zeros((len(loss_fns), *ax_sizes))
            
        for loss_nr, loss_fn in enumerate(loss_fns):
            for ax_index in product(*(range(d) for d in ax_sizes)):
                index_tuple = [slice(None)] * outputs.ndim  # Default to [:, :, ...]
                for d, i in zip(axes, ax_index):
                    index_tuple[d] = i  # Replace with specific indices
                    
                loss = loss_fn(outputs[tuple(index_tuple)], 
                               predictions[tuple(index_tuple)]).item() * batch_size
                running_losses[(loss_nr, *ax_index)] += loss
    
    avg_losses = running_losses / n

    avg_losses = avg_losses.tolist()
    return avg_losses if multiple_losses else avg_losses[0]


def compute_autoregressive_loss(model: torch.nn.Module, forecast_seq_length: int, test_loader: DataLoader, loss_fns, 
                                samples: int = None, batch_size: int = None, model_forward_kwargs: dict = {},
                                confidence_interval: float = 0.95):
    
    
    multiple_losses = isinstance(loss_fns, Iterable)
    if not multiple_losses: loss_fns = [loss_fns]
    
    # store old loss reductions since we need to modify them here
    old_loss_reductions = [loss_fn.reduction for loss_fn in loss_fns]
    old_parallel_ops = model.parallel_ops
        
    predictions = compute_predictions(model, test_loader, samples, batch_size, model_forward_kwargs)
    
    model.parallel_ops = False # compute sequence output sequentially due to memory constraint
    
    losses = torch.zeros(len(loss_fns), samples, forecast_seq_length)
    n = 0
    for batch_nr, (outputs, predictions) in enumerate(predictions, 1):
        batch_size = outputs.size(0)
        for loss_nr, loss_fn in enumerate(loss_fns):
            loss_fn.reduction = 'none'
            loss = loss_fn(outputs, predictions).mean(dim=(2,3,4,5)) # shape (batch_size, seq_length)
            losses[loss_nr, n:n+batch_size, :] = loss
        n += batch_size
    
    model.parallel_ops = old_parallel_ops
    for loss_fn, reduction in zip(loss_fns, old_loss_reductions):
        loss_fn.reduction = reduction
        
    avg_losses = losses.mean(dim=1)
    median_losses = torch.quantile(losses, 0.5, dim=1)
    lower_bounds = torch.quantile(losses, (1-confidence_interval)/2, dim=1)
    upper_bounds = torch.quantile(losses, 1-(1-confidence_interval)/2, dim=1)
    
    avg_losses = avg_losses.tolist()
    median_losses = median_losses.tolist()
    lower_bounds = lower_bounds.tolist()
    upper_bounds = upper_bounds.tolist()
    
    return (avg_losses if multiple_losses else avg_losses[0],
            median_losses if multiple_losses else median_losses[0],
            lower_bounds if multiple_losses else lower_bounds[0],
            upper_bounds if multiple_losses else upper_bounds[0],)



def compute_latent_sensitivity(model: torch.nn.Module, dataset: RBDataset, samples: int = None, 
                               save_dir: str = None, filename: str = None, parallel_channels: int = 1):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)

    model.eval()
    aggregated_jacobian = None
    aggregated_jacobian_abs = None
    
    n = 0
    try:
        for batch, _ in tqdm(dataloader, total=samples):
            # Compute the Jacobian for the current batch
            
            # Compute the Jacobian for the current batch
            channel_gradients = []
            num_channels = model.encode(batch).size(-1)
            for channel in range(1, num_channels, parallel_channels):
                end_channel = min(channel+parallel_channels, num_channels)
                jacobian_batch = torch.autograd.functional.jacobian(lambda x: model.encode(x)[..., channel:end_channel], 
                                                                    batch, vectorize=True)
                channel_gradients.append(jacobian_batch) # of shape (b,lw,ld,lh,lc,b,w,d,h,c) - l stands for latent
                    
            jacobian_batch = torch.cat(channel_gradients , dim=4) # concat along latent channel dimension
            
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
                        os.path.join(save_dir, filename+'.pt'),
                        pickle_protocol=4)

    return average_jacobian, average_jacobian_abs


def compute_latent_integrated_gradients(model: torch.nn.Module, dataset: RBDataset, samples: int = None, save_dir: str = None, filename: str = None, 
                                        steps: int = 50, num_channels: int = None, batch_size: int = 1):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)

    model.eval()
    aggregated_ig = None
    aggregated_abs_ig = None
    baseline = None
    
    n = 0
    try:
        for batch, _ in tqdm(dataloader, total=samples):
            if baseline is None:
                baseline = torch.zeros_like(batch)
            
            output_channels = model.encode(batch).size(-1)
            num_channels = min(output_channels, num_channels) if num_channels else output_channels
            
            ig = None
            for step in tqdm(range(1, steps+1), total=steps):
                input = baseline + step/steps * (batch-baseline)
                
                
                # Compute the Jacobian for the current batch
                channel_gradients = []
                for channel in range(1, num_channels, 16):
                    end_channel = min(channel+16, num_channels)
                    channel_gradient = torch.autograd.functional.jacobian(lambda x: model.encode(x)[..., channel:end_channel], input, vectorize=True)
                    channel_gradients.append(channel_gradient) # of shape (b,lw,ld,lh,lc,b,w,d,h,c) - l stands for latent
                    
                gradients = torch.cat(channel_gradients , dim=4) # concat along latent channel dimension
                
                # remove batch=1 dimensions
                gradients = torch.squeeze(gradients)
                
                if num_channels == 1:
                    gradients = gradients.unsqueeze(3) # add 1 latent channel dimension
                
                if ig is None:
                    ig = torch.zeros_like(gradients)
                    
                ig += gradients * 1/steps
                
            ig = (batch-baseline) * ig
            
            delta = torch.sum(ig) - torch.sum(model.encode(batch)[..., num_channels] - model.encode(baseline)[..., num_channels])
            print(delta.item())
                    

            # Initialize the aggregator if not done yet
            if aggregated_ig is None:
                aggregated_ig = torch.zeros_like(ig)
                aggregated_abs_ig = torch.zeros_like(ig)

            # Sum the absolute Jacobian values
            aggregated_ig += ig
            aggregated_abs_ig += ig.abs()
            
            n += batch.size(0)
            
            if samples is not None and n >= samples:
                break
    finally:
        # Average over all test samples
        avg_ig = aggregated_ig.cpu().detach().numpy() / n
        avg_abs_ig = aggregated_abs_ig.cpu().detach().numpy() / n
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if filename is None:
                filename = 'latent_sensitivity'
            torch.save({'avg_ig': avg_ig,
                        'avg_abs_ig': avg_abs_ig,
                        'n': n,
                        'steps': steps},
                        os.path.join(save_dir, filename+'.pt'),
                        pickle_protocol=4)

    return avg_ig, avg_abs_ig


def load_latent_sensitivity(save_dir: str, filename: str):
    sensitivity_data = torch.load(os.path.join(save_dir, filename+'.pt'))
    return sensitivity_data['avg_sensitivity'], sensitivity_data['avg_abs_sensitivity'], sensitivity_data['n']
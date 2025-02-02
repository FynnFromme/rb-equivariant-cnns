from .data_reader import DataReader, num_samples

import math
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from datetime import timedelta

from typing import Generator, Literal

import os
import json


def _predict_batches(model: torch.nn.Module, data_generator: Generator, data_reader: DataReader):
    """Calculates the models output of a batch of raw simulation data."""

    for (inputs, _) in data_generator:
        # predict
        model.eval()
        preds = model(inputs)
        
        inputs_stand = inputs.cpu().detach().numpy()
        preds_stand = preds.cpu().detach().numpy()
        
        # remove standardization
        inputs = data_reader.de_standardize_batch(inputs_stand)
        preds = data_reader.de_standardize_batch(preds_stand)
        
        yield inputs, preds, inputs_stand, preds_stand


def auto_encoder_animation(model: torch.nn.Module, 
                           axis: int,
                           anim_dir: str, 
                           anim_name: str, 
                           slice: int, 
                           fps: int, 
                           sim_file: str, 
                           device: str,
                           feature: Literal['t', 'u', 'v', 'w'],
                           dataset='test', 
                           batch_size=32,
                           frames=np.inf):
    channel = ['t', 'u', 'v', 'w'].index(feature)
    
    data_reader = DataReader(sim_file, dataset, device, shuffle=False)
    data_generator = data_reader(batch_size=batch_size)
    
    predicted_batches_gen = _predict_batches(model, data_generator, data_reader)
    batch_x, batch_y, batch_x_stand, batch_y_stand = next(predicted_batches_gen)
    in_batch_frame = 0

    # prepare plot
    if axis == 2:
        # is square when looking from above
        img_extent = [0, 2*np.pi, 0, 2*np.pi]
    else:
        img_extent = [0, 2*np.pi, 0, 2]
        
    fig = plt.figure()
    
    # initialize input snapshot
    ax = plt.subplot(1,3,1)
    orig_data = np.rot90(batch_x[0, :, :, :, channel].take(indices=slice, axis=axis))
    orig_im = plt.imshow(orig_data, cmap='rainbow', extent=img_extent)
    plt.axis('off')
    ax.set_title('input')

    # initialize output snapshot
    ax = plt.subplot(1,3,2)
    pred_data = np.rot90(batch_y[0, :, :, :, channel].take(indices=slice, axis=axis))
    pred_im = plt.imshow(pred_data, cmap='rainbow', extent=img_extent)
    plt.axis('off')
    ax.set_title('output')
    
    # initialize diff snapshot
    ax = plt.subplot(1,3,3)
    orig_data_stand = np.rot90(batch_x_stand[0, :, :, :, channel].take(indices=slice, axis=axis))
    pred_data_stand = np.rot90(batch_y_stand[0, :, :, :, channel].take(indices=slice, axis=axis))
    diff_im = plt.imshow(pred_data_stand-orig_data_stand, cmap='RdBu_r', extent=img_extent)
    plt.axis('off')
    ax.set_title('difference')
    diff_im.set_clim(vmin=-1, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(diff_im, cax=cax, orientation='vertical')

    last_frame = 0
    def frame_updater(frame):
        """Computes the next frame of the animation."""
        nonlocal batch_x, batch_y, batch_x_stand, batch_y_stand, in_batch_frame, last_frame
        if frame <= last_frame:
            # no new frame
            return orig_im, pred_im
        
        in_batch_frame +=1
        last_frame = frame

        if in_batch_frame >= batch_size:
            in_batch_frame = 0
            batch_x, batch_y, batch_x_stand, batch_y_stand = next(predicted_batches_gen)
        
        # update frames
        orig_data = np.rot90(batch_x[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        pred_data = np.rot90(batch_y[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        orig_data_stand = np.rot90(batch_x_stand[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        pred_data_stand = np.rot90(batch_y_stand[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        orig_im.set_array(orig_data)
        pred_im.set_array(pred_data)
        diff_im.set_array(pred_data_stand-orig_data_stand)
        
        # update color map limits
        vmin = min(np.min(orig_data), np.min(pred_data))
        vmax = max(np.max(orig_data), np.max(pred_data))
        orig_im.set_clim(vmin=vmin, vmax=vmax)
        pred_im.set_clim(vmin=vmin, vmax=vmax)
        
        return orig_im, pred_im
    
    frames = min(num_samples(sim_file, dataset), frames)
    anim = animation.FuncAnimation(fig, frame_updater, frames=frames, interval=1000/fps, blit=True)
    
    os.makedirs(anim_dir, exist_ok=True)
    anim.save(os.path.join(anim_dir, anim_name))
    plt.close()
    
    
def show_latent_patterns(sensitivity_data: np.ndarray, abs_sensitivity: bool, num: int, channel: int, 
                         slice: int, axis: int, cols: int = 5, seed: int = 0, contour: bool = True):
    sensitivity_data = sensitivity_data.reshape(-1, *sensitivity_data.shape[-4:]) # flatten latent indices
    random.seed(seed)
    latent_indices = random.sample(range(sensitivity_data.shape[0]), num)
    
    imgs_data = []
    for latent_indx in latent_indices:
        data = np.rot90(sensitivity_data[latent_indx, :, :, :, channel].take(indices=slice, axis=axis))
        imgs_data.append(data)
        
    max_abs_value = max(np.abs(arr).max() for arr in imgs_data)
    min_value = min(arr.min() for arr in imgs_data)
    max_value = max(arr.max() for arr in imgs_data)
    img_extent = [0, 2*np.pi, 0, 2*np.pi] if axis == 2 else [0, 2*np.pi, 0, 2] # depending on whether looking from above

    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(math.ceil(num/cols), cols),  # creates 2x2 grid of Axes
                    axes_pad=0.1,  # pad between Axes in inch.
                    cbar_mode='single')

    for ax, im_data in zip(grid, imgs_data):
        ax_show = ax.contourf if contour else ax.imshow
        if abs_sensitivity:
            im = ax_show(im_data, vmin=min_value, vmax=max_value, cmap='viridis', extent=img_extent)
        else:
            im = ax_show(im_data, vmin=-max_abs_value, vmax=max_abs_value, cmap='RdBu_r', extent=img_extent)
        ax.axis('off')
        
    grid.cbar_axes[0].colorbar(im)
    
    plt.show()
    
    
def plot_loss_history(model_dir: str, model_names: str | list, train_names: str | list, two_plots: bool = True, 
                      log_scale: bool = False, time_x: bool = False, remove_outliers: bool = True, smoothing: float = 0):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    fig = plt.figure(figsize=(20, 5) if two_plots else (8, 5))
    
    train_losses = []
    valid_losses = []
    for model_name, train_name in zip(model_names, train_names):
        log_file = os.path.join(model_dir, model_name, train_name, 'log.json')
        hyperparameters_file = os.path.join(model_dir, model_name, train_name, 'hyperparameters.json')
        with open(log_file, 'r') as f:
            log = json.load(f)
        with open(hyperparameters_file, 'r') as f:
            hyperparameters = json.load(f)
        train_loss_incl_dropout = hyperparameters['drop_rate'] > 0 and not hyperparameters['train_loss_in_eval']

        x = range(1, len(log['train_loss'])+1)
        if time_x:
            x = [sum(log['epoch_duration'][:i]) for i in x]
            
        if two_plots: 
            ax = plt.subplot(1, 2, 1)
        train_loss = log['train_loss']
        label = f'{model_name}/{train_name} - train'
        if train_loss_incl_dropout:
            label += ' (affected by dropout)'
        if smoothing > 0:
            smoothed_train_loss = _exponential_moving_average(train_loss, smoothing)
            smoothed_line, = plt.plot(x, smoothed_train_loss, label=label)
            train_line, = plt.plot(x, train_loss, alpha=0.15, color=smoothed_line.get_color())
        else:
            train_line, = plt.plot(x, train_loss, label=label)            
        train_losses.append(train_loss)
        
        if two_plots:
            ax = plt.subplot(1, 2, 2)
        valid_loss = log['valid_loss']
        color = train_line.get_color() if two_plots else _darken_color(train_line.get_color(), 0.8)
        if smoothing > 0:
            smoothed_valid_loss = _exponential_moving_average(valid_loss, smoothing)
            smoothed_line, = plt.plot(x, smoothed_valid_loss, label=f'{model_name}/{train_name} - valid', color=color)
            plt.plot(x, valid_loss, alpha=0.15, color=smoothed_line.get_color())
        else:
            plt.plot(x, valid_loss, label=f'{model_name}/{train_name} - valid', color=color) 
        valid_losses.append(log['valid_loss'])
        
    
    if two_plots:
        ax = plt.subplot(1, 2, 1)
        plt.title('Training Losses')
        plt.legend() 
        if log_scale: ax.set_yscale('log')
        if time_x: ax.xaxis.set_major_formatter(plt.FuncFormatter(_custom_duration_formatter))
        plt.grid(True)
        plt.xlabel('time (dd:hh:mm)' if time_x else 'epochs')
        plt.ylabel('loss (log-scale)' if log_scale else 'loss')
        if remove_outliers: plt.ylim(top=_max_upper_bound(train_losses))
        
        ax = plt.subplot(1, 2, 2)
        plt.title('Validation Losses')
        plt.legend() 
        if log_scale: ax.set_yscale('log')
        if time_x: ax.xaxis.set_major_formatter(plt.FuncFormatter(_custom_duration_formatter))
        plt.grid(True)
        plt.xlabel('time (dd:hh:mm)' if time_x else 'epochs')
        plt.ylabel('loss (log-scale)' if log_scale else 'loss')
        if remove_outliers: plt.ylim(top=_max_upper_bound(valid_losses))
    else:
        plt.title('Losses During Training')
        plt.legend() 
        if log_scale: plt.yscale('log')
        if time_x: plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(_custom_duration_formatter))
        plt.grid(True)
        plt.xlabel('time (dd:hh:mm)' if time_x else 'epochs')
        plt.ylabel('loss (log-scale)' if log_scale else 'loss')
        if remove_outliers: plt.ylim(top=max(_max_upper_bound(train_losses),
                                             _max_upper_bound(valid_losses)))
    
    if time_x: fig.autofmt_xdate() # auto format
    
    
def _custom_duration_formatter(secs, *args):
    td = timedelta(seconds=secs)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}"


def _lower_upper_bounds(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return lower_bound, upper_bound


def _max_upper_bound(x):
    return max(_lower_upper_bounds(l)[1] for l in x)


def _exponential_moving_average(data, alpha=0.7):
    ema = [data[0]]  # Start with the first data point
    for value in data[1:]:
        ema.append((1-alpha) * value + alpha * ema[-1])
    return ema


def _darken_color(color, amount=0.5):
    rgb = mcolors.to_rgb(color)  # Convert to RGB
    darker_rgb = tuple(max(0, c * amount) for c in rgb)  # Scale down
    return darker_rgb


def plot_performance(results_dir: str, model_names: str | list, train_names: str | list, metric: str):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    fig = plt.figure(figsize=(8, 5))
    
    performances = []
    performances_train = []
    for model_name, train_name in zip(model_names, train_names):
        results_file = os.path.join(results_dir, model_name, train_name, 'performance.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        performances.append(results[metric])
        performances_train.append(results[f'{metric}_train'])
        
    x_labels = [f'{model_name}/{train_name}' for model_name, train_name in zip(model_names, train_names)]
    plt.plot(x_labels, performances, label=f'test {metric.upper()}', marker='o', linestyle='--')
    plt.plot(x_labels, performances_train, label=f'train {metric.upper()}', marker='o', linestyle='--')
    
    fig.autofmt_xdate()
    plt.ylabel(metric.upper())
    plt.xlabel('model')
    plt.legend()
    plt.grid(axis='y')
    
    
def plot_performance_per_sim(results_dir: str, model_names: str | list, train_names: str | list, metric: str):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    fig = plt.figure(figsize=(8, 5))
    
    for model_name, train_name in zip(model_names, train_names):
        results_file = os.path.join(results_dir, model_name, train_name, 'performance_per_sim.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        x = range(1, len(results[metric])+1)
        plt.plot(x, results[metric], label=f'{model_name}/{train_name}', marker='o')
    
    plt.xticks(x)
    plt.ylabel(metric.upper())
    plt.xlabel('simulation')
    plt.legend()
    plt.grid(axis='y')
    
    
def plot_performance_per_channel(results_dir: str, model_names: str | list, train_names: str | list, metric: str):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    fig = plt.figure(figsize=(8, 5))
    
    for model_name, train_name in zip(model_names, train_names):
        results_file = os.path.join(results_dir, model_name, train_name, 'performance_per_channel.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        x = ['t', 'u', 'v', 'w']
        plt.plot(x, results[metric], label=f'{model_name}/{train_name}', marker='o')
    
    plt.xticks(x)
    plt.ylabel(metric.upper())
    plt.xlabel('channel')
    plt.legend()
    plt.grid(axis='y')
    
    
def plot_performance_per_height(results_dir: str, model_names: str | list, train_names: str | list, metric: str):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    fig = plt.figure(figsize=(8, 5))
    
    for model_name, train_name in zip(model_names, train_names):
        results_file = os.path.join(results_dir, model_name, train_name, 'performance_per_height.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        plt.plot(results[metric], label=f'{model_name}/{train_name}')
    
    plt.ylabel(metric.upper())
    plt.xlabel('height')
    plt.legend()
    plt.grid(axis='y')
    
    
def plot_performance_per_height_and_channel(results_dir: str, model_names: str | list, train_names: str | list, 
                                            metric: str, channel: int):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    fig = plt.figure(figsize=(8, 5))
    
    for model_name, train_name in zip(model_names, train_names):
        results_file = os.path.join(results_dir, model_name, train_name, 'performance_per_height.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        plt.plot(results[f'{metric}_per_channel'][channel], label=f'{model_name}/{train_name}')
    
    plt.ylabel(metric.upper())
    plt.xlabel('height')
    plt.legend()
    plt.grid(axis='y')
from .data_reader import DataReader, num_samples

import math
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

from typing import Generator, Literal

import os


def predict_batches(model: torch.nn.Module, data_generator: Generator, data_reader: DataReader):
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
    
    predicted_batches_gen = predict_batches(model, data_generator, data_reader)
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

    def frame_updater(frame):
        """Computes the next frame of the animation."""
        nonlocal batch_x, batch_y, batch_x_stand, batch_y_stand, in_batch_frame
        in_batch_frame += 1

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
                         slice: int, axis: int, cols: int = 5, seed: int = 0):
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
        if abs_sensitivity:
            im = ax.imshow(im_data, vmin=min_value, vmax=max_value, cmap='viridis', extent=img_extent)
        else:
            im = ax.imshow(im_data, vmin=-max_abs_value, vmax=max_abs_value, cmap='RdBu_r', extent=img_extent)
        plt.axis('off')
        
    cbar = grid.cbar_axes[0].colorbar(im)
    
    plt.show()
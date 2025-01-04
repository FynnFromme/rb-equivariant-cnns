from .data_reader import DataReader, num_samples

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import Generator, Literal

import os


def predict_batches(model: torch.nn.Module, data_generator: Generator, data_reader: DataReader):
    """Calculates the models output of a batch of raw simulation data."""

    for (inputs, _) in data_generator:
        # predict
        model.eval()
        preds = model(inputs)
        
        inputs = inputs.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        
        # remove standardization
        inputs = data_reader.de_standardize_batch(inputs)
        preds = data_reader.de_standardize_batch(preds)
        
        yield inputs, preds


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
    """Animates the output of a AE next to the input across a 2D slice.

    Args:
        axis: 0, 1, 2 for slices across the width, depth or height axis respectively
    """
    channel = ['t', 'u', 'v', 'w'].index(feature)
    
    data_reader = DataReader(sim_file, dataset, device, shuffle=False)
    data_generator = data_reader(batch_size=batch_size)
    
    predicted_batches_gen = predict_batches(model, data_generator, data_reader)
    batch_x, batch_y = next(predicted_batches_gen)
    in_batch_frame = 0

    # prepare plot
    if axis == 2:
        # is square when looking from above
        img_extent = [0, 2*np.pi, 0, 2*np.pi]
    else:
        img_extent = [0, 2*np.pi, 0, 2]
        
    fig = plt.figure()
    
    # initialize input snapshot
    ax = plt.subplot(1,2,1)
    orig_im = plt.imshow(np.rot90(batch_x[0, :, :, :, channel].take(indices=slice, axis=axis)), 
                         cmap='rainbow', extent=img_extent)
    plt.axis('off')
    ax.set_title('input')

    # initialize output snapshot
    ax = plt.subplot(1,2,2)
    pred_im = plt.imshow(np.rot90(batch_y[0, :, :, :, channel].take(indices=slice, axis=axis)), 
                         cmap='rainbow', extent=img_extent)
    plt.axis('off')
    ax.set_title('output')

    def frame_updater(frame):
        """Computes the next frame of the animation."""
        nonlocal batch_x, batch_y, in_batch_frame
        in_batch_frame += 1

        if in_batch_frame >= batch_size:
            in_batch_frame = 0
            batch_x, batch_y = next(predicted_batches_gen)
        
        # update frames
        orig_data = np.rot90(batch_x[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        pred_data = np.rot90(batch_y[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        orig_im.set_array(orig_data)
        pred_im.set_array(pred_data)
        
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
from .dataset import RBDataset, RBForecastDataset, num_samples, num_samples_per_sim

import math
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from datetime import timedelta

from collections import defaultdict

from typing import Literal
from torch.utils.data import DataLoader

import os
import json



def _predict_batches(model: torch.nn.Module, data_loader: DataLoader, dataset: RBDataset,
                     model_forward_kwargs: dict = {}):
    """Calculates the models output of a batch of raw simulation data."""
    
    model.eval()
    with torch.no_grad():
        for (inputs, ground_truth) in data_loader:
            # predict
            preds = model(inputs, **model_forward_kwargs)
            
            ground_truth_stand = ground_truth.cpu().detach().numpy()
            preds_stand = preds.cpu().detach().numpy()
            
            # remove standardization
            ground_truth = dataset.de_standardize_batch(ground_truth_stand)
            preds = dataset.de_standardize_batch(preds_stand)
            
            yield ground_truth, preds, ground_truth_stand, preds_stand


def rb_autoencoder_animation(model: torch.nn.Module, 
                             axis: int,
                             anim_dir: str, 
                             anim_name: str, 
                             slice: int, 
                             fps: int, 
                             sim_file: str, 
                             device: str,
                             feature: Literal['t', 'u', 'v', 'w'],
                             dataset_name='test', 
                             batch_size=32,
                             frames=np.inf):
    channel = ['t', 'u', 'v', 'w'].index(feature)
    
    dataset = RBDataset(sim_file, dataset_name, device, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)
    
    predicted_batches_gen = _predict_batches(model, data_loader, dataset)
    batch_y, batch_pred, batch_y_stand, batch_pred_stand = next(predicted_batches_gen)
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
    orig_data = np.rot90(batch_y[0, :, :, :, channel].take(indices=slice, axis=axis))
    orig_im = plt.imshow(orig_data, cmap='rainbow', extent=img_extent)
    plt.axis('off')
    ax.set_title('input')

    # initialize output snapshot
    ax = plt.subplot(1,3,2)
    pred_data = np.rot90(batch_pred[0, :, :, :, channel].take(indices=slice, axis=axis))
    pred_im = plt.imshow(pred_data, cmap='rainbow', extent=img_extent)
    plt.axis('off')
    ax.set_title('output')
    
    # initialize diff snapshot
    ax = plt.subplot(1,3,3)
    orig_data_stand = np.rot90(batch_y_stand[0, :, :, :, channel].take(indices=slice, axis=axis))
    pred_data_stand = np.rot90(batch_pred_stand[0, :, :, :, channel].take(indices=slice, axis=axis))
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
        nonlocal batch_y, batch_pred, batch_y_stand, batch_pred_stand, in_batch_frame, last_frame
        if frame <= last_frame:
            # no new frame
            return orig_im, pred_im
        
        in_batch_frame +=1
        last_frame = frame

        if in_batch_frame >= batch_size:
            in_batch_frame = 0
            batch_y, batch_pred, batch_y_stand, batch_pred_stand = next(predicted_batches_gen)
        
        # update frames
        orig_data = np.rot90(batch_y[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        pred_data = np.rot90(batch_pred[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        orig_data_stand = np.rot90(batch_y_stand[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        pred_data_stand = np.rot90(batch_pred_stand[in_batch_frame, :, :, :, channel].take(indices=slice, axis=axis))
        orig_im.set_array(orig_data)
        pred_im.set_array(pred_data)
        diff_im.set_array(pred_data_stand-orig_data_stand)
        
        plt.suptitle(f'Simulation {frame//dataset.snaps_per_sim + 1}')
        
        # update color map limits
        vmin = min(np.min(orig_data), np.min(pred_data))
        vmax = max(np.max(orig_data), np.max(pred_data))
        orig_im.set_clim(vmin=vmin, vmax=vmax)
        pred_im.set_clim(vmin=vmin, vmax=vmax)
        
        return orig_im, pred_im
    
    frames = min(num_samples(sim_file, dataset_name), frames)
    anim = animation.FuncAnimation(fig, frame_updater, frames=frames, interval=1000/fps, blit=True)
    
    os.makedirs(anim_dir, exist_ok=True)
    anim.save(os.path.join(anim_dir, anim_name), dpi=500)
    plt.close()
    
    
def rb_autoencoder_animation_3d(model: torch.nn.Module, 
                                anim_dir: str, 
                                anim_name: str,
                                fps: int, 
                                sim_file: str, 
                                device: str,
                                feature: Literal['t', 'u', 'v', 'w'],
                                dataset_name='test', 
                                rotate: bool = False,
                                angle_per_sec: int = 10,
                                contour_levels: int = 50,
                                batch_size=32,
                                frames=np.inf,):
    channel = ['t', 'u', 'v', 'w'].index(feature)
    
    dataset = RBDataset(sim_file, dataset_name, device, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)
    
    predicted_batches_gen = _predict_batches(model, data_loader, dataset)
    batch_y, batch_pred, batch_y_stand, batch_pred_stand = next(predicted_batches_gen)
    batch_diff = batch_y_stand-batch_pred_stand
    in_batch_frame = 0
        
    fig = plt.figure()
    
    ax1 = plt.subplot(1,3,1, projection='3d')
    plt.axis('off')
    ax2 = plt.subplot(1,3,2, projection='3d')
    plt.axis('off')
    ax3 = plt.subplot(1,3,3, projection='3d')
    plt.axis('off')
    
    orig_faces = plot_cube_faces(batch_y[0, :, :, :, channel], ax1, 
                                 contour_levels=contour_levels, show_back_faces=rotate)
    pred_faces = plot_cube_faces(batch_pred[0, :, :, :, channel], ax2, 
                                 contour_levels=contour_levels, show_back_faces=rotate)
    diff_faces = plot_cube_faces(batch_diff[0, :, :, :, channel], ax3, cmap='RdBu_r',
                                 contour_levels=contour_levels, show_back_faces=rotate)
    
    ax2.set_aspect('equal', adjustable='box')
    ax1.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    
    ax1.set_title('input')
    ax2.set_title('output')
    ax3.set_title('difference')
    
    elev = 15
    ax1.view_init(elev=elev)
    ax2.view_init(elev=elev)
    ax3.view_init(elev=elev)
    

    set_clims(diff_faces, vmin=-1, vmax=1)
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cbar = fig.colorbar(diff_faces[1], cax=cax, ticks=[0, 0.5, 1], extend="both", orientation='vertical')

    last_frame = 0
    def frame_updater(frame):
        """Computes the next frame of the animation."""
        nonlocal batch_y, batch_pred, batch_diff, batch_y_stand, batch_pred_stand, in_batch_frame, last_frame
        nonlocal orig_faces, pred_faces, diff_faces
        if frame <= last_frame:
            # no new frame
            return orig_faces + pred_faces + diff_faces
        
        in_batch_frame +=1
        last_frame = frame

        if in_batch_frame >= batch_size:
            in_batch_frame = 0
            batch_y, batch_pred, batch_y_stand, batch_pred_stand = next(predicted_batches_gen)
            batch_diff = batch_y_stand-batch_pred_stand
            
        # update frames
        reset_contours(orig_faces + pred_faces + diff_faces)
        orig_faces = plot_cube_faces(batch_y[in_batch_frame, :, :, :, channel], ax1, 
                                     contour_levels=contour_levels, show_back_faces=rotate)
        pred_faces = plot_cube_faces(batch_pred[in_batch_frame, :, :, :, channel], ax2, 
                                     contour_levels=contour_levels, show_back_faces=rotate)
        diff_faces = plot_cube_faces(batch_diff[in_batch_frame, :, :, :, channel], ax3, cmap='RdBu_r',
                                     contour_levels=contour_levels, show_back_faces=rotate)
        
        # cbar.remove()
        # cbar = fig.colorbar(diff_faces[1], ax=ax3, orientation='vertical')
        
        plt.suptitle(f'Simulation {frame//dataset.snaps_per_sim + 1}')
        
        # update color map limits
        orig_data = batch_y[in_batch_frame, :, :, :, channel]
        pred_data = batch_pred[in_batch_frame, :, :, :, channel]
        vmin = min(np.min(orig_data), np.min(pred_data))
        vmax = max(np.max(orig_data), np.max(pred_data))
        set_clims(orig_faces+pred_faces, vmin, vmax)
        set_clims(diff_faces, vmin=-1, vmax=1)
        
        if rotate:
            azim = -50 + angle_per_sec * frame/fps
            ax1.view_init(elev=elev, azim=azim)
            ax2.view_init(elev=elev, azim=azim)
            ax3.view_init(elev=elev, azim=azim)
        
        return orig_faces + pred_faces + diff_faces
    
    frames = min(num_samples(sim_file, dataset_name), frames)
    anim = animation.FuncAnimation(fig, frame_updater, frames=frames, interval=1000/fps, blit=True)
    
    os.makedirs(anim_dir, exist_ok=True)
    anim.save(os.path.join(anim_dir, anim_name), dpi=500)
    plt.close()
    
    
def plot_cube_faces(arr, ax, dims=(2*np.pi, 2*np.pi, 2), contour_levels=100, cmap='rainbow',
                    show_back_faces=False) -> list:
    x0 = np.linspace(0, dims[0], arr.shape[0])
    y0 = np.linspace(0, dims[1], arr.shape[1])
    z0 = np.linspace(0, dims[2], arr.shape[2])
    x, y, z = np.meshgrid(x0, y0, z0)
    
    xmax, ymax, zmax = max(x0), max(y0), max(z0)
    vmin, vmax = np.min(arr), np.max(arr)
    levels = np.linspace(vmin, vmax, contour_levels)

    z1 = ax.contourf(x[:, :, 0], y[:, :, 0], arr[:, :, -1].T,
                     zdir='z', offset=zmax, vmin=vmin, vmax=vmax,
                     levels=levels, cmap=cmap)
    if show_back_faces:
        z2 = ax.contourf(x[:, :, 0], y[:, :, 0], arr[:, :, 0].T,
                        zdir='z', offset=0, vmin=vmin, vmax=vmax,
                        levels=levels, cmap=cmap)
    y1 = ax.contourf(x[0, :, :].T, arr[:, 0, :].T, z[0, :, :].T,
                     zdir='y', offset=0, vmin=vmin, vmax=vmax,
                     levels=levels, cmap=cmap)
    if show_back_faces:
        y2 = ax.contourf(x[0, :, :].T, arr[:, -1, :].T, z[0, :, :].T,
                        zdir='y', offset=ymax, vmin=vmin, vmax=vmax,
                        levels=levels, cmap=cmap)
    x1 = ax.contourf(arr[-1, :, :].T, y[:, 0, :].T, z[:, 0, :].T,
                     zdir='x', offset=xmax, vmin=vmin, vmax=vmax,
                     levels=levels, cmap=cmap)
    if show_back_faces:
        x2 = ax.contourf(arr[0, :, :].T, y[:, 0, :].T, z[:, 0, :].T,
                        zdir='x', offset=0, vmin=vmin, vmax=vmax,
                        levels=levels, cmap=cmap)
    
    return [z1, z2, y1, y2, x1, x2] if show_back_faces else [z1, y1, x1]


def reset_contours(contours):
    for cont in contours:
        for c in cont.collections:
            c.remove()  # removes only the contours, leaves the rest intact
            
def set_clims(contours, vmin, vmax):
    for cont in contours:
        cont.set_clim(vmin, vmax)
    
    
def _forecast(model: torch.nn.Module, data_loader: DataLoader, dataset: RBDataset,
              steps: int):
    
    model.eval()
    with torch.no_grad():
        for (inputs, ground_truth_stand) in data_loader:
            # predict
            ground_truth_stand = ground_truth_stand.cpu().detach().numpy()
            ground_truth = dataset.de_standardize_batch(ground_truth_stand)
            
            for i, snap in enumerate(model.forward_gen(inputs, steps=steps)):
            
                snap_stand = snap.cpu().detach().numpy()
                
                # remove standardization
                snap = dataset.de_standardize_batch(snap_stand)
                
                yield ground_truth[:, i], snap, ground_truth_stand[:, i], snap_stand
    
    
def rb_forecaster_animation(model: torch.nn.Module, 
                            axis: int,
                            anim_dir: str, 
                            anim_name: str, 
                            slice: int, 
                            fps: int, 
                            sim_file: str, 
                            warmup_seq_length: int,
                            device: str,
                            feature: Literal['t', 'u', 'v', 'w'],
                            dataset_name: str = 'test',
                            num_forecasts: int = np.inf):
    channel = ['t', 'u', 'v', 'w'].index(feature)
    
    # dataset that includes one forecast sample per simulation
    snaps_per_sim = num_samples_per_sim(sim_file, dataset_name)
    forecast_steps = snaps_per_sim - warmup_seq_length
    dataset = RBForecastDataset(sim_file, dataset_name, device=device, shuffle=False, 
                                warmup_seq_length=warmup_seq_length, 
                                forecast_seq_length=forecast_steps)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)
    
    predicted_forecast_gen = _forecast(model, data_loader, dataset, steps=forecast_steps)
    y, forecast, y_stand, forecast_stand = next(predicted_forecast_gen)
    forecast_frame = 0

    # prepare plot
    if axis == 2:
        # is square when looking from above
        img_extent = [0, 2*np.pi, 0, 2*np.pi]
    else:
        img_extent = [0, 2*np.pi, 0, 2]
        
    fig = plt.figure()
    
    # initialize input snapshot
    ax = plt.subplot(1,3,1)
    orig_data = np.rot90(y[0, :, :, :, channel].take(indices=slice, axis=axis))
    orig_im = plt.imshow(orig_data, cmap='rainbow', extent=img_extent)
    plt.axis('off')
    ax.set_title('ground truth')

    # initialize output snapshot
    ax = plt.subplot(1,3,2)
    pred_data = np.rot90(forecast[0, :, :, :, channel].take(indices=slice, axis=axis))
    pred_im = plt.imshow(pred_data, cmap='rainbow', extent=img_extent)
    plt.axis('off')
    ax.set_title('forecast')
    
    # initialize diff snapshot
    ax = plt.subplot(1,3,3)
    diff_im = plt.imshow(pred_data-orig_data, cmap='RdBu_r', extent=img_extent)
    plt.axis('off')
    ax.set_title('difference')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(diff_im, cax=cax, orientation='vertical')
    
    # set color map limits
    vmin = min(np.min(y[..., channel]), np.min(forecast[..., channel]))
    vmax = max(np.max(y[..., channel]), np.max(forecast[..., channel]))
    orig_im.set_clim(vmin=vmin, vmax=vmax)
    pred_im.set_clim(vmin=vmin, vmax=vmax)
    diff_im.set_clim(vmin=-1, vmax=1)

    last_frame = 0
    sim = 1
    def frame_updater(frame):
        """Computes the next frame of the animation."""
        nonlocal y, forecast, y_stand, forecast_stand, forecast_frame, last_frame, sim
        if frame <= last_frame:
            # no new frame
            return orig_im, pred_im
        elif frame > last_frame+1:
            raise Exception('Frames are assumed to be handled in order')
        
        forecast_frame += 1
        last_frame = frame

        y, forecast, y_stand, forecast_stand = next(predicted_forecast_gen)
        
        if forecast_frame >= forecast_steps:
            forecast_frame = 0
            sim += 1
        
        # update frames
        orig_data = np.rot90(y[0, :, :, :, channel].take(indices=slice, axis=axis))
        pred_data = np.rot90(forecast[0, :, :, :, channel].take(indices=slice, axis=axis))
        orig_im.set_array(orig_data)
        pred_im.set_array(pred_data)
        diff_im.set_array(orig_data-pred_data)
        
        # update color map limits
        vmin = min(np.min(orig_data), np.min(orig_data))
        vmax = max(np.max(pred_data), np.max(pred_data))
        orig_im.set_clim(vmin=vmin, vmax=vmax)
        pred_im.set_clim(vmin=vmin, vmax=vmax)
        
        plt.suptitle(f'Simulation {sim} - {forecast_frame+1} autoregressive steps')
        
        return orig_im, pred_im
    
    frames = min(dataset.num_samples, num_forecasts) * dataset.forecast_seq_length
    anim = animation.FuncAnimation(fig, frame_updater, frames=frames, interval=1000/fps, blit=True)
    
    os.makedirs(anim_dir, exist_ok=True)
    anim.save(os.path.join(anim_dir, anim_name), dpi=500)
    plt.close()
    

def rb_forecaster_animation_3d(model: torch.nn.Module, 
                               anim_dir: str, 
                               anim_name: str,
                               fps: int, 
                               sim_file: str, 
                               warmup_seq_length: int,
                               device: str,
                               feature: Literal['t', 'u', 'v', 'w'],
                               dataset_name='test', 
                               rotate: bool = False,
                               angle_per_sec: int = 10,
                               contour_levels: int = 50,
                               num_forecasts: int = np.inf):
    channel = ['t', 'u', 'v', 'w'].index(feature)
    
    # dataset that includes one forecast sample per simulation
    snaps_per_sim = num_samples_per_sim(sim_file, dataset_name)
    forecast_steps = snaps_per_sim - warmup_seq_length
    dataset = RBForecastDataset(sim_file, dataset_name, device=device, shuffle=False, 
                                warmup_seq_length=warmup_seq_length, 
                                forecast_seq_length=forecast_steps)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)
    
    predicted_forecast_gen = _forecast(model, data_loader, dataset, steps=forecast_steps)
    y, forecast, y_stand, forecast_stand = next(predicted_forecast_gen)
    diff = y_stand-forecast_stand
    forecast_frame = 0
        
    fig = plt.figure()
    
    ax1 = plt.subplot(1,3,1, projection='3d')
    plt.axis('off')
    ax2 = plt.subplot(1,3,2, projection='3d')
    plt.axis('off')
    ax3 = plt.subplot(1,3,3, projection='3d')
    plt.axis('off')
    
    orig_faces = plot_cube_faces(y[0, :, :, :, channel], ax1, 
                                 contour_levels=contour_levels, show_back_faces=rotate)
    pred_faces = plot_cube_faces(forecast[0, :, :, :, channel], ax2, 
                                 contour_levels=contour_levels, show_back_faces=rotate)
    diff_faces = plot_cube_faces(diff[0, :, :, :, channel], ax3, cmap='RdBu_r',
                                 contour_levels=contour_levels, show_back_faces=rotate)
    
    ax2.set_aspect('equal', adjustable='box')
    ax1.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    
    ax1.set_title('input')
    ax2.set_title('output')
    ax3.set_title('difference')
    
    elev = 15
    ax1.view_init(elev=elev)
    ax2.view_init(elev=elev)
    ax3.view_init(elev=elev)
    

    set_clims(diff_faces, vmin=-1, vmax=1)
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cbar = fig.colorbar(diff_faces[1], cax=cax, ticks=[0, 0.5, 1], extend="both", orientation='vertical')

    last_frame = 0
    sim=1
    def frame_updater(frame):
        """Computes the next frame of the animation."""
        nonlocal y, forecast, y_stand, forecast_stand, forecast_frame, last_frame, sim
        nonlocal orig_faces, pred_faces, diff_faces
        if frame <= last_frame:
            # no new frame
            return orig_faces + pred_faces + diff_faces
        elif frame > last_frame+1:
            raise Exception('Frames are assumed to be handled in order')

        forecast_frame += 1
        last_frame = frame

        y, forecast, y_stand, forecast_stand = next(predicted_forecast_gen)
        diff = y_stand-forecast_stand
        
        if forecast_frame >= forecast_steps:
            forecast_frame = 0
            sim += 1
            
        # update frames
        reset_contours(orig_faces + pred_faces + diff_faces)
        orig_faces = plot_cube_faces(y[0, :, :, :, channel], ax1, 
                                     contour_levels=contour_levels, show_back_faces=rotate)
        pred_faces = plot_cube_faces(forecast[0, :, :, :, channel], ax2, 
                                     contour_levels=contour_levels, show_back_faces=rotate)
        diff_faces = plot_cube_faces(diff[0, :, :, :, channel], ax3, cmap='RdBu_r',
                                     contour_levels=contour_levels, show_back_faces=rotate)
        
        # cbar.remove()
        # cbar = fig.colorbar(diff_faces[1], ax=ax3, orientation='vertical')
        
        plt.suptitle(f'Simulation {sim} - {forecast_frame+1} autoregressive steps')
        
        # update color map limits
        orig_data = y[0, :, :, :, channel]
        pred_data = forecast[0, :, :, :, channel]
        vmin = min(np.min(orig_data), np.min(pred_data))
        vmax = max(np.max(orig_data), np.max(pred_data))
        set_clims(orig_faces+pred_faces, vmin, vmax)
        set_clims(diff_faces, vmin=-1, vmax=1)
        
        if rotate:
            azim = -50 + angle_per_sec * frame/fps
            ax1.view_init(elev=elev, azim=azim)
            ax2.view_init(elev=elev, azim=azim)
            ax3.view_init(elev=elev, azim=azim)
        
        return orig_faces + pred_faces + diff_faces
    
    frames = min(dataset.num_samples, num_forecasts) * dataset.forecast_seq_length
    anim = animation.FuncAnimation(fig, frame_updater, frames=frames, interval=1000/fps, blit=True)
    
    os.makedirs(anim_dir, exist_ok=True)
    anim.save(os.path.join(anim_dir, anim_name), dpi=500)
    plt.close()    

    
def show_latent_patterns(sensitivity_data: np.ndarray, abs_sensitivity: bool, num: int, channel: int, 
                         slice: int | None, axis: int, cols: int = 5, seed: int = 0, contour: bool = True,
                         unified_cbar: bool = True):
    """Set slice to None to have it automatically pick the slice of highest sensitivity for each pattern"""
    if slice is not None and unified_cbar is False:
        print('When using manual slicing, a unified colorbar is recommended. Otherwise small sensitivities will '\
            'be visualized as relevant patterns.')
    
    sensitivity_data = sensitivity_data.reshape(-1, *sensitivity_data.shape[-4:]) # flatten latent indices
    random.seed(seed)
    latent_indices = random.sample(range(sensitivity_data.shape[0]), num)
    
    imgs_data = []
    for latent_indx in latent_indices:
        if slice is None:
            remaining_axes = tuple({0,1,2}-{axis})
            highest_sensitivity_slice = np.argmax(abs(sensitivity_data[latent_indx, :, :, :, channel]).mean(axis=remaining_axes))
            data = np.rot90(sensitivity_data[latent_indx, :, :, :, channel].take(indices=highest_sensitivity_slice, axis=axis))
        else:
            data = np.rot90(sensitivity_data[latent_indx, :, :, :, channel].take(indices=slice, axis=axis))
        imgs_data.append(data)
        
    max_abs_values = [np.abs(arr).max() for arr in imgs_data]
    min_abs_values = [-max_abs for max_abs in max_abs_values]
    max_values = [arr.max() for arr in imgs_data]
    min_values = [arr.min() for arr in imgs_data]
    if unified_cbar:
        max_abs_value = max(max_abs_values)
        min_abs_value = -max_abs_value
        max_value = max(max_values)
        min_value = min(min_values)        
        
    img_extent = [0, 2*np.pi, 0, 2*np.pi] if axis == 2 else [0, 2*np.pi, 0, 2] # depending on whether looking from above

    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(math.ceil(num/cols), cols),  # creates 2x2 grid of Axes
                    axes_pad=0.1,  # pad between Axes in inch.
                    cbar_mode='single' if unified_cbar else None)

    for i, (ax, im_data) in enumerate(zip(grid, imgs_data)):
        ax_show = ax.contourf if contour else ax.imshow
        if abs_sensitivity:
            if not unified_cbar:
                min_value = min_values[i]
                max_value = max_values[i]
            im = ax_show(im_data, vmin=min_value, vmax=max_value, cmap='viridis', extent=img_extent)
        else:
            if not unified_cbar:
                min_abs_value = min_abs_values[i]
                max_abs_value = max_abs_values[i]
            im = ax_show(im_data, vmin=min_abs_value, vmax=max_abs_value, cmap='RdBu_r', extent=img_extent)
        ax.axis('off')
        
    if unified_cbar:
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
        train_loss_w_forced_decoding = hyperparameters.get('use_force_decoding', False) and not hyperparameters['train_loss_in_eval']

        x = range(1, len(log['train_loss'])+1)
        if time_x:
            x = [sum(log['epoch_duration'][:i]) for i in x]
            
        if two_plots: 
            ax = plt.subplot(1, 2, 1)
        train_loss = log['train_loss']
        label = f'{model_name}/{train_name} - train'
        if train_loss_incl_dropout and train_loss_w_forced_decoding:
            label += ' (affected by dropout & forced decoding)'
        elif train_loss_w_forced_decoding:
            label += ' (affected by forced decoding)'
        elif train_loss_incl_dropout:
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
        if remove_outliers: plt.ylim(top=min(_max_upper_bound(train_losses),
                                             plt.ylim()[1]), 
                                     bottom=max(plt.ylim()[0], 0))
        
        ax = plt.subplot(1, 2, 2)
        plt.title('Validation Losses')
        plt.legend() 
        if log_scale: ax.set_yscale('log')
        if time_x: ax.xaxis.set_major_formatter(plt.FuncFormatter(_custom_duration_formatter))
        plt.grid(True)
        plt.xlabel('time (dd:hh:mm)' if time_x else 'epochs')
        plt.ylabel('loss (log-scale)' if log_scale else 'loss')
        if remove_outliers: plt.ylim(top=min(_max_upper_bound(valid_losses),
                                             plt.ylim()[1]),
                                     bottom=max(plt.ylim()[0], 0))
    else:
        plt.title('Losses During Training')
        plt.legend() 
        if log_scale: plt.yscale('log')
        if time_x: plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(_custom_duration_formatter))
        plt.grid(True)
        plt.xlabel('time (dd:hh:mm)' if time_x else 'epochs')
        plt.ylabel('loss (log-scale)' if log_scale else 'loss')
        if remove_outliers: plt.ylim(top=min(max(_max_upper_bound(train_losses),
                                                _max_upper_bound(valid_losses)),
                                             plt.ylim()[1]),
                                     bottom=max(plt.ylim()[0], 0))
    
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


def plot_performance(results_dir: str, model_names: str | list, train_names: str | list, metric: str,
                     group_same_model: bool = False, show_train: bool = True):
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
        
    if group_same_model:
        model_performances = defaultdict(list)
        model_performances_train = defaultdict(list)
        for model_name, perf, perf_train in zip(model_names, performances, performances_train):
            model_performances[model_name].append(perf)
            model_performances_train[model_name].append(perf_train)
        mean = [np.mean(p) for m, p in model_performances.items()]
        mean_train = [np.mean(p) for m, p in model_performances_train.items()]
        std = [np.std(p) for m, p in model_performances.items()]
        std_train = [np.std(p) for m, p in model_performances_train.items()]
        
        x = model_performances.keys()
        
        plt.errorbar(x, mean, yerr=std, label=f'test {metric.upper()}', marker='o', linestyle='--', capsize=3)
        if show_train:
            plt.errorbar(x, mean_train, yerr=std_train, label=f'train {metric.upper()}', 
                         marker='o', linestyle='--', capsize=3)
    else:
        x_labels = [f'{model_name}/{train_name}' for model_name, train_name in zip(model_names, train_names)]
        plt.plot(x_labels, performances, label=f'test {metric.upper()}', marker='o', linestyle='--')
        if show_train:
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
    
    
def plot_performance_per_channel(results_dir: str, model_names: str | list, train_names: str | list, metric: str,
                                 show_train: bool = False):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    fig = plt.figure(figsize=(8, 5))
    
    for model_name, train_name in zip(model_names, train_names):
        results_file = os.path.join(results_dir, model_name, train_name, 'performance_per_channel.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        x = ['t', 'u', 'v', 'w']
        plt.plot(x, results[metric], label=f'{model_name}/{train_name}', marker='o')
        if show_train:
            plt.plot(x, results[f'{metric}_train'], label=f'{model_name}/{train_name} - train', marker='o')
            
    
    plt.xticks(x)
    plt.ylabel(metric.upper())
    plt.xlabel('channel')
    plt.legend()
    plt.grid(axis='y')
    
    
def plot_performance_per_height(results_dir: str, model_names: str | list, train_names: str | list, metric: str,
                                channel: int = None, show_train: bool = False):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    fig = plt.figure(figsize=(8, 5))
    
    for model_name, train_name in zip(model_names, train_names):
        results_file = os.path.join(results_dir, model_name, train_name, 'performance_per_height.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        if channel is not None:
            plt.plot(results[f'{metric}_per_channel'][channel], label=f'{model_name}/{train_name}')
        else:
            plt.plot(results[metric], label=f'{model_name}/{train_name}')
            
        if show_train:
            if channel is not None:
                plt.plot(results[f'{metric}_train_per_channel'][channel], label=f'{model_name}/{train_name}')
            else:
                plt.plot(results[f'{metric}_train'], label=f'{model_name}/{train_name}')
            
    
    plt.ylabel(metric.upper())
    plt.xlabel('height')
    plt.legend()
    plt.grid(axis='y')
    
    
def plot_performance_per_hp(trains_dir: str, results_dir: str, model_names: str | list, train_names: str | list, 
                            metric: str, hp: str, x_label: str = None, rounding: int = None, show_train: bool = False,
                            fill_error: bool = True):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    # group trainings of the same models
    trainings_of_model = defaultdict(list)
    for model, training in zip(model_names, train_names):
        trainings_of_model[model].append(training)
    
    fig = plt.figure(figsize=(8, 5))
    
    for model_name, trainings in trainings_of_model.items():
        performance_per_hp = defaultdict(lambda: ([], []))
        for train_name in trainings:
            results_file = os.path.join(results_dir, model_name, train_name, 'performance.json')
            with open(results_file, 'r') as f:
                results = json.load(f)
            performance = results[metric]
            performance_train = results[f'{metric}_train']
            
            hp_file = os.path.join(trains_dir, model_name, train_name, 'hyperparameters.json')
            with open(hp_file, 'r') as f:
                hps = json.load(f)
                
            hp_value = hps[hp]
            if rounding is not None:
                hp_value = round(hp_value, rounding)
                
            performance_per_hp[hp_value][0].append(performance)
            performance_per_hp[hp_value][1].append(performance_train)
        
        # sort by hp value and compute statistics
        hp_values, mean, std, mean_train, std_train = [], [], [], [], []
        error_bar = False
        for hp_value in sorted(performance_per_hp.keys()):
            hp_values.append(hp_value)
            
            performances, performances_train = performance_per_hp[hp_value]
            if len(performances) > 1:
                error_bar = True
            mean.append(np.mean(performances))
            std.append(np.std(performances))
            mean_train.append(np.mean(performances_train))
            std_train.append(np.std(performances_train))
        
        if error_bar:
            ebar = plt.errorbar(hp_values, mean, yerr=std, capsize=3, marker='o', label=model_name)
            if fill_error:
                mean, std = np.array(mean), np.array(std) # to be able to substract these
                plt.fill_between(hp_values, mean-std, mean+std, color=ebar.lines[0].get_color(), alpha=0.2)
            
            if show_train:
                ebar = plt.errorbar(hp_values, mean_train, yerr=std_train, capsize=3, marker='o', 
                                    label=f'{model_name} - train')
                if fill_error:
                    mean_train, std_train = np.array(mean_train), np.array(std_train) # to be able to substract these
                    plt.fill_between(hp_values, mean_train-std_train, mean_train+std_train, 
                                    color=ebar.lines[0].get_color(), alpha=0.2)
        else:
            plt.plot(hp_values, mean, marker='o', label=model_name)
            if show_train:
                plt.plot(hp_values, mean_train, marker='o', label=f'{model_name} - train')
                
    
    if x_label is None: x_label = hp
    plt.ylabel(metric.upper())
    plt.xlabel(x_label)
    plt.legend()
    plt.grid(axis='y')


def plot_autoregressive_performance(results_dir: str, model_names: str | list, train_names: str | list, metric: str,
                                    show_train: bool = False, show_bounds: bool = True, median: bool = True):
    if type(model_names) == str: model_names = [model_names]
    if type(train_names) == str: train_names = [train_names]
    
    fig = plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots()
    
    for model_name, train_name in zip(model_names, train_names):
        results_file = os.path.join(results_dir, model_name, train_name, 'autoregressive_performance.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        x = range(1, len(results[metric])+1)
        
        median_suffix = '_median' if median else ''
        test_line, = plt.plot(x, results[metric+median_suffix], label=f'{model_name}/{train_name} - test')
        if show_bounds:
            ax.fill_between(x, results[f'{metric}_lower'], results[f'{metric}_upper'], 
                            color=test_line.get_color(), alpha=0.2)
            
        if show_train:
            x = range(1, len(results[f'{metric}_train'])+1)
            train_line, = plt.plot(x, results[f'{metric}_train'+median_suffix], label=f'{model_name}/{train_name} - train')
            if show_bounds:
                ax.fill_between(x, results[f'{metric}_train_lower'], results[f'{metric}_train_upper'], 
                                color=train_line.get_color(), alpha=0.2)
        
        
    
    plt.ylabel(metric.upper())
    plt.xlabel('autoregressive steps')
    plt.legend()
    plt.grid(axis='y')
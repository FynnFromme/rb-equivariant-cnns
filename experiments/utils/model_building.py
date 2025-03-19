import torch
from torch import nn
from escnn import nn as enn
from escnn import gspaces

from experiments.models.autoencoders.steerable_conv_autoencoder import RBSteerableAutoencoder
from experiments.models.autoencoders.steerable_conv3d_autoencoder import RB3DSteerableAutoencoder
from experiments.models.autoencoders.conv_autoencoder import RBAutoencoder
from experiments.models.autoencoders.conv3d_autoencoder import RB3DAutoencoder

from experiments.models.forecasters.conv3d_forecaster import RB3DForecaster, RB3DLatentForecaster
from experiments.models.forecasters.steerable_conv_forecaster import RBSteerableForecaster, RBSteerableLatentForecaster

from utils.flipRot2dOnR3 import flipRot2dOnR3
from utils.training import load_trained_model

import os
import json


nonlinearity_mapping_ae = {
    "ELU": nn.ELU,
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
}

steerable_nonlinearity_mapping_ae = {
    "ELU": enn.ELU,
    "ReLU": enn.ReLU,
    "LeakyReLU": enn.LeakyReLU,
}

nonlinearity_mapping_fc = {
    "ELU": torch.nn.functional.elu,
    "ReLU": torch.relu,
    "tanh": torch.tanh
}

steerable_nonlinearity_mapping_fc = {
    "ELU": "elu",
    "ReLU": "relu",
    "tanh": "tanh"
}


def build_and_load_trained_model(models_dir: str, model_name: str, train_name: str, epoch: int = -1,
                                 override_hps: dict = {}):
    train_dir = os.path.join(models_dir, model_name, train_name)
    
    with open(os.path.join(train_dir, 'hyperparameters.json'), 'r') as f:
        hyperparameters = json.load(f)
        
    hyperparameters.update(override_hps)
        
    if model_name.startswith('AE') or models_dir.endswith('AE'):
        model = build_autoencoder(**hyperparameters)
    elif model_name.startswith('FC') or models_dir.endswith('FC'):
        model = build_forecaster(models_dir=models_dir, **hyperparameters)
    else:
        raise Exception('The model type (autoencoder or forecaster) must be specified in either the models_dir or model_name')
        
    # load weights into the model (changes the model itself)
    load_trained_model(model, models_dir, model_name, train_name, epoch=epoch, optimizer=None)
    return model


def build_autoencoder(conv_type: str, **hyperparameters):
    match conv_type:
        case 'SteerableConv':
            return build_RBSteerableAutoencoder(**hyperparameters)
        case 'SteerableConv3D':
            return build_RB3DSteerableAutoencoder(**hyperparameters)
        case 'Conv':
            return build_RBAutoencoder(**hyperparameters)
        case 'Conv3D':
            return build_RB3DAutoencoder(**hyperparameters)
        
        
def build_forecaster(conv_type: str, models_dir:str, **hyperparameters):
    match conv_type:
        case 'SteerableConv':
            return build_RBSteerableForecaster(models_dir=models_dir, **hyperparameters)
        case 'Conv3D':
            return build_RB3DForecaster(models_dir=models_dir, **hyperparameters)
    raise NotImplementedError()
            
            
def build_RBSteerableAutoencoder(simulation_name: str, rots: int, flips: int, encoder_channels: tuple, 
                                 latent_channels: int, v_kernel_size: int, h_kernel_size: int, 
                                 drop_rate: float, nonlinearity: str, **kwargs):
    horizontal_size = int(simulation_name.split('_')[0][1:])
    height = int(simulation_name.split('_')[2][1:])

    nonlinearity = steerable_nonlinearity_mapping_ae[nonlinearity]
    gspace = gspaces.flipRot2dOnR2 if flips else gspaces.rot2dOnR2
    G_size = 2*rots if flips else rots
    
    return RBSteerableAutoencoder(gspace=gspace(N=rots),
                                  rb_dims=(horizontal_size, horizontal_size, height),
                                  encoder_channels=encoder_channels,
                                  latent_channels=latent_channels//G_size,
                                  v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                  drop_rate=drop_rate, nonlinearity=nonlinearity, **kwargs)
            

def build_RB3DSteerableAutoencoder(simulation_name: str, rots: int, flips: int, encoder_channels: tuple, 
                                   latent_channels: int, v_kernel_size: int, h_kernel_size: int, 
                                   latent_v_kernel_size: int, latent_h_kernel_size: int,
                                   drop_rate: float, nonlinearity, **kwargs):
    horizontal_size = int(simulation_name.split('_')[0][1:])
    height = int(simulation_name.split('_')[2][1:])

    nonlinearity = steerable_nonlinearity_mapping_ae[nonlinearity]
    gspace = flipRot2dOnR3 if flips else gspaces.rot2dOnR3
    G_size = 2*rots if flips else rots

    return RB3DSteerableAutoencoder(gspace=gspace(n=rots),
                                    rb_dims=(horizontal_size, horizontal_size, height),
                                    encoder_channels=encoder_channels,
                                    latent_channels=latent_channels//G_size,
                                    kernel_size=v_kernel_size,
                                    latent_kernel_size=latent_v_kernel_size,
                                    drop_rate=drop_rate, 
                                    nonlinearity=nonlinearity, **kwargs)
    

def build_RBAutoencoder(simulation_name: str, encoder_channels: tuple, latent_channels: int, v_kernel_size: int, 
                        h_kernel_size: int, drop_rate: float, nonlinearity, **kwargs):
    horizontal_size = int(simulation_name.split('_')[0][1:])
    height = int(simulation_name.split('_')[2][1:])

    nonlinearity = nonlinearity_mapping_ae[nonlinearity]
    
    return RBAutoencoder(rb_dims=(horizontal_size, horizontal_size, height),
                        encoder_channels=encoder_channels,
                        latent_channels=latent_channels,
                        v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                        drop_rate=drop_rate, nonlinearity=nonlinearity, **kwargs)
    

def build_RB3DAutoencoder(simulation_name: str, encoder_channels: tuple, latent_channels: int, v_kernel_size: int, 
                          h_kernel_size: int, drop_rate: float, nonlinearity, **kwargs):
    horizontal_size = int(simulation_name.split('_')[0][1:])
    height = int(simulation_name.split('_')[2][1:])
    
    nonlinearity = nonlinearity_mapping_ae[nonlinearity]

    return RB3DAutoencoder(rb_dims=(horizontal_size, horizontal_size, height),
                           encoder_channels=encoder_channels,
                           latent_channels=latent_channels,
                           v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                           drop_rate=drop_rate, nonlinearity=nonlinearity, **kwargs)
    
    
def build_RB3DForecaster(models_dir: str, ae_model_name: str, ae_train_name: str, lstm_channels: int, v_kernel_size: int, 
                         h_kernel_size: int, drop_rate: float, recurrent_drop_rate: float, nonlinearity, 
                         include_autoencoder: bool, **kwargs):
    autoencoder = build_and_load_trained_model(models_dir, os.path.join('AE', ae_model_name), ae_train_name)
    *latent_dims, latent_channels = autoencoder.latent_shape      
    
    nonlinearity = nonlinearity_mapping_fc[nonlinearity]
    
    latent_forecaster = RB3DLatentForecaster(num_layers=len(lstm_channels), 
                                             latent_channels=latent_channels, 
                                             hidden_channels=lstm_channels, 
                                             latent_dims=latent_dims, 
                                             v_kernel_size=v_kernel_size, 
                                             h_kernel_size=h_kernel_size,
                                             nonlinearity=nonlinearity,
                                             drop_rate=drop_rate, 
                                             recurrent_drop_rate=recurrent_drop_rate, 
                                             **kwargs)
    if include_autoencoder:
        return RB3DForecaster(latent_forecaster=latent_forecaster,
                              autoencoder=autoencoder,
                              **kwargs)
    else:
        del autoencoder
        return latent_forecaster
    
    
def build_RBSteerableForecaster(models_dir: str, ae_model_name: str, ae_train_name: str, rots: int, flips: int, 
                                lstm_channels: int, v_kernel_size: int, h_kernel_size: int, drop_rate: float, 
                                recurrent_drop_rate: float, nonlinearity, include_autoencoder: bool, **kwargs):
    autoencoder = build_and_load_trained_model(models_dir, os.path.join('AE', ae_model_name), ae_train_name)
    G_size = 2*rots if flips else rots
    *latent_dims, latent_fieldsizes = autoencoder.latent_shape
    latent_channels = latent_fieldsizes//G_size

    nonlinearity = steerable_nonlinearity_mapping_fc[nonlinearity]
    gspace = gspaces.flipRot2dOnR2 if flips else gspaces.rot2dOnR2
    
    latent_forecaster = RBSteerableLatentForecaster(gspace=gspace(N=rots),
                                                    num_layers=len(lstm_channels), 
                                                    latent_channels=latent_channels, 
                                                    hidden_channels=lstm_channels, 
                                                    latent_dims=latent_dims, 
                                                    v_kernel_size=v_kernel_size, 
                                                    h_kernel_size=h_kernel_size,
                                                    nonlinearity=nonlinearity,
                                                    drop_rate=drop_rate, 
                                                    recurrent_drop_rate=recurrent_drop_rate,
                                                    **kwargs)
    if include_autoencoder:
        return RBSteerableForecaster(latent_forecaster=latent_forecaster,
                                           autoencoder=autoencoder,
                                           **kwargs)
    else:
        del autoencoder
        return latent_forecaster
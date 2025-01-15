from torch import nn
from escnn import nn as enn

from models.steerable_cnn_model import RBSteerableAutoencoder
from models.steerable_cnn3d_model import RB3DSteerableAutoencoder
from models.cnn_model import RBAutoencoder
from models.cnn3d_model import RB3DAutoencoder

from utils.flipRot2dOnR3 import flipRot2dOnR3

from escnn import gspaces


nonlinearity_mapping = {
    "ELU": nn.ELU,
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
}

steerable_nonlinearity_mapping = {
    "ELU": enn.ELU,
    "ReLU": enn.ReLU,
    "LeakyReLU": enn.LeakyReLU,
}


def build_model(model_type: str, **hyperparameters):
    match model_type:
        case 'steerableCNN':
            return build_RBSteerableAutoencoder(**hyperparameters)
        case 'steerable3DCNN':
            return build_RB3DSteerableAutoencoder(**hyperparameters)
        case 'CNN':
            return build_RBAutoencoder(**hyperparameters)
        case '3DCNN':
            return build_RB3DAutoencoder(**hyperparameters)
            
            
def build_RBSteerableAutoencoder(simulation_name: str, rots: int, flips: int, encoder_channels: tuple, 
                                 latent_channels: int, v_kernel_size: int, h_kernel_size: int, 
                                 drop_rate: float, nonlinearity: str, **kwargs):
    horizontal_size = int(simulation_name.split('_')[0][1:])
    height = int(simulation_name.split('_')[2][1:])

    nonlinearity = steerable_nonlinearity_mapping[nonlinearity]
    gspace = gspaces.flipRot2dOnR2 if flips else gspaces.rot2dOnR2
    G_size = 2*rots if flips else rots
    
    return RBSteerableAutoencoder(gspace=gspace(N=rots),
                                  rb_dims=(horizontal_size, horizontal_size, height),
                                  encoder_channels=encoder_channels,
                                  latent_channels=latent_channels//G_size,
                                  v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                                  drop_rate=drop_rate, nonlinearity=nonlinearity)
            

def build_RB3DSteerableAutoencoder(simulation_name: str, rots: int, flips: int, encoder_channels: tuple, 
                                   latent_channels: int, v_kernel_size: int, h_kernel_size: int, 
                                   drop_rate: float, nonlinearity, **kwargs):
    horizontal_size = int(simulation_name.split('_')[0][1:])
    height = int(simulation_name.split('_')[2][1:])

    nonlinearity = steerable_nonlinearity_mapping[nonlinearity]
    gspace = flipRot2dOnR3 if flips else gspaces.rot2dOnR3
    G_size = 2*rots if flips else rots

    return RB3DSteerableAutoencoder(gspace=gspace(n=rots),
                                    rb_dims=(horizontal_size, horizontal_size, height),
                                    encoder_channels=encoder_channels,
                                    latent_channels=latent_channels//G_size,
                                    kernel_size=v_kernel_size,
                                    drop_rate=drop_rate, nonlinearity=nonlinearity)
    

def build_RBAutoencoder(simulation_name: str, encoder_channels: tuple, latent_channels: int, v_kernel_size: int, 
                        h_kernel_size: int, drop_rate: float, nonlinearity, **kwargs):
    horizontal_size = int(simulation_name.split('_')[0][1:])
    height = int(simulation_name.split('_')[2][1:])

    nonlinearity = nonlinearity_mapping[nonlinearity]
    
    return RBAutoencoder(rb_dims=(horizontal_size, horizontal_size, height),
                        encoder_channels=encoder_channels,
                        latent_channels=latent_channels,
                        v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                        drop_rate=drop_rate, nonlinearity=nonlinearity)
    

def build_RB3DAutoencoder(simulation_name: str, encoder_channels: tuple, latent_channels: int, v_kernel_size: int, 
                          h_kernel_size: int, drop_rate: float, nonlinearity, **kwargs):
    horizontal_size = int(simulation_name.split('_')[0][1:])
    height = int(simulation_name.split('_')[2][1:])
    
    nonlinearity = nonlinearity_mapping[nonlinearity]

    return RB3DAutoencoder(rb_dims=(horizontal_size, horizontal_size, height),
                           encoder_channels=encoder_channels,
                           latent_channels=latent_channels,
                           v_kernel_size=v_kernel_size, h_kernel_size=h_kernel_size,
                           drop_rate=drop_rate, nonlinearity=nonlinearity)
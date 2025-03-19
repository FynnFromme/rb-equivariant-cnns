import os
import sys
import json

EXPERIMENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, '..'))

from experiments.utils import dataset
from torch.utils.data import DataLoader
from utils.data_augmentation import DataAugmentation
from utils import training
from experiments.utils.model_building import build_autoencoder

from escnn import gspaces
import torch, numpy as np, random

from argparse import ArgumentParser
from utils.argument_parsing import str2bool


TRAINED_MODELS_DIR = os.path.join(EXPERIMENT_DIR, 'trained_models')
DATA_DIR = os.path.join(EXPERIMENT_DIR, '..', 'data')


########################
# Parsing arguments
########################
parser = ArgumentParser()

parser.add_argument('conv_type', type=str, choices=['Conv3D', 'Conv', 'SteerableConv3D', 'SteerableConv'],
                    help='The type of convolutions used for the autoencoder.')
parser.add_argument('train_name', type=str,
                    help='The name of the trained model.')
parser.add_argument('epochs', type=int,
                    help='The number of epochs to train the model.')
parser.add_argument('-start_epoch', type=int, default=-1,
                    help='When restarting training, start_epoch specifies the saved epoch used as a \
                        starting point. For `-1`, the latest saved epoch will be used. Set to `0` to not \
                        load a previous state. Defaults to `-1`.')
parser.add_argument('-only_save_best', type=str2bool, default=True,
                    help='When set to `True`, previously saved epochs will be deleted once a better \
                        validation accuracy is achieved. Defaults to `True`.')
parser.add_argument('-train_loss_in_eval', type=str2bool, default=False,
                    help='When set to `True`, the training loss will be computed another time in eval mode \
                        and therefore mitigating the effect of e.g. Dropout. Note that this slows down training. \
                        Defaults to `False`.')

# data parameters
parser.add_argument('-simulation_name', type=str, default='x48_y48_z32_Ra2500_Pr0.7_t0.01_snap0.125_dur300',
                    help='The name of the dataset used to train the model. It must be located in "data/datasets". \
                        Defaults to "x48_y48_z32_Ra2500_Pr0.7_t0.01_snap0.125_dur300".')
parser.add_argument('-n_train', type=int, default=-1,
                    help='The number of samples used for training. Set to `-1` to use all available samples. \
                        Defaults to `-1`.')
parser.add_argument('-n_valid', type=int, default=-1,
                    help='The number of samples used for validation. Set to `-1` to use all available samples. \
                        Defaults to `-1`.')
parser.add_argument('-batch_size', type=int, default=64,
                    help='The batch size used during training. Defaults to `64`.')

# model hyperparameters
parser.add_argument('-flips', type=bool, default=True,
                    help='Whether to be equivariant to flips (only relevant for SteerableConv). Defaults to `True`.')
parser.add_argument('-rots', type=int, default=4,
                    help='The number of rotations to be equivariant to (only relevant for SteerableConv). \
                        Defaults to `4`.')
parser.add_argument('-v_kernel_size', type=int, default=5,
                    help='The vertical kernel size. Defaults to `5`.')
parser.add_argument('-h_kernel_size', type=int, default=5,
                    help='The horizontal kernel size. Defaults to `5`.')
parser.add_argument('-latent_v_kernel_size', type=int, default=3,
                    help='The vertical kernel size used for the convolution in latent space. Defaults to `5`.')
parser.add_argument('-latent_h_kernel_size', type=int, default=3,
                    help='The horizontal kernel size used for the convolution in latent space. Defaults to `5`.')
parser.add_argument('-drop_rate', type=float, default=0.2,
                    help='The drop rate used in between autoencoder layers. Defaults to `0.2`.')
parser.add_argument('-nonlinearity', type=str, default='ELU', choices=['ELU', 'ReLU', 'LeakyReLU'],
                    help='The nonlinearity used in the autoencoder. Defaults to "ELU".')
parser.add_argument('-encoder_channels', nargs='+', type=int, default=None,
                    help='The channels of the encoder. Each entry results in a corresponding layer. \
                        The decoder uses the channels in reversed order.')
parser.add_argument('-v_shares', nargs='+', type=int, default=None,
                    help='The number of neighboring output-heights sharing the same kernel for each encoder \
                        layer (and latent convolution!). Therefore, len(v_shares)=len(encoder_channels)+1. \
                        The same is used in reversed order for the decoder. Defaults to a 1-tuple.')
parser.add_argument('-pool_layers', nargs='+', type=str2bool, default=None,
                    help='A boolean tuple specifying the encoder layer to pool afterwards. \
                        The same is used in reversed order for upsampling in the decoder. \
                        Defaults to pooling/upsampling after each layer.')
parser.add_argument('-latent_channels', type=int, default=32,
                    help='The number of channels of the encoded latent representation. \
                        Defaults to `32`.')
parser.add_argument('-weight_decay', type=float, default=0,
                    help='The L2-regularization parameter. Defaults to `0`.')

# training hyperparameters
parser.add_argument('-lr', type=float, default=1e-3,
                    help='The learning rate. Defaults to `1e-3`.')
parser.add_argument('-use_lr_scheduler', type=str2bool, default=True,
                    help='Whether to use the `ReduceLROnPlateau` learning rate scheduler. Defaults to `True`.')
parser.add_argument('-lr_decay', type=float, default=0.5,
                    help='The factor applied to the learning rate when the validation loss stagnates. \
                        Defaults to `0.5`.')
parser.add_argument('-lr_decay_patience', type=int, default=5,
                    help='The number of epochs without an improvement in validation performance until \
                        the learning rate decays. Defaults to `5`.')
parser.add_argument('-early_stopping', type=int, default=20,
                    help='The number of epochs without an improvement in validation performance until \
                        training is stopped. Defaults to `20`.')
parser.add_argument('-early_stopping_threshold', type=float, default=1e-5,
                    help='The threshold which must be surpassed in order to count as an improvement in \
                        validation performance in the context of early stopping. Defaults to `1e-5`.')
parser.add_argument('-seed', type=int, default=0,
                    help='The seed used for initializing the model. Defaults to `0`.')

args = parser.parse_args()


########################
# Seed and GPU
########################
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
        
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    DEVICE = torch.cuda.current_device()
    print('Current device:', torch.cuda.get_device_name(DEVICE))
else:
    print('Failed to find GPU. Will use CPU.')
    DEVICE = 'cpu'
    
    
########################
# Data
########################
sim_file = os.path.join(DATA_DIR, 'datasets', f'{args.simulation_name}.h5')

train_dataset = dataset.RBDataset(sim_file, 'train', device=DEVICE, shuffle=True, samples=args.n_train)
valid_dataset = dataset.RBDataset(sim_file, 'valid', device=DEVICE, shuffle=True, samples=args.n_valid)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, drop_last=False)


########################
# Building Model
########################
# select default encoder channels if not specified
if args.encoder_channels is None:
    match args.conv_type:
        case 'SteerableConv':
            args.encoder_channels = {
                (True, 4): (6, 12, 23, 47)
                }[(args.flips, args.rots)]
        case 'SteerableConv3D':
            args.encoder_channels = {
                (True, 4): (24, 48, 93, 186)
                }[(args.flips, args.rots)]
        case 'Conv':
            args.encoder_channels = (10, 20, 40, 80)
        case 'Conv3D':
            args.encoder_channels = (24, 47, 93, 186)

model_hyperparameters = {
    'conv_type': args.conv_type,
    'simulation_name': args.simulation_name,
    'h_kernel_size': args.h_kernel_size,
    'v_kernel_size': args.v_kernel_size,
    'latent_h_kernel_size': args.latent_h_kernel_size,
    'latent_v_kernel_size': args.latent_v_kernel_size,
    'drop_rate': args.drop_rate,
    'nonlinearity': args.nonlinearity,
    'flips': args.flips,
    'rots': args.rots,
    'encoder_channels': args.encoder_channels,
    'latent_channels': args.latent_channels,
    'pool_layers': args.pool_layers,
    'v_shares': args.v_shares
}

print('Building model...')
model = build_autoencoder(**model_hyperparameters)
model.to(DEVICE)

model.summary()


########################
# Prepare Training
########################
G = f'{"D" if args.flips else "C"}{args.rots}'
model_name = {'SteerableConv': f'{G}cnn',
              'SteerableConv3D': f'3D-{G}cnn',
              'Conv': 'cnn',
              'Conv3D': '3Dcnn'}[args.conv_type]
model_name = os.path.join('AE', model_name)
    
train_dir = os.path.join(TRAINED_MODELS_DIR, model_name, args.train_name)
os.makedirs(train_dir, exist_ok=True)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_decay_patience, 
                                                          factor=args.lr_decay)

# data augmentation only by 90° rotations for efficiency reasons
data_augmentation = DataAugmentation(in_height=model.in_dims[-1], gspace=gspaces.flipRot2dOnR2(N=4))

# loads pretrained model if start_epoch > 0, loads last available epoch for -1
initial_early_stop_count, loaded_epoch = training.load_trained_model(model=model, 
                                                                     optimizer=optimizer, 
                                                                     models_dir=TRAINED_MODELS_DIR, 
                                                                     model_name=model_name, 
                                                                     train_name=args.train_name,
                                                                     epoch=args.start_epoch)
remaining_epochs = args.epochs - loaded_epoch


########################
# Save Hyperparameters
########################
train_hyperparameters = {
    'batch_size': args.batch_size,
    'n_train': train_dataset.num_samples,
    'n_valid': valid_dataset.num_samples,
    'learning_rate': args.lr,
    'optimizer': str(optimizer.__class__),
    'lr_decay': args.lr_decay,
    'lr_decay_patience': args.lr_decay_patience,
    'use_lr_scheduler': args.use_lr_scheduler,
    'weight_decay': args.weight_decay,
    'early_stopping': args.early_stopping,
    'early_stopping_threshold': args.early_stopping_threshold,
    'epochs': args.epochs,
    'train_loss_in_eval': args.train_loss_in_eval
}

hyperparameters = model_hyperparameters | train_hyperparameters

hyperparameters['latent_size'] = np.prod(model.latent_shape)/np.prod(model.layer_out_shapes()["Input"]) * 100
hyperparameters['parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

hp_file = os.path.join(train_dir, 'hyperparameters.json')
if loaded_epoch > 0 and os.path.isfile(hp_file):
    # check whether new hyperparameters are the same as the ones from the loaded model
    with open(hp_file, 'r') as f:
        prev_hps = json.load(f)
        prev_hps['epochs'] = hyperparameters['epochs'] # ignore epochs
        assert hyperparameters == prev_hps, f"new hyperparameters do not correspond to the old ones"
else:
    # save hyperparameters
    with open(hp_file, 'w+') as f:
        json.dump(hyperparameters, f, indent=4)

########################
# Training
########################
training.train(model=model, models_dir=TRAINED_MODELS_DIR, model_name=model_name, train_name=args.train_name, 
               start_epoch=loaded_epoch, epochs=remaining_epochs, train_loader=train_loader, valid_loader=valid_loader, 
               loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler, use_lr_scheduler=args.use_lr_scheduler, 
               early_stopping=args.early_stopping, only_save_best=args.only_save_best, batch_size=args.batch_size,
               train_samples=train_dataset.num_samples, data_augmentation=data_augmentation, plot=False, 
               initial_early_stop_count=initial_early_stop_count, train_loss_in_eval=args.train_loss_in_eval,
               early_stopping_threshold=args.early_stopping_threshold)
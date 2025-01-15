import os
import sys
import json

EXPERIMENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, '..'))

from utils import data_reader
from utils.data_augmentation import DataAugmentation
from torch.utils.data import DataLoader
from utils import training

from escnn import gspaces

from argparse import ArgumentParser

########################
# Parsing arguments
########################

parser = ArgumentParser()

parser.add_argument('model', type=str, choices=['3DCNN', 'CNN', 'steerable3DCNN', 'steerableCNN'])
parser.add_argument('epochs', type=int)
parser.add_argument('train_name', type=str)
parser.add_argument('-start_epoch', type=int, default=-1)
parser.add_argument('-including_loaded_epochs', action='store_true', default=False)
parser.add_argument('-only_save_best', type=bool, default=True)
parser.add_argument('-simulation_name', type=str, default='x48_y48_z32_Ra2500_Pr0.7_t0.01_snap0.125_dur300')
parser.add_argument('-n_train', type=int, default=-1)
parser.add_argument('-n_valid', type=int, default=-1)
parser.add_argument('-flips', type=bool, default=True)
parser.add_argument('-rots', type=int, default=4)
parser.add_argument('-lr', type=float, default=1e-3)

args = parser.parse_args()

########################
# Seed and GPU
########################
import torch, numpy as np, random
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(0)
    DEVICE = torch.cuda.current_device()
    print('Current device:', torch.cuda.get_device_name(DEVICE))
else:
    print('Failed to find GPU. Will use CPU.')
    DEVICE = 'cpu'
    
    
    
########################
# Data
########################
BATCH_SIZE = 64

SIMULATION_NAME = args.simulation_name

HORIZONTAL_SIZE = int(SIMULATION_NAME.split('_')[0][1:])
HEIGHT = int(SIMULATION_NAME.split('_')[2][1:])

sim_file = os.path.join(EXPERIMENT_DIR, '..', 'data', 'datasets', f'{SIMULATION_NAME}.h5')

N_train_avail, N_valid_avail, N_test_avail = data_reader.num_samples(sim_file, ['train', 'valid', 'test'])

# Reduce the amount of data manually
N_TRAIN = min(args.n_train, N_train_avail) if args.n_train > 0 else N_train_avail
N_VALID = min(args.n_valid, N_valid_avail) if args.n_valid > 0 else N_valid_avail

train_dataset = data_reader.DataReader(sim_file, 'train', device=DEVICE, shuffle=True, samples=N_TRAIN)
valid_dataset = data_reader.DataReader(sim_file, 'valid', device=DEVICE, shuffle=True, samples=N_VALID)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False)

print(f'Using {N_TRAIN}/{N_train_avail} training samples')
print(f'Using {N_VALID}/{N_valid_avail} validation samples')



########################
# Hyperparameter
########################

H_KERNEL_SIZE, V_KERNEL_SIZE = 3, 5
DROP_RATE = 0.2
NONLINEARITY = 'ELU'

LEARNING_RATE = args.lr
LR_DECAY = 0.1
LR_DECAY_EPOCHS = [10] # epochs at which the learning rate is multiplied by LR_DECAY
USE_LR_SCHEDULER = False
WEIGHT_DECAY = 0
EARLY_STOPPING = 20 # early stopping patience

OPTIMIZER = torch.optim.Adam


########################
# Building Model
########################
print('Building model...')
from utils.build_model import build_model


FLIPS, ROTS = args.flips, args.rots
LATENT_CHANNELS = 32
match args.model:
    case 'steerableCNN':
        print(f'Selected Steerable CNN with {ROTS=}, {FLIPS=}')
        encoder_channels = {(True, 4): (8, 16, 32, 64)}[(FLIPS, ROTS)]
    case 'steerable3DCNN':
        print(f'Selected Steerable 3D CNN with {ROTS=}, {FLIPS=}')
        encoder_channels = {(True, 4): (24, 48, 96, 192)}[(FLIPS, ROTS)]
    case 'CNN':
        print('Selected CNN')
        encoder_channels = (16, 32, 66, 160)
    case '3DCNN':
        print('Selected 3DCNN')
        encoder_channels = (40, 80, 168, 320)

model_hyperparameters = {
    'model_type': args.model,
    'simulation_name': SIMULATION_NAME,
    'h_kernel_size': H_KERNEL_SIZE,
    'v_kernel_size': V_KERNEL_SIZE,
    'drop_rate': DROP_RATE,
    'nonlinearity': NONLINEARITY,
    'flips': FLIPS,
    'rots': ROTS,
    'encoder_channels': encoder_channels,
    'latent_channels': LATENT_CHANNELS,
}

model = build_model(**model_hyperparameters)

model.to(DEVICE)
model.summary()

########################
# Prepare Training
########################

models_dir = os.path.join(EXPERIMENT_DIR, 'trained_models')
model_name = {'steerableCNN': f'{"D" if FLIPS else "C"}{ROTS}cnn',
              'steerable3DCNN': f'3D-{"D" if FLIPS else "C"}{ROTS}cnn',
              'CNN': 'cnn',
              '3DCNN': '3Dcnn'}[args.model]
train_name = args.train_name
train_dir = os.path.join(models_dir, model_name, train_name)
os.makedirs(train_dir, exist_ok=True)

loss_fn = torch.nn.MSELoss()
optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# data augmentation only by 90° rotations for efficiency reasons
data_augmentation = DataAugmentation(in_height=model.in_dims[-1], gspace=gspaces.flipRot2dOnR2(N=4))


START_EPOCH = args.start_epoch # loads pretrained model if greater 0, loads last available epoch for -1

initial_early_stop_count, loaded_epoch = training.load_trained_model(model=model, 
                                                                     optimizer=optimizer, 
                                                                     models_dir=models_dir, 
                                                                     model_name=model_name, 
                                                                     train_name=train_name,
                                                                     epoch=START_EPOCH)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DECAY_EPOCHS, 
                                                    gamma=LR_DECAY, last_epoch=loaded_epoch-1)

EPOCHS = args.epochs - loaded_epoch if args.including_loaded_epochs else args.epochs


########################
# Save Hyperparameters
########################

train_hyperparameters = {
    'batch_size': BATCH_SIZE,
    'n_train': N_TRAIN,
    'n_valid': N_VALID,
    'learning_rate': LEARNING_RATE,
    'lr_decay': LR_DECAY,
    'lr_decay_epochs': LR_DECAY_EPOCHS,
    'use_lr_scheduler': USE_LR_SCHEDULER,
    'weight_decay': WEIGHT_DECAY,
    'early_stopping': EARLY_STOPPING,
    'optimizer': str(OPTIMIZER),
    'epochs': loaded_epoch+EPOCHS
}

hyperparameters = model_hyperparameters | train_hyperparameters

with open(os.path.join(train_dir, 'hyperparameters.json'), 'w+') as f:
    json.dump(hyperparameters, f, indent=4)

########################
# Training
########################

training.train(model=model, models_dir=models_dir, model_name=model_name, train_name=train_name, start_epoch=loaded_epoch, 
               epochs=EPOCHS, train_loader=train_loader, valid_loader=valid_loader, loss_fn=loss_fn, 
               optimizer=optimizer, lr_scheduler=lr_scheduler, use_lr_scheduler=USE_LR_SCHEDULER, early_stopping=EARLY_STOPPING, only_save_best=args.only_save_best, train_samples=N_TRAIN, 
               batch_size=BATCH_SIZE, data_augmentation=data_augmentation, plot=False, 
               initial_early_stop_count=initial_early_stop_count)
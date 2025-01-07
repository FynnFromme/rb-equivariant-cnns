import os
import sys

EXPERIMENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, '..'))

from utils import data_reader
from utils.data_augmentation import DataAugmentation
from torch.utils.data import DataLoader
from utils import training

import escnn
from escnn import gspaces

from argparse import ArgumentParser

########################
# Parsing arguments
########################

parser = ArgumentParser()

parser.add_argument('model', type=str, choices=['3DCNN', 'CNN', 'steerableCNN'])
parser.add_argument('epochs', type=int)
parser.add_argument('-start_epoch', type=int, default=-1)
parser.add_argument('-only_save_best', type=bool, default=True)
parser.add_argument('-simulation_name', type=str, default='x48_y48_z32_Ra2500_Pr0.7_t0.01_snap0.125_dur300')
parser.add_argument('-n_train', type=int, default=-1)
parser.add_argument('-n_valid', type=int, default=-1)
parser.add_argument('-n_test', type=int, default=-1)
parser.add_argument('-flips', type=bool, default=True)
parser.add_argument('-rots', type=int, default=4)

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
N_TRAIN = args.n_train if args.n_train > 0 else N_train_avail
N_VALID = args.n_valid if args.n_valid > 0 else N_valid_avail
N_TEST = args.n_test if args.n_test > 0 else N_test_avail

train_dataset = data_reader.DataReader(sim_file, 'train', device=DEVICE, shuffle=True, samples=N_TRAIN)
valid_dataset = data_reader.DataReader(sim_file, 'valid', device=DEVICE, shuffle=True, samples=N_VALID)
test_dataset = data_reader.DataReader(sim_file, 'test', device=DEVICE, shuffle=True, samples=N_TEST)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=False)

print(f'Using {N_TRAIN}/{N_train_avail} training samples')
print(f'Using {N_VALID}/{N_valid_avail} validation samples')
print(f'Using {N_TEST}/{N_test_avail} testing samples')



########################
# Hyperparameter
########################

H_KERNEL_SIZE, V_KERNEL_SIZE = 3, 5
DROP_RATE = 0.2
NONLINEARITY, STEERABLE_NONLINEARITY = torch.nn.ELU, escnn.nn.ELU

LEARNING_RATE = 1e-3
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
from models.steerable_cnn_model import RBSteerableAutoencoder
from models.cnn_model import RBAutoencoder
from models.cnn3d_model import RB3DAutoencoder


FLIPS, ROTS = args.flips, args.rots
match args.model:
    case 'steerableCNN':
        print(f'Selected Steerable CNN with {ROTS=}, {FLIPS=}')
        gspace = gspaces.flipRot2dOnR2 if FLIPS else gspaces.rot2dOnR2
        model = RBSteerableAutoencoder(gspace=gspace(N=ROTS),
                                    rb_dims=(HORIZONTAL_SIZE, HORIZONTAL_SIZE, HEIGHT),
                                    encoder_channels=(8, 16, 32, 64),
                                    latent_channels=32//8, # 32 // |G|
                                    v_kernel_size=V_KERNEL_SIZE, h_kernel_size=H_KERNEL_SIZE,
                                    drop_rate=DROP_RATE, nonlinearity=STEERABLE_NONLINEARITY) 
    case 'CNN':
        print('Selected CNN')
        model = RBAutoencoder(rb_dims=(HORIZONTAL_SIZE, HORIZONTAL_SIZE, HEIGHT),
                            encoder_channels=(16, 32, 66, 160),
                            latent_channels=32,
                            v_kernel_size=V_KERNEL_SIZE, h_kernel_size=H_KERNEL_SIZE,
                            drop_rate=DROP_RATE, nonlinearity=NONLINEARITY)
    case '3DCNN':
        print('Selected 3DCNN')
        model = RB3DAutoencoder(rb_dims=(HORIZONTAL_SIZE, HORIZONTAL_SIZE, HEIGHT),
                                encoder_channels=(40, 80, 168, 320),
                                latent_channels=32,
                                v_kernel_size=V_KERNEL_SIZE, h_kernel_size=H_KERNEL_SIZE,
                                drop_rate=DROP_RATE, nonlinearity=NONLINEARITY)

model.to(DEVICE)
model.summary()



########################
# Training
########################

models_dir = os.path.join(EXPERIMENT_DIR, 'trained_models')

model_name = {RBSteerableAutoencoder: f'{"D" if FLIPS else "C"}{ROTS}cnn',
              RBAutoencoder: 'cnn',
              RB3DAutoencoder: '3Dcnn'}[model.__class__]

loss_fn = torch.nn.MSELoss()
optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# data augmentation only by 90° rotations for efficiency reasons
data_augmentation = DataAugmentation(in_height=model.in_dims[-1], gspace=gspaces.flipRot2dOnR2(N=4))


EPOCHS = args.epochs
START_EPOCH = args.start_epoch # loads pretrained model if greater 0, loads last available epoch for -1

initial_early_stop_count, loaded_epoch = training.load_trained_model(model=model, 
                                                                     optimizer=optimizer, 
                                                                     models_dir=models_dir, 
                                                                     model_name=model_name, 
                                                                     epoch=START_EPOCH)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DECAY_EPOCHS, 
                                                    gamma=LR_DECAY, last_epoch=loaded_epoch-1)



training.train(model=model, models_dir=models_dir, model_name=model_name, start_epoch=loaded_epoch, 
               epochs=EPOCHS, train_loader=train_loader, valid_loader=valid_loader, loss_fn=loss_fn, 
               optimizer=optimizer, lr_scheduler=lr_scheduler, use_lr_scheduler=USE_LR_SCHEDULER, early_stopping=EARLY_STOPPING, only_save_best=args.only_save_best, train_samples=N_TRAIN, 
               batch_size=BATCH_SIZE, data_augmentation=data_augmentation, plot=False, 
               initial_early_stop_count=initial_early_stop_count)
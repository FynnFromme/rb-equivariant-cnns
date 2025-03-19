import os
import sys
import json

EXPERIMENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, '..'))

from experiments.utils import dataset
from torch.utils.data import DataLoader
from utils.latent_dataset import compute_latent_dataset
from utils.data_augmentation import DataAugmentation
from experiments.utils.model_building import build_forecaster, build_and_load_trained_model
from utils import training

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
                    help='The type of convolutions used for the convolutional LSTMs.')
parser.add_argument('train_name', type=str,
                    help='The name of the trained model.')
parser.add_argument('ae_model_name', type=str,
                    help='The model name of the autoencoder used for the forecaster (e.g. cnn, 3Dcnn, D4cnn, 3D-D4cnn)')
parser.add_argument('ae_train_name', type=str,
                    help='The name of the trained autoencoder used for the forecaster.')
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
parser.add_argument('-v_kernel_size', type=int, default=3,
                    help='The vertical kernel size. Defaults to `3`.')
parser.add_argument('-h_kernel_size', type=int, default=3,
                    help='The horizontal kernel size. Defaults to `3`.')
parser.add_argument('-drop_rate', type=float, default=0.2,
                    help='The drop rate applied to the inputs of the LSTM. Defaults to `0.2`.')
parser.add_argument('-recurrent_drop_rate', type=float, default=0,
                    help='The dropout rate applied to the hidden states of the LSTM. Defaults to `0`.')
parser.add_argument('-nonlinearity', type=str, default='tanh', choices=['tanh', 'ReLU', 'ELU'],
                    help='The nonlinearity used in the autoencoder. Defaults to "tanh".')
parser.add_argument('-weight_decay', type=float, default=0,
                    help='The L2-regularization parameter. Defaults to `0`.')
parser.add_argument('-lstm_channels', nargs='+', type=int, default=None,
                    help='The number of channels used for the hidden and cell state in each layer.')
parser.add_argument('-residual_connection', type=str2bool, default=True,
                    help='Whether to add a residual connection between the input and the output of \
                        the forecaster. Defaults to `True`.')
parser.add_argument('-use_lstm_encoder', type=str2bool, default=True,
                    help='Whether to use a separate LSTM encoder to encode the warmup sequence into \
                        some representation. Defaults to `True`.')

# training hyperparameters
parser.add_argument('-warmup_seq_length', type=int, default=25,
                    help='The length of the warmup sequence, which the model gets as input during training \
                        Defaults to `25`.')
parser.add_argument('-forecast_seq_length', type=int, default=50,
                    help='The length of the forecasted sequence during training. Defaults to `50`.')
parser.add_argument('-parallel_ops', type=str2bool, default=False,
                    help='Whether to apply certain operations (like autoencoder, output head, etc.) \
                        in parallel to the whole sequence. Defaults to `False`.')
parser.add_argument('-loss_on_decoded', type=str2bool, default=False,
                    help='When set to `True`, the loss is computed based on the decoded data rather than \
                        on the latent representation directly. Defaults to `False`.')
parser.add_argument('-use_force_decoding', type=str2bool, default=True,
                    help='If set to `True`, a certain percentage of predictions during training \
                        are made based on the ground truth. The percentage linearly decreases over \
                        the epochs. Defaults to `True`.')
parser.add_argument('-init_forced_decoding_prob', type=float, default=1,
                    help='The initial probability for forced decoding at the start of the training. \
                        Defaults to `1.0`.')
parser.add_argument('-min_forced_decoding_prob', type=float, default=0,
                    help='The minimum probability for forced decoding during training. Defaults to `0.0`.')
parser.add_argument('-forced_decoding_epochs', type=int, default=100,
                    help='The number of epochs over which the forced decoding probability is linearly decreased \
                        Defaults to `100`.')
parser.add_argument('-backprop_through_autoregression', type=str2bool, default=True,
                    help='Whether to backpropagate through autoregressive steps. Defaults to `True`.')

parser.add_argument('-train_autoencoder', type=str2bool, default=False,
                    help='Whether to train the autoencoder weights or freeze them. Defaults to `False`.')
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

if args.train_autoencoder and not args.loss_on_decoded:
    raise Exception('When training the autoencoder, the loss must be computed on the decoded output. \
        Please add `-loss_on_decoded true`.')
if args.loss_on_decoded and args.use_force_decoding:
    raise Exception('When training on decoded data, force decoding is currently not supported. \
        Please add `-use_force_decoding false`.')

if not args.use_force_decoding:
    args.init_forced_decoding_prob = 0
    args.min_forced_decoding_prob = 0
    

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

if not args.loss_on_decoded: 
    # model is directly trained on the latent representations
    # these will be precomputed in the following for higher efficiency
    latent_file = os.path.join(EXPERIMENT_DIR, 'latent_datasets', args.ae_model_name, args.ae_train_name, f'{args.simulation_name}.h5')
    
    if not os.path.isfile(latent_file):
        print('Loading autoencoder to precompute latent dataset...')
        autoencoder = build_and_load_trained_model(TRAINED_MODELS_DIR, os.path.join('AE', args.ae_model_name), args.ae_train_name)
        autoencoder.to(DEVICE)
        
        # data augmentation will be included in the dataset itself since transformation laws do not apply
        # on latent space for non-equivariant models
        latent_data_aug = DataAugmentation(in_height=autoencoder.in_dims[-1], gspace=gspaces.flipRot2dOnR2(N=4))
        
        print('Precomputing latent dataset...')
        compute_latent_dataset(autoencoder, latent_file, sim_file, device=DEVICE, batch_size=args.batch_size,
                               data_augmentation=latent_data_aug)
        
        autoencoder.to('cpu') # required to fix device mismatch when loading autoencoder later again
        del autoencoder
        
    sim_file = latent_file

data_aug_in_dataset = not args.loss_on_decoded # data augmentation is included in latent datasets only
train_dataset = dataset.RBForecastDataset(sim_file, 'train', device=DEVICE, shuffle=True, samples=args.n_train, 
                                          warmup_seq_length=args.warmup_seq_length, forecast_seq_length=args.forecast_seq_length, 
                                          data_aug=data_aug_in_dataset)
valid_dataset = dataset.RBForecastDataset(sim_file, 'valid', device=DEVICE, shuffle=True, samples=args.n_valid, 
                                          warmup_seq_length=args.warmup_seq_length, forecast_seq_length=args.forecast_seq_length)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, drop_last=False)


########################
# Building Model
########################
# select default encoder channels if not specified
if args.lstm_channels is None:
    match args.conv_type:
        case 'SteerableConv':
            args.lstm_channels = {
                (True, 4): (8, 8)
                }[(args.flips, args.rots)]
        case 'SteerableConv3D':
            args.lstm_channels = {
                (True, 4): (8, 8)
                }[(args.flips, args.rots)]
        case 'Conv':
            args.lstm_channels = (35, 35)
        case 'Conv3D':
            args.lstm_channels = (35, 35)    

model_hyperparameters = {
    'conv_type': args.conv_type,
    'ae_model_name': args.ae_model_name,
    'ae_train_name': args.ae_train_name,
    'simulation_name': args.simulation_name,
    'h_kernel_size': args.h_kernel_size,
    'v_kernel_size': args.v_kernel_size,
    'drop_rate': args.drop_rate,
    'recurrent_drop_rate': args.recurrent_drop_rate,
    'nonlinearity': args.nonlinearity,
    'flips': args.flips,
    'rots': args.rots,
    'lstm_channels': args.lstm_channels,
    'parallel_ops': args.parallel_ops,
    'residual_connection': args.residual_connection,
    'train_autoencoder': args.train_autoencoder,
    'include_autoencoder': args.loss_on_decoded,
    'use_lstm_encoder': args.use_lstm_encoder,
    'init_forced_decoding_prob': args.init_forced_decoding_prob,
    'min_forced_decoding_prob': args.min_forced_decoding_prob,
    'forced_decoding_epochs': args.forced_decoding_epochs,
    'use_force_decoding': args.use_force_decoding,
    'backprop_through_autoregression': args.backprop_through_autoregression
}

print('Building model...')
model = build_forecaster(models_dir=TRAINED_MODELS_DIR, **model_hyperparameters)
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
model_name = os.path.join('FC', model_name)

train_dir = os.path.join(TRAINED_MODELS_DIR, model_name, args.train_name)
os.makedirs(train_dir, exist_ok=True)

loss_fn = torch.nn.MSELoss()
trainable_parameters = model.parameters()
optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_decay_patience, 
                                                          factor=args.lr_decay, threshold=1e-5)

if args.loss_on_decoded:
    # data augmentation only by 90° rotations for efficiency reasons
    data_augmentation = DataAugmentation(in_height=model.in_dims[-1], gspace=gspaces.flipRot2dOnR2(N=4))
else:
    # data augmentation is already included in the latent dataset
    data_augmentation = None

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
    'train_loss_in_eval': args.train_loss_in_eval,
    'warmup_seq_length': args.warmup_seq_length,
    'forecast_seq_length': args.forecast_seq_length,
}

hyperparameters = model_hyperparameters | train_hyperparameters

hyperparameters['parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

hp_file = os.path.join(train_dir, 'hyperparameters.json')
if loaded_epoch > 0 and os.path.isfile(hp_file):
    # check whether new hyperparameters are the same as the ones from the loaded model
    with open(hp_file, 'r') as f:
        prev_hps = json.load(f)
        prev_hps['epochs'] = hyperparameters['epochs'] # ignore epochs
        assert hyperparameters == prev_hps, f"New hyperparameters do not correspond to the old ones"
else:
    # save hyperparameters
    with open(hp_file, 'w+') as f:
        json.dump(hyperparameters, f, indent=4)
        

########################
# Training
########################
if args.loss_on_decoded:
    if args.train_autoencoder:
        for param in model.autoencoder.parameters():
            param.requires_grad = True
    else:
        # Freeze the autoencoder parameters so they are not updated during training.
        for param in model.autoencoder.parameters():
            param.requires_grad = False
    
# epoch and ground_truth will be replaced by training script
model_forward_kwargs = {'steps': args.forecast_seq_length, 'epoch': None, 'ground_truth': None}
training.train(model=model, models_dir=TRAINED_MODELS_DIR, model_name=model_name, train_name=args.train_name, 
               start_epoch=loaded_epoch, epochs=remaining_epochs, train_loader=train_loader, valid_loader=valid_loader, 
               loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler, use_lr_scheduler=args.use_lr_scheduler, 
               early_stopping=args.early_stopping, only_save_best=args.only_save_best, train_samples=train_dataset.num_samples, 
               model_forward_kwargs=model_forward_kwargs, initial_early_stop_count=initial_early_stop_count,
               train_loss_in_eval=args.train_loss_in_eval, early_stopping_threshold=args.early_stopping_threshold, 
               batch_size=args.batch_size, data_augmentation=data_augmentation, plot=False)
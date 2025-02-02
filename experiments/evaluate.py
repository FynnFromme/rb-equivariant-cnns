import os
import sys
import json
import numpy as np

EXPERIMENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, '..'))

from utils import data_reader
from utils.evaluation import compute_loss, compute_loss_per_channel
from utils.visualization import auto_encoder_animation
from utils.evaluation import compute_latent_sensitivity

import escnn
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from tqdm.auto import tqdm


########################
# Parsing arguments
########################

parser = ArgumentParser()

parser.add_argument('model_name', type=str)
parser.add_argument('train_name', type=str)
parser.add_argument('-eval_performance', action='store_true', default=False)
parser.add_argument('-eval_performance_per_sim', action='store_true', default=False)
parser.add_argument('-eval_performance_per_channel', action='store_true', default=False)
parser.add_argument('-eval_performance_per_height', action='store_true', default=False)
parser.add_argument('-check_equivariance', action='store_true', default=False)
parser.add_argument('-animate', action='store_true', default=False)
parser.add_argument('-compute_latent_sensitivity', action='store_true', default=False)
parser.add_argument('-animation_samples', type=int, default=np.inf)
parser.add_argument('-latent_sensitivity_samples', type=int, default=-1)
parser.add_argument('-simulation_name', type=str, default='x48_y48_z32_Ra2500_Pr0.7_t0.01_snap0.125_dur300')
parser.add_argument('-n_test', type=int, default=-1)
parser.add_argument('-n_train', type=int, default=-1)
parser.add_argument('-batch_size', type=int, default=64)

args = parser.parse_args()

if not any([args.eval_performance, 
            args.eval_performance_per_sim,
            args.eval_performance_per_channel,
            args.eval_performance_per_height,
            args.check_equivariance, 
            args.animate, 
            args.compute_latent_sensitivity]):
    print('No evaluation selected')
    print('Please add at least one of the following flags:')
    print('-eval_performance')
    print('-eval_performance_per_sim')
    print('-eval_performance_per_channel')
    print('-eval_performance_per_height')
    print('-check_equivariance')
    print('-animate')
    print('-compute_latent_sensitivity')
    exit()


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

sim_file = os.path.join(EXPERIMENT_DIR, '..', 'data', 'datasets', f'{args.simulation_name}.h5')
N_test_avail, N_train_avail = data_reader.num_samples(sim_file, ['test', 'train'])
N_TEST = min(args.n_test, N_test_avail) if args.n_test > 0 else N_test_avail
N_TRAIN = min(args.n_train, N_train_avail) if args.n_train > 0 else N_train_avail
test_dataset = data_reader.DataReader(sim_file, 'test', device=DEVICE, shuffle=False, samples=N_TEST)
train_dataset = data_reader.DataReader(sim_file, 'train', device=DEVICE, shuffle=False, samples=N_TRAIN)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

print(f'Using {N_TEST}/{N_test_avail} testing samples')



########################
# Load Model
########################

from utils.training import build_and_load_trained_model
models_dir = os.path.join(EXPERIMENT_DIR, 'trained_models')

model = build_and_load_trained_model(models_dir, args.model_name, args.train_name, epoch=-1)
model.to(DEVICE) 
   
   
   
########################
# Evaluate Model
########################  

def load_json(json_file):
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    else:
        return {}   
    
def save_json(dict, json_file):
    with open(json_file, 'w+') as f:
        json.dump(dict, f, indent=4)

results_dir = os.path.join(EXPERIMENT_DIR, 'results', args.model_name, args.train_name)
os.makedirs(results_dir, exist_ok=True)


if args.eval_performance:
    print('Evaluating model performance...')
    
    # read current performances
    performance_file = os.path.join(results_dir, 'performance.json')
    performance = load_json(performance_file)
    
    # update new performance metrics but keep other contents
    performance['mse'], performance['mae'] = compute_loss(model, test_loader,
                                                          [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                                                          N_TEST, args.batch_size)
    performance['rmse'] = np.sqrt(performance['mse'])
    
    performance['mse_train'], performance['mae_train'] = compute_loss(model, train_loader,
                                                                      [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                                                                      N_TRAIN, args.batch_size)
    performance['rmse_train'] = np.sqrt(performance['mse_train'])

    print(f'MSE={performance["mse"]:.4f}')
    print(f'RMSE={performance["rmse"]:.4f}')
    print(f'MAE={performance["mae"]:.4f}')
    
    print(f'MSE_train={performance["mse_train"]:.4f}')
    print(f'RMSE_train={performance["rmse_train"]:.4f}')
    print(f'MAE_train={performance["mae_train"]:.4f}')
    
    # update performances in file
    save_json(performance, performance_file)
        
        
        
if args.eval_performance_per_sim:
    print('Evaluating model performance per simulation...')
    
    # read current simulation performances
    sim_performances_file = os.path.join(results_dir, 'performance_per_sim.json')
    sim_performances = load_json(sim_performances_file)
    sim_performances = sim_performances | {'mse': [], 'rmse': [], 'mae': []}
    
    # update new performance metrics but keep other contents
    for i, sim_dataset in tqdm(enumerate(test_dataset.iterate_simulations(), 1), total=test_dataset.num_simulations,
                               desc='evaluating simulations', unit='sim'):
        sim_loader = DataLoader(sim_dataset, batch_size=args.batch_size, drop_last=False)
        
        mse, mae = compute_loss(model, sim_loader,
                                [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                                sim_dataset.num_samples, args.batch_size)
        rmse = np.sqrt(mse)
        
        sim_performances['mse'].append(mse)
        sim_performances['rmse'].append(rmse)
        sim_performances['mae'].append(mae)

    sim_performances['std_mse'] = np.std(sim_performances['mse'])
    sim_performances['std_rmse'] = np.std(sim_performances['rmse'])
    sim_performances['std_mae'] = np.std(sim_performances['mae'])
    
    print(sim_performances)
    
    # update performances in file
    save_json(sim_performances, sim_performances_file)



if args.eval_performance_per_channel:
    print('Evaluating model performance per channel...')
    
    # read current performances
    channel_performance_file = os.path.join(results_dir, 'performance_per_channel.json')
    channel_performance = load_json(channel_performance_file)
    
    # update new performance metrics but keep other contents
    channel_performance['mse'], channel_performance['mae'] = compute_loss_per_channel(model, test_loader,
                                                          [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                                                          -1, N_TEST, args.batch_size)
    channel_performance['rmse'] = list(np.sqrt(channel_performance['mse']))
    
    channel_performance['mse_train'], channel_performance['mae_train'] = compute_loss_per_channel(model,
                                                                        train_loader,
                                                                        [torch.nn.MSELoss(), 
                                                                         torch.nn.L1Loss()], 
                                                                        -1, N_TRAIN, args.batch_size)
    channel_performance['rmse_train'] = list(np.sqrt(channel_performance['mse_train']))

    print(f'MSE={channel_performance["mse"]}')
    print(f'RMSE={channel_performance["rmse"]}')
    print(f'MAE={channel_performance["mae"]}')
    
    print(f'MSE_train={channel_performance["mse_train"]}')
    print(f'RMSE_train={channel_performance["rmse_train"]}')
    print(f'MAE_train={channel_performance["mae_train"]}')
    print(channel_performance, type(channel_performance))
    # update performances in file
    save_json(channel_performance, channel_performance_file)


if args.eval_performance_per_height:
    print('Evaluating model performance per height...')
    
    # read current performances
    height_performance_file = os.path.join(results_dir, 'performance_per_height.json')
    height_performance = load_json(height_performance_file)
    
    # update new performance metrics but keep other contents
    # over all channels:
    height_performance['mse'], height_performance['mae'] = compute_loss_per_channel(model, test_loader,
                                                          [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                                                          -2, N_TEST, args.batch_size)
    height_performance['rmse'] = list(np.sqrt(height_performance['mse']))
    
    height_performance['mse_train'], height_performance['mae_train'] = compute_loss_per_channel(model,
                                                                        train_loader,
                                                                        [torch.nn.MSELoss(), 
                                                                         torch.nn.L1Loss()], 
                                                                        -2, N_TRAIN, args.batch_size)
    height_performance['rmse_train'] = list(np.sqrt(height_performance['mse_train']))
    
    # per channel:
    height_performance['mse_per_channel'], height_performance['mae_per_channel'] = compute_loss_per_channel(model, test_loader,
                                                          [torch.nn.MSELoss(), torch.nn.L1Loss()], 
                                                          [-1, -2], N_TEST, args.batch_size)
    height_performance['rmse_per_channel'] = np.sqrt(height_performance['mse_per_channel']).tolist()
    
    height_performance['mse_train_per_channel'], height_performance['mae_train_per_channel'] = compute_loss_per_channel(model,
                                                                        train_loader,
                                                                        [torch.nn.MSELoss(), 
                                                                         torch.nn.L1Loss()], 
                                                                        [-1, -2], N_TRAIN, args.batch_size)
    height_performance['rmse_train_per_channel'] = np.sqrt(height_performance['mse_train_per_channel']).tolist()
    
    # update performances in file
    save_json(height_performance, height_performance_file)


if args.check_equivariance:
    # TODO save to file
    if isinstance(model, escnn.nn.EquivariantModule):
        print('Checking Equivariance...')
        model.check_equivariance(gpu_device=DEVICE, atol=1e-3) 
    else:
        # TODO implement computation of equivariance error for other models
        print('Equivariance can currently only be checked for steerable models.')
        
        
        
if args.animate:
    anim_dir = os.path.join(results_dir, 'animations')
    horizontal_size = int(args.simulation_name.split('_')[0][1:])
    height = int(args.simulation_name.split('_')[2][1:])
    
    print('Animating...')
    with tqdm(total=12, desc='animating', unit='animation') as pbar:
        for feature in ['t', 'u', 'v', 'w']:
            for axis, dim in enumerate(['width', 'depth', 'height']):
                slice = height//2 if axis == 2 else horizontal_size//2
                auto_encoder_animation(slice=slice, fps=25, frames=args.animation_samples, feature=feature, 
                                    axis=axis, anim_dir=os.path.join(anim_dir, feature), anim_name=f'{dim}.mp4', 
                                    model=model, sim_file=sim_file, device=DEVICE)
                pbar.update(1)


if args.compute_latent_sensitivity:
    print('Computing latent sensitivity...')
    
    SAMPLES = min(args.latent_sensitivity_samples, N_test_avail) if args.latent_sensitivity_samples > 0 else N_test_avail
    sensitivity_dataset = data_reader.DataReader(sim_file, 'test', device=DEVICE, shuffle=False, samples=SAMPLES)
    
    avg_sensitivity, avg_abs_sensitivity = compute_latent_sensitivity(model, sensitivity_dataset, 
                                                                      samples=SAMPLES, 
                                                                      save_dir=results_dir,
                                                                      filename='latent_sensitivity')
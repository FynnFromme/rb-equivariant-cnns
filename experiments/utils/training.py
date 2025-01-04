from tqdm.auto import tqdm

import math
import torch
import numpy as np

from torch.utils.data import DataLoader

import os
import glob
import re

from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .data_augmentation import DataAugmentation


def train(model: torch.nn.Module, models_dir: str, model_name: str, start_epoch: int, epochs: int, 
          train_loader: DataLoader, valid_loader: DataLoader, loss_fn, optimizer, lr_scheduler, 
          use_lr_scheduler: bool, early_stopping: int, only_save_best: bool, train_samples: int, batch_size: int,
          data_augmentation: DataAugmentation, plot: bool, initial_early_stop_count: int = 0):

    writer = SummaryWriter(f"runs/{model_name}") # Tensorboard writer

    output_dir = os.path.join(models_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    best_loss = np.inf
    best_epoch = -1
    train_loss_values = []
    valid_loss_values = []
    early_stop_count = initial_early_stop_count
    
    with tqdm(total=epochs, desc='training', unit='epoch') as pbar:
        for epoch in range(1+start_epoch, 1+start_epoch+epochs):
            train_loss, valid_loss = train_epoch(train_loader, valid_loader, model, loss_fn, optimizer, 
                                                data_augmentation, epoch, batch_size, train_samples)
            
            pbar.update(1)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            train_loss_values.append(train_loss)
            valid_loss_values.append(valid_loss)
            
            if use_lr_scheduler:
                lr_scheduler.step()
                
            if valid_loss < best_loss:
                early_stop_count = 0
                
                best_loss = valid_loss
                best_epoch = epoch
                best_weights = deepcopy(model.state_dict())
                best_optim = optimizer.state_dict()
                pbar.set_postfix_str(f'{best_epoch=:}, {train_loss=:.4f}, {best_loss=:.4f}')
                
                if only_save_best:
                    remove_saved_models(output_dir)
                
                output_file = os.path.join(output_dir, f'epoch{epoch}.tar')
                save_checkpoint(output_file, best_weights, best_optim, early_stop_count)
            else:
                early_stop_count += 1
            
            if early_stop_count >= early_stopping:
                print(f"Early stopping at epoch {epoch}. \
                    Best epoch is {best_epoch} with a validation loss of {best_loss}.")
                break
            
    writer.flush()
    writer.close()
    
    if plot:
        epochs = range(start_epoch+1, start_epoch+epochs+1)
        plt.plot(epochs, train_loss_values, label="train")
        plt.plot(epochs, valid_loss_values, label="valid")
        plt.xticks(epochs)
        
        plt.title(f"{model_name}   Best loss: {best_loss:>.3f}   Best epoch: {best_epoch}")
        plt.legend()
        
        plt.show()
    

def train_epoch(train_loader: DataLoader, valid_loader: DataLoader, model, loss_fn, optimizer, 
                data_augmentation, epochnum, batch_size, samples):
    model.train() # Sets the model to training mode -- important for batch normalization and dropout layers
    running_loss = 0.0
    
    with tqdm(total=math.ceil(samples/batch_size), desc=f'epoch {epochnum}', unit='batch') as pbar:
        for i, (x, _) in enumerate(train_loader, 1):
            x = data_augmentation(x)
            
            # Compute prediction and loss
            pred = model(x)
            
            loss = loss_fn(pred, x)
            running_loss += loss.item()
        
            # Backpropagation
            optimizer.zero_grad() # Resets gradient
            loss.backward() # Backpropagates the prediction loss
            optimizer.step() # Adjusts the parameters by the gradients collected in the backward pass
            
            pbar.set_postfix_str(f'train_loss={running_loss/i:.4f}')
            pbar.update(1)
            
        train_loss = running_loss / i
        valid_loss = compute_validation_loss(valid_loader, model, loss_fn)
        
        pbar.set_postfix_str(f'{train_loss=:.4f}, {valid_loss=:.4f}')
    
    return train_loss, valid_loss


def compute_validation_loss(dataloader: DataLoader, model, loss_fn):
    model.eval() # Sets the model to evaluation mode -- important for batch normalization and dropout layers
    valid_loss = 0.0
    with torch.no_grad(): # Ensures that no gradients are computed during test mode
        for i, (x, y) in enumerate(dataloader, 1):
            pred = model(x)
            valid_loss += loss_fn(pred, y).item()

    return valid_loss / i


def save_checkpoint(path, weights, optimizer_state, early_stop_count):
    torch.save({
        'model_state_dict': weights,
        'optimizer_state_dict': optimizer_state,
        'early_stop_count': early_stop_count
    }, path)
    

def load_trained_model(model, optimizer, models_dir, model_name, epoch=-1):
    directory = os.path.join(models_dir, model_name)
    
    if epoch == 0:
        return 0
    if not os.path.exists(directory) or not os.listdir(directory):
        raise Exception('no saved model to load')
    elif epoch == -1:
        # load latest epoch
        pattern = re.compile(r"epoch([0-9]+)\.tar")
        saved_epochs = [filename for filename in os.listdir(directory) if pattern.fullmatch(filename)]
        file_path = max(saved_epochs, key=lambda f: int(pattern.fullmatch(f).group(1)))
    else:
        # load specified epoch
        file_path = os.path.join(directory, f'epoch{epoch}.tar')
        if not os.path.isfile(file_path):
            raise Exception('model not saved for that epoch')
    
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    early_stop_count = checkpoint['early_stop_count']
    model.eval()
    
    print(f"Loaded state at epoch {epoch} with an early stop count of {early_stop_count}.")
    
    return early_stop_count
    
    
def remove_saved_models(directory):
    path_pattern = os.path.join(directory, 'epoch*.tar')

    for file_path in glob.glob(path_pattern):
        os.remove(file_path)
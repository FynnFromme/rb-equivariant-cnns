import torch
from torch.utils.data import DataLoader

def compute_test_loss(model: torch.nn.Module, test_loader: DataLoader, loss_fn):
    model.eval()

    running_test_loss = 0
    with torch.no_grad():
        for i, (inputs, outputs) in enumerate(test_loader, 1):
            predictions = model(inputs)
            
            test_loss = loss_fn(outputs, predictions).item()
            running_test_loss += test_loss
            
    avg_test_loss = running_test_loss / i

    return avg_test_loss
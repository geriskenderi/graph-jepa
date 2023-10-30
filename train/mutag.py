import torch
import numpy as np
import torch.nn.functional as F
from core.config import cfg, update_cfg
from core.get_data import create_dataset
from core.get_model import create_model
from core.trainer import run_k_fold
from core.model_utils.hyperbolic_dist import hyperbolic_dist

def train(train_loader, model, optimizer, evaluator, device, momentum_weight,sharp=None, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss()
    step_losses, num_targets = [], []
    for data in train_loader:
        if model.use_lap: # Sign flips for eigenvalue PEs
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()
        target_x, target_y = model(data)

        if criterion_type == 0:
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            exit()

        # Will need these for the weighted average at the end of the epoch
        step_losses.append(loss.item())
        num_targets.append(len(target_y))
        
        # Update weights of the network 
        loss.backward()
        optimizer.step()

        # Other than the target encoder, here we use exponential smoothing
        with torch.no_grad():
            for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
                param_k.data.mul_(momentum_weight).add_((1.-momentum_weight) * param_q.detach().data)
        
    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss # Leave none for now since maybe we'd like to return the embeddings for visualization


@ torch.no_grad()
def test(loader, model, evaluator, device, criterion_type=0):
    criterion = torch.nn.SmoothL1Loss()
    step_losses, num_targets = [], []
    for data in loader:
        data = data.to(device)
        target_x, target_y = model(data)
        if criterion_type == 0:
            loss = criterion(target_x, target_y)
        elif criterion_type == 1:
            loss = F.mse_loss(target_x, target_y)
        elif criterion_type == 2:
            loss = hyperbolic_dist(target_x, target_y)
        else:
            print('Loss function not supported! Exiting!')
            exit()

        # Will need these for the weighted average at the end of the epoch
        step_losses.append(loss.item())
        num_targets.append(len(target_y))

    epoch_loss = np.average(step_losses, weights=num_targets)
    return None, epoch_loss


if __name__ == '__main__':
    cfg.merge_from_file('train/configs/mutag.yaml')
    cfg = update_cfg(cfg) # we can specify the config file for the update here
    cfg.k = 10
    run_k_fold(cfg, create_dataset, create_model, train, test)

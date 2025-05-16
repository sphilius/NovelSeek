import torch
import torch.nn as nn

def RMSE(predictions, targets):
    mse_eval = nn.MSELoss()
    rmse = torch.sqrt(mse_eval(predictions, targets)).item()
    return rmse
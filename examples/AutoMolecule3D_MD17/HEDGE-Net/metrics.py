import numpy as np

def calculate_mae(y_true, y_pred):

    mae = np.abs(y_true - y_pred).mean()
    return mae

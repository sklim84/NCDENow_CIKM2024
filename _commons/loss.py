import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, kl_divergence
from sklearn.metrics import mean_absolute_percentage_error

def loss_mse(y_pred, y_target):
    """
    This loss function is based on Mean Squared Error

    Args:
        y_pred: prediction value
        y_target: target value

    Returns:
        mse_loss: loss value

    """
    if type(y_pred) is Tensor and type(y_target) is Tensor:
        mse_loss_func = nn.MSELoss()
        mse_loss = mse_loss_func(y_pred, y_target)
        return mse_loss
    else:
        return np.mean(np.square(np.subtract(y_pred, y_target)))


# below loss was taken from 'Are Transformers Effective for Time Series Forecasting?'

def MAE(pred_y, true_y):
    return np.mean(np.abs(pred_y - true_y))


def MSE(pred_y, true_y):
    return np.mean((pred_y - true_y) ** 2)


def MAPE(pred_y, true_y):
    return mean_absolute_percentage_error(true_y, pred_y) * 100
    # return np.mean(np.abs((pred_y - true_y) / true_y))

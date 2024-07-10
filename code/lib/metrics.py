import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def mape_np(predictions, labels):
    # print(np.sum(np.abs(np.subtract(predictions, labels))))
    # print(np.sum(labels))
    # print(predictions.shape)
    # print(labels.shape)
    mape = np.divide(np.sum(np.abs(np.subtract(predictions, labels)).astype('float32')), np.sum(labels))
    mape = np.nan_to_num(mape)
    return mape


def mae_np(predictions, labels):
    mae = np.abs(np.subtract(predictions, labels)).astype('float32')
    mae = np.nan_to_num(mae)
    return np.mean(mae)


def mse_np(predictions, labels):
    mse = np.square(np.subtract(predictions, labels)).astype('float32')
    mse = np.nan_to_num(mse)
    return np.mean(mse)


def rmse_np(predictions, labels):
    return np.sqrt(mse_np(predictions, labels))


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    # print(loss.sum().item(), loss.mean().item())
    loss = loss.mean()
    # loss = loss.sum()
    return loss


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, x, y):
        y_sum = torch.sum(y, dim=(-2, -1)).unsqueeze(dim=1).unsqueeze(dim=2)
        weights = torch.abs(y) / y_sum
        loss = torch.abs(x - y)
        weight_loss = torch.matmul(weights, loss)
        return torch.sum(weight_loss)

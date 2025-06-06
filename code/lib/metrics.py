import numpy as np

def mape_np(predictions, labels):
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

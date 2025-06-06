import argparse
import os

import numpy as np
import yaml
import time

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from models.odmixer import ODMixer
from lib import utils_data as utils
from lib import metrics

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--config_filename',
                    default='data/config/train_hz.yaml',
                    type=str,
                    help='Configuration filename for restoring the model')
args = parser.parse_args()


def read_cfg_file(filename):
    with open(filename, 'r') as f:
        cfg = yaml.load(f, Loader=Loader)
    return cfg


def _get_log_dir(kwargs, status=''):
    log_dir = kwargs['train'].get('log_dir')
    if log_dir is None:
        batch_size = kwargs['data'].get('batch_size')
        learning_rate = kwargs['train'].get('base_lr')
        input_seq = kwargs['model']['input_seq']

        run_id = 'demo_lr{}_bs{}_seq{}_{}'.format(learning_rate, batch_size, input_seq, time.strftime('%m%d%H%M%S'))
        base_dir = kwargs.get('base_dir')
        log_dir = os.path.join(base_dir, status + run_id)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    return log_dir

def evaluate(model, dataset, dataset_type, cfg, logger, log_dir, device, scaler, gaussian_std=None):
    logger.info('{} begin'.format(dataset_type))

    data_loader = dataset[dataset_type + '_loader']
    model.eval()
    y_od = []
    y_do = []

    begin_time = time.perf_counter()
    for _, sequence in enumerate(data_loader):
        for key, element in sequence.items():
            sequence[key] = element.float().to(device)

        with torch.no_grad():
            prediction_od, _a = model(sequence)
            y_od.append(prediction_od.detach().cpu().numpy())
    end_time = time.perf_counter()
    logger.info('Infer Time: {}'.format(end_time - begin_time))

    scaler_od = dataset['scaler_od']
    gt = dataset[dataset_type]['y_od']
    y_od = np.concatenate(y_od, axis=0)

    print('Length: {}'.format(y_od.shape))

    mae_list = []
    mape_net_list = []
    rmse_list = []
    mae_sum = 0
    mape_net_sum = 0
    rmse_sum = 0
    horizon = cfg['model']['horizon']
    for horizon_i in range(horizon):
        y_truth = scaler_od.inverse_transform(
            gt[:, horizon_i, :, :])

        y_pred = scaler_od.inverse_transform(
            y_od[:y_truth.shape[0], horizon_i, :, :])
        y_pred[y_pred < 0] = 0

        mae = metrics.mae_np(y_pred, y_truth)
        mape_net = metrics.mape_np(y_pred, y_truth)
        rmse = metrics.rmse_np(y_pred, y_truth)
        mae_sum += mae
        mape_net_sum += mape_net
        rmse_sum += rmse
        mae_list.append(mae)
        mape_net_list.append(mape_net)
        rmse_list.append(rmse)

        msg = "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE_net: {:.6f}"
        logger.info(msg.format(horizon_i + 1, mae, rmse, mape_net))

    mae = mae_sum / horizon
    rmse = rmse_sum / horizon
    mape = mape_net_sum / horizon

    logger.info('MAE: {:.3f} RMSE: {:.3f} MAPE: {:.5f}'.format(mae, rmse, mape))
    logger.info('{} end'.format(dataset_type))

    
    return {'mae': mae, 'mape': mape, 'rmse': rmse}


def eval(cfg, logger, device, log_dir): 
    # load dataset
    dataset = utils.load_dataset(cfg['data']['final_dataset_dir'], cfg['data']['batch_size'],
                                    test_batch_size=cfg['data']['test_batch_size'], scaler_axis=(0, 1, 2, 3))
    for k, v in dataset['test'].items():
        if hasattr(v, 'shape'):
            logger.info((k, v.shape))

    model = ODMixer(cfg, logger)
    pt_file = cfg['data']['pt_file']

    # single-gpu save, single-gpu load
    model.load_state_dict(torch.load(pt_file))
    
    # multi-gpu save, single-gpu load
    # loaded_dict = torch.load(pt_file)
    # model.load_state_dict({k.replace('module.', ''): v for k, v in loaded_dict.items()})

    scaler_od = dataset['scaler_od']
    scaler_od_torch = utils.StandardScaler_Torch(scaler_od.mean, scaler_od.std, device)
    logger.info('scaler_od.mean: {}, scaler_od.std: {}'.format(scaler_od.mean, scaler_od.std))

    if torch.cuda.device_count() > 1:
        print('Use {} gpus'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    evaluate(model, dataset, 'test', cfg, logger, log_dir, device, scaler_od_torch)


def main(args):
    cfg = read_cfg_file(args.config_filename)

    log_dir = _get_log_dir(cfg)
    log_level = cfg.get('log_level', 'INFO')

    logger = utils.get_logger(log_dir, __name__, 'info.log', log_level)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(log_dir)
    logger.info(cfg)

    eval(cfg, logger, device, log_dir)


if __name__ == '__main__':
    main(args)
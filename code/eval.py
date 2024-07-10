import argparse
import os

import numpy as np
import yaml
import time

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.optimizer import Optimizer

from models.contra import ContraNet, ContrastiveLoss, Loss, SiamMAE
from models.net import Net, GraphMAE
from lib import utils_data as utils
from lib import metrics
import wandb

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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


def init_weights(m):
    classname = m.__class__.__name__
    if type(classname) == nn.Linear:
        xavier_uniform_(m.weight.data)


def evaluate(model, dataset, dataset_type, cfg, logger, log_dir, device):
    logger.info('{} begin'.format(dataset_type))

    # data_loader = dataset[dataset_type + '_loader'].get_iterator()
    data_loader = dataset[dataset_type + '_loader']
    model.eval()
    y_od = []

    begin_time = time.perf_counter()
    for _, sequence in enumerate(data_loader):
        # sequence.to(device)
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

    logger.info('size: {}'.format(len(y_od)))
    # logger.info('gt shape: {}'.format(gt.shape))
    gt = scaler_od.inverse_transform(gt[:, :, :, :])
    y_od = scaler_od.inverse_transform(y_od[:gt.shape[0], :, :, :])

    y_od[y_od < 0] = 0
    mae = metrics.mae_np(y_od, gt)
    mape = metrics.mape_np(y_od, gt)
    rmse = metrics.rmse_np(y_od, gt)

    logger.info('MAE: {:.3f} RMSE: {:.3f} MAPE: {:.5f}'.format(mae, rmse, mape))
    logger.info('{} end'.format(dataset_type))

    return {'mae': mae, 'mape': mape, 'rmse': rmse}


def eval(cfg, logger, device, log_dir): 

    # load dataset
    dataset = utils.load_dataset(cfg['data']['final_dataset_dir'], cfg['data']['batch_size'],
                                    test_batch_size=cfg['data']['test_batch_size'], scaler_axis=(0, 1, 2, 3))
    for k, v in dataset['train'].items():
        if hasattr(v, 'shape'):
            logger.info((k, v.shape))

    model = Net(cfg, logger)
    pt_file = ''
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

    test_data = evaluate(model, dataset, 'test', cfg, logger, log_dir, device)


def main(args):
    cfg = read_cfg_file(args.config_filename)

    # log_dir = _get_log_dir(cfg, 'contra/')
    log_dir = _get_log_dir(cfg, 'final/')
    log_level = cfg.get('log_level', 'INFO')

    logger = utils.get_logger(log_dir, __name__, 'info.log', log_level)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(log_dir)
    logger.info(cfg)

    # contra_train(cfg, logger, device, log_dir)
    eval(cfg, logger, device, log_dir)


if __name__ == '__main__':
    main(args)

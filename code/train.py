import random
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
import torch.nn.functional as F
from models.net import Net
from models.MaskNet import MaskNet
from models.LinearNet import LinearNet

from lib import utils_data as utils
from lib import metrics
import wandb
from lib import data_load 

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from torch.optim.lr_scheduler import  _LRScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
os.environ['WANDB_MODE'] = 'offline'

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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


def train(cfg, logger, device, log_dir):
    writer = SummaryWriter('./runs/log')

    # some parameters
    max_grad_norm = cfg['train']['max_grad_norm']

    # load graph adj matrix
    graph_pkl_filename = cfg['data']['graph_pkl_filename']
    adj_mx = utils.load_graph_data(graph_pkl_filename)
    adj_mx = adj_mx / (adj_mx.sum(axis=1) + 1e-18)

    # load dataset
    dataset = data_load.load_dataset(cfg['data']['final_dataset_dir'], cfg['data']['batch_size'],
                                 test_batch_size=cfg['data']['test_batch_size'], scaler_axis=(0, 1, 2, 3))
    for k, v in dataset['train'].items():
        if hasattr(v, 'shape'):
            logger.info((k, v.shape))

    model = LinearNet(cfg, logger)
    model.apply(init_weights)

    
    scaler_od = dataset['scaler_od']
    scaler_od_torch = utils.StandardScaler_Torch(scaler_od.mean, scaler_od.std, device)
    logger.info('scaler_od.mean: {}, scaler_od.std: {}'.format(scaler_od.mean, scaler_od.std))

    if torch.cuda.device_count() > 1:
        print('Use {} gpus'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    adj_mx = torch.from_numpy(adj_mx).to(device)
    adj_mx = adj_mx.unsqueeze(dim=0).repeat(torch.cuda.device_count(), 1, 1)
    # 多GPU训练, 需要将adj_mx 转换为 [batch_size, num_nodes, num_nodes]

    criterion_l1 = nn.L1Loss(reduction='mean').to(device)
    criterion_smooth_l1 = nn.SmoothL1Loss(reduction='mean', beta=5).to(device)

    # criterion = NetLoss()
    lr = cfg['train']['base_lr']
    eps = cfg['train']['epsilon']
    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr, 'eps': eps},
    ])

    

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     patience=cfg['train']['lr_patience'],
                                                     factor=cfg['train']['lr_factor'],
                                                     min_lr=cfg['train']['min_learning_rate'],
                                                     eps=1.0e-12,
                                                     verbose=True)


    update = {'val_steady_count': 0, 'last_val_mae': 1e6, 'last_val_mape': 1e6}
    train_patience = 100

    for epoch in tqdm(range(cfg['train']['epochs'])):
        total_loss = 0.0
        total_od_loss = 0.0
        total_prev_od_loss = 0.0
        # dataset['train_loader'].shuffle()
        train_loader = dataset['train_loader']
        model.train()
        iter_cnt = 0
        begin_time = time.perf_counter()
        for _, sequence in enumerate(train_loader):
            optimizer.zero_grad()

            # sequence.to(device)
            for key, element in sequence.items():
                sequence[key] = element.float().to(device)

            prediction_od, prev_prediction_od = model(sequence)

            # Loss calculate
            prediction_od = scaler_od_torch.inverse_transform(prediction_od)
            labels_od = scaler_od_torch.inverse_transform(sequence['y_od'][:prediction_od.shape[0], :, :, :])
            od_loss = criterion_l1(prediction_od, labels_od)

            prev_prediction_od = scaler_od_torch.inverse_transform(prev_prediction_od)
            prev_labels_od = scaler_od_torch.inverse_transform(sequence['prev_y_od'][:prev_prediction_od.shape[0], :, :, :])
            prev_od_loss = criterion_l1(prev_prediction_od, prev_labels_od)
            

            loss = od_loss + prev_od_loss
            total_loss += loss.item()
            total_od_loss += od_loss.item()
            # total_prev_od_loss += prev_od_loss.item()

            loss.backward()
            # clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            iter_cnt += 1

        time_elapsed = time.perf_counter() - begin_time
        logger.info('Train Time: {}'.format(time_elapsed))
        logger.info('Epoch: {}, time: {}, total_loss: {}'.format(epoch, time_elapsed, total_loss / iter_cnt))


        eval_data = evaluate(model, dataset, 'val', cfg, logger, log_dir, device, adj_mx)

        test_data = evaluate(model, dataset, 'test', cfg, logger, log_dir, device, adj_mx)

        for category in ['od']:
            change_flag = False

            if eval_data['mae'] < update['last_val_mae']:
                update['last_val_mae'] = eval_data['mae']
                change_flag = True
            if eval_data['mape'] < update['last_val_mape']:
                update['last_val_mape'] = eval_data['mape']
                change_flag = True
            # if eval_data['rmse'] < update['last_val_rmse']:
            #     update['last_val_rmse'] = eval_data['rmse']
            #     change_flag = True

            if change_flag:
                update['val_steady_count'] = 0
            else:
                update['val_steady_count'] += 1   

            if (epoch + 1) % cfg['train']['save_every_n_epochs'] == 0:
                save_dir = log_dir
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                epoch_path = os.path.join(save_dir, 'epoch-{}.pt'.format(epoch + 1))
                config_path = os.path.join(save_dir, 'config-{}.yaml'.format(epoch + 1))
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), epoch_path)
                else:
                    torch.save(model.state_dict(), epoch_path)
                with open(config_path, 'w') as f:
                    from copy import deepcopy
                    save_cfg = deepcopy(cfg)
                    save_cfg['model']['save_path'] = epoch_path
                    f.write(yaml.dump(save_cfg, Dumper=Dumper))

        if update['val_steady_count'] >= train_patience:
            logger.info('early stopping')
            break

        scheduler.step(eval_data['mape'])
        

    writer.close()


def main(args):
    cfg = read_cfg_file(args.config_filename)

    log_dir = _get_log_dir(cfg, 'final/')
    log_level = cfg.get('log_level', 'INFO')

    logger = utils.get_logger(log_dir, __name__, 'info.log', log_level)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(log_dir)
    logger.info(cfg)

    train(cfg, logger, device, log_dir)


if __name__ == '__main__':
    main(args)

import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import copy
import numpy as np
import logging
import sys

def get_logger(log_dir, name, log_filename='info.log', log_level=logging.INFO, write_to_file=True):
    logger = logging.getLogger(name=name)
    logger.setLevel(log_level)

    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(file_formatter)
    if write_to_file:
        logger.addHandler(file_handler)

    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info('Log directory: {}'.format(log_dir))
    return logger


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ': ', e)
        raise
    return pickle_data


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class StandardScaler_Torch:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class MyDataset(Dataset):
    def __init__(self, data, batch_size):
        super().__init__()

        self.data = self.padding_data(data, batch_size)
        self.size = len(self.data['od'])

    def padding_data(self, sequence, batch_size):
        new_sequence = copy.deepcopy(sequence)
        num_padding = (batch_size - (len(sequence['od']) % batch_size)) % batch_size
        padding = {}
        for key in sequence.keys():
            padding[key] = np.repeat(sequence[key][-1:], num_padding, axis=0)
            new_sequence[key] = np.concatenate([sequence[key], padding[key]], axis=0)

        return new_sequence

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        batch = {
            'od': self.data['od'][idx],
            'y_od': self.data['y_od'][idx],
            'prev_od': self.data['prev_od'][idx],
            'prev_y_od': self.data['prev_y_od'][idx]
        }

        return batch


def load_dataset(dataset_dir, batch_size, test_batch_size, scaler_axis=(0, 1, 2, 3)):
    data = {'train': {}, 'test': {}, 'val': {}}

    for category in ['train', 'val', 'test']:
        category_data = load_pickle(os.path.join(dataset_dir, category + '.pkl'))
        data[category] = category_data
        for key in data[category]:
            data[category][key] = data[category][key].astype(float)

    scaler_od = StandardScaler(mean=data['train']['od'].mean(axis=scaler_axis),
                               std=data['train']['od'].std(axis=scaler_axis))
    data['scaler_od'] = scaler_od
    
    for category in ['train', 'val', 'test']:
        data[category]['od'] = scaler_od.transform(data[category]['od'])
        data[category]['y_od'] = scaler_od.transform(data[category]['y_od'])
        data[category]['prev_od'] = scaler_od.transform(data[category]['prev_od'])
        data[category]['prev_y_od'] = scaler_od.transform(data[category]['prev_y_od'])

    for category in ['train']:
        data[category + '_dataset'] = MyDataset(data[category], batch_size)
        data[category + '_loader'] = DataLoader(data[category + '_dataset'], batch_size, num_workers=10, shuffle=True, pin_memory=True)
    for category in ['val', 'test']:
        data[category + '_dataset'] = MyDataset(data[category], test_batch_size)
        data[category + '_loader'] = DataLoader(data[category + '_dataset'], test_batch_size, num_workers=10, shuffle=False, pin_memory=True)
    return data

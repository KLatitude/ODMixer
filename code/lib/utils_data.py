import copy
import logging
import os
import pickle
import sys
import numpy as np
import math
import torch


class DataLoader(object):
    def __init__(self, batch_size, sequence, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.cur_idx = 0

        new_sequence = copy.deepcopy(sequence)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(sequence['incomplete_od']) % batch_size)) % batch_size
            padding = {}
            for key in sequence.keys():
                padding[key] = np.repeat(sequence[key][-1:], num_padding, axis=0)
                new_sequence[key] = np.concatenate([sequence[key], padding[key]], axis=0)

        self.sequence = new_sequence
        self.size = len(self.sequence['incomplete_od'])

        self.num_batch = int(self.size // self.batch_size)

        # self.od = od
        # self.unfinished = unfinished
        # self.incomplete_od = incomplete_od
        # self.time_day = time_day
        # self.time = time

        # self.num_batch = math.ceil(self.size / self.batch_size)

    def shuffle(self):
        permutation = np.random.permutation(self.size)

        for key in self.sequence.keys():
            per_val = self.sequence[key][permutation]
            self.sequence[key] = per_val

    def get_iterator(self):
        self.cur_idx = 0

        def _wrapper():
            while self.cur_idx < self.num_batch:
                start_idx = self.batch_size * self.cur_idx
                end_idx = min(self.size, self.batch_size * (self.cur_idx + 1))

                data = {}
                for key in self.sequence.keys():
                    data[key] = self.sequence[key][start_idx:end_idx, ...]

                # od_i = self.od[start_idx:end_idx, ...]
                # unfinished_i = self.unfinished[start_idx:end_idx, ...]
                # incomplete_od_i = self.incomplete_od[start_idx:end_idx, ...]
                # time_day_i = self.time_day[start_idx:end_idx, ...]
                # time_i = self.time[start_idx:end_idx, ...]
                #
                # data = {'od': od_i, 'unfinished': unfinished_i, 'incomplete_od': incomplete_od_i,
                #         'time_day': time_day_i, 'time': time_i}

                yield SimpleBatch(data)
                self.cur_idx += 1
        return _wrapper()


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


class SimpleBatch(dict):
    def to(self, device):
        for key, element in self.items():
            self[key] = torch.from_numpy(element).float().to(device)
        return self


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


# def load_dataset(dataset_dir, batch_size, test_batch_size=None, scaler_axis=(0, 1, 2, 3), **kwargs):
#     data = {'train': {}, 'test': {}, 'val': {}}

#     for category in ['train', 'val', 'test']:
#         category_data = load_pickle(os.path.join(dataset_dir, category + '.pkl'))
#         data[category] = category_data
#         for key in data[category]:
#             data[category][key] = data[category][key].astype(float)

#     scaler_od = StandardScaler(mean=data['train']['incomplete_od'].mean(axis=scaler_axis),
#                                std=data['train']['incomplete_od'].std(axis=scaler_axis))

#     for category in ['train', 'val', 'test']:
#         shape = data[category]['incomplete_od'].shape
#         unf = np.repeat(data[category]['unfinished'][:, :, :, np.newaxis], shape[-1], axis=-1)

#         uod_yest = data[category]['unfinished_od_yest']
#         uod_yest = uod_yest / (np.sum(uod_yest, axis=-1, keepdims=True) + 1e-18)
#         data[category]['unfinished_od_yest'] = unf * uod_yest

#         uod_week = data[category]['unfinished_od_week']
#         uod_week = uod_week / (np.sum(uod_week, axis=-1, keepdims=True) + 1e-18)
#         data[category]['unfinished_od_week'] = unf * uod_week


#     for category in ['train', 'val', 'test']:
#         data[category]['incomplete_od'] = scaler_od.transform(data[category]['incomplete_od'])
#         data[category]['unfinished_od_yest'] = scaler_od.transform(data[category]['unfinished_od_yest'])
#         data[category]['unfinished_od_week'] = scaler_od.transform(data[category]['unfinished_od_week'])
#         data[category]['y_od'] = scaler_od.transform(data[category]['y_od'])

#         data[category]['prev_od'] = scaler_od.transform(data[category]['prev_od'])
#         data[category]['prev_y_od'] = scaler_od.transform(data[category]['prev_y_od'])
        

#     data['scaler_od'] = scaler_od

#     for category in ['train']:
#         data[category + '_loader'] = DataLoader(batch_size, data[category])
#     for category in ['val', 'test']:
#         data[category + '_loader'] = DataLoader(test_batch_size, data[category])

#     """"
#     for category in ['train']:
#         category_data = load_pickle(os.path.join(dataset_dir, category + '.pkl'))
#         # print(type(category['od']))
#         data['od_' + category] = category_data['od']
#         data['unfinished_' + category] = category_data['unfinished']
#         data['incomplete_od_' + category] = category_data['incomplete_od']
#         data['time_day_' + category] = category_data['time_day']
#         data['time_' + category] = category_data['time']

#     scaler_od = StandardScaler(mean=data['od_train'].mean(axis=scaler_axis), std=data['od_train'].std(axis=scaler_axis))

#     for category in ['train']:
#         data['od_' + category] = scaler_od.transform(data['od_' + category])
#         data['unfinished_' + category] = scaler_od.transform(data['unfinished_' + category])
#         data['incomplete_od_' + category] = scaler_od.transform(data['incomplete_od_' + category])

#     data['train_loader'] = DataLoader(batch_size=batch_size,
#                                       od=data['od_train'],
#                                       unfinished=data['unfinished_train'],
#                                       incomplete_od=data['incomplete_od_train'],
#                                       time_day=data['time_day_train'],
#                                       time=data['time_train'])

#     data['scaler_od'] = scaler_od
#     """
#     return data


def load_graph_data(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx.astype(np.float32)


def collate_wrapper(sequence, device):
    res = {}
    for key, val in sequence:
        res[key] = torch.tensor(val, dtype=torch.float, device=device)

import torch
from torch import nn
from torch.nn import functional as F


class SingleInteractModule(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, dropout=0.1):
        super(SingleInteractModule, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, output_dim)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, output_dim)
        )
        self.conv1d = nn.Conv1d(2, 1, 1)

    def forward(self, x, y):
        shape = x.shape
        x_to_y, y_to_x = x.reshape(shape[0], -1), y.reshape(shape[0], -1)
        z = torch.cat((x_to_y.unsqueeze(-1), y_to_x.unsqueeze(-1)), -1)
        z = torch.squeeze(self.conv1d(z.permute(0, 2, 1)))
        z = z.reshape(shape)
        gate = torch.sigmoid(self.linear1(z))
        output = self.linear2(x) * gate
        return output + x
    

class BTL(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout=0.1):
        super().__init__()

        self.up_interact = SingleInteractModule(input_dim, hid_dim, input_dim, dropout)
        self.down_interact = SingleInteractModule(input_dim, hid_dim, input_dim, dropout)

    def forward(self, x, y):
        output_x = self.up_interact(x, y)
        output_y = self.down_interact(y, x)
        return output_x, output_y


class MixerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        y = self.ffn(x)
        return y


class CM(nn.Module):
    def __init__(self, cfg, logger):
        super().__init__()

        self.input_seq = cfg['model']['input_seq']
        self.input_dim = cfg['model']['input_dim']
        self.hid_dim = cfg['model']['hidden_dim']
        self.num_nodes = cfg['model']['num_nodes']
        self.dropout = cfg['model']['dropout']
        
        self.mixer = MixerLayer(self.hid_dim, 2 * self.hid_dim, self.hid_dim, self.dropout)
        self.ln = nn.LayerNorm([self.num_nodes, self.input_dim, self.hid_dim])

    def forward(self, x):
        b, n, m, d = x.shape
        feat = self.mixer(x)
        output = self.ln(feat + x)
        return output


class MM(nn.Module):
    def __init__(self, cfg, logger):
        super().__init__()

        self.input_seq = cfg['model']['input_seq']
        self.input_dim = cfg['model']['input_dim']
        self.hid_dim = cfg['model']['hidden_dim']
        self.num_nodes = cfg['model']['num_nodes']
        self.dropout = cfg['model']['dropout']
        self.origin_mixer = MixerLayer(self.num_nodes, 2 * self.hid_dim, self.num_nodes, self.dropout)
        self.des_mixer = MixerLayer(self.num_nodes, 2 * self.hid_dim, self.num_nodes, self.dropout)
        self.ln = nn.LayerNorm([self.num_nodes, self.input_dim, self.hid_dim])

    def forward(self, x):
        b, n, m, d = x.shape

        origin_feat = x.permute(0, 1, 3, 2)
        origin_feat = self.origin_mixer(origin_feat)
        origin_feat = origin_feat.permute(0, 1, 3, 2)

        des_feat = x.permute(0, 2, 3, 1)
        des_feat = self.des_mixer(des_feat)
        des_feat = des_feat.permute(0, 3, 1, 2)

        feat = origin_feat + des_feat
        output = self.ln(feat + x)
        return output


class ODIM(nn.Module):
    def __init__(self, cfg, logger):
        super().__init__()
        
        self.cfg = cfg
        self.logger = logger

        self.cm = CM(cfg, logger)
        self.mm = MM(cfg, logger)

    def forward(self, x):
        h = self.cm(x)
        output = self.mm(h)
        return output


class ODMixer(nn.Module):
    def __init__(self, cfg, logger):
        super().__init__()

        self.cfg = cfg
        self.logger = logger

        self.input_seq = cfg['model']['input_seq']
        self.input_dim = cfg['model']['input_dim']
        self.hid_dim = cfg['model']['hidden_dim']
        self.num_nodes = cfg['model']['num_nodes']
        self.layer_nums = cfg['model']['layer_nums']
        self.dropout = cfg['model']['dropout']

        # OD pair view embedding layer
        self.emb_layer = nn.Linear(self.input_seq, self.hid_dim)

        self.encoder_layer = nn.ModuleList([
            ODIM(cfg, logger) for _ in range(self.layer_nums)
        ])
        self.trend_layer = nn.ModuleList([
            BTL(self.hid_dim, self.hid_dim, self.dropout) for _ in range(self.layer_nums)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.PReLU(),
            nn.Linear(self.hid_dim // 2, 1)
        )

    
    def forward(self, sequence):
        od = sequence['od']
        prev_od = sequence['prev_od']

        b, t, n, m = od.shape
        od = od.permute(0, 2, 3, 1)
        od_feat = self.emb_layer(od)

        
        prev_od = prev_od.permute(0, 2, 3, 1)
        prev_od_feat = self.emb_layer(prev_od)

        for i in range(self.layer_nums):
            od_feat = self.encoder_layer[i](od_feat)
            prev_od_feat = self.encoder_layer[i](prev_od_feat)

            prev_od_feat, od_feat = self.trend_layer[i](prev_od_feat, od_feat)

        od_output = self.output_layer(od_feat)
        od_output = od_output.reshape(b, n, m, -1).permute(0, 3, 1, 2)

        prev_output = self.output_layer(prev_od_feat)
        prev_output = prev_output.reshape(b, n, m, -1).permute(0, 3, 1, 2)
        
        return od_output, prev_output
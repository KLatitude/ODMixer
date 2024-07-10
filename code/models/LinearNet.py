import torch
from torch import nn
from torch.nn import functional as F
from models.st_layer import InformationCoSelectModule
import math

class MixerLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MixerLayer, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        y = self.ffn(x)
        return y

class PairMixerLayer(nn.Module):
        def __init__(self, cfg, logger):
            super(PairMixerLayer, self).__init__()

            self.cfg = cfg
            self.logger = logger

            self.input_seq = cfg['model']['input_seq']
            self.input_dim = cfg['model']['input_dim']
            self.hid_dim = cfg['model']['hidden_dim']
            self.num_nodes = cfg['model']['num_nodes']
            self.dropout = cfg['model']['dropout']

            self.token_num = self.num_nodes * self.input_dim

            self.channel_ln = nn.LayerNorm([self.token_num, self.hid_dim])
            self.channel_mixer = MixerLayer(self.hid_dim, 2 * self.hid_dim, self.hid_dim, self.dropout)
            
            self.token_ln = nn.LayerNorm([self.hid_dim, self.token_num])
            self.orgin_mixer = MixerLayer(self.num_nodes, 2 * self.hid_dim, self.num_nodes, self.dropout)
            self.des_mixer = MixerLayer(self.num_nodes, 2 * self.hid_dim, self.num_nodes, self.dropout)


        def forward(self, x):         
            b, n, m, d = x.shape
            channel_output = self.channel_mixer(x) + x
            channel_output = channel_output.reshape(b, n * m, -1)
            y = self.channel_ln(channel_output).reshape(b, n, m, -1)

            orgin_feat = y.permute(0, 1, 3, 2)
            orgin_feat = self.orgin_mixer(orgin_feat)
            orgin_feat = orgin_feat.permute(0, 1, 3, 2).reshape(b, n * m, -1).permute(0, 2, 1)

            des_feat = y.permute(0, 2, 3, 1)
            des_feat = self.des_mixer(des_feat)
            des_feat = des_feat.permute(0, 3, 1, 2).reshape(b, n * m, -1).permute(0, 2, 1)
            
            final_feat = orgin_feat + des_feat

            y = y.reshape(b, n * m, -1).permute(0, 2, 1)
            output = self.token_ln(final_feat + y)
            output = output.permute(0, 2, 1).reshape(b, n, m, -1)
            return output

class AttentionLayer(nn.Module):
    def __init__(self, cfg, logger):
        super(AttentionLayer, self).__init__()

        self.cfg = cfg
        self.logger = logger

        self.input_dim = cfg['model']['input_dim']
        self.hid_dim = cfg['model']['hidden_dim']    
        self.num_nodes = cfg['model']['num_nodes']
        self.dropout = cfg['model']['dropout']
        self.token_num = self.num_nodes * self.input_dim
        self.h = 2
        self.d_k = self.hid_dim // self.h

        self.linears = nn.ModuleList([
            nn.Linear(self.hid_dim, self.hid_dim) for _ in range(3)
        ])
        self.dropout_layer = nn.Dropout(self.dropout)
        self.ln = nn.LayerNorm(self.hid_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid_dim, self.hid_dim)
        )

    def attention(self, q, k, v):
        B, n_head, n_query, d_k = q.shape
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout_layer(attn)
        out = torch.matmul(attn, v)

        return out
    
    def forward(self, x):
        b, n, m, d = x.shape
        q, k, v = [l(y).reshape(b, n * m, self.h, self.d_k).permute(0, 2, 1, 3) for l, y in zip(self.linears, (x, x, x))]

        y = self.attention(q, k, v)
        y = y.permute(0, 2, 1, 3).reshape(b, n, m, self.h * self.d_k)

        x = self.ln(x + y)

        z = self.ffn(x)
        x = self.ln(z + x)
        return x


    
class LinearNet(nn.Module):
    def __init__(self, cfg, logger):
        super(LinearNet, self).__init__()

        self.cfg = cfg
        self.logger = logger

        self.input_seq = cfg['model']['input_seq']
        self.input_dim = cfg['model']['input_dim']
        self.hid_dim = cfg['model']['hidden_dim']
        self.num_nodes = cfg['model']['num_nodes']
        self.layer_nums = cfg['model']['layer_nums']
        self.dropout = cfg['model']['dropout']

        # OD pair view
        self.emb_layer = nn.Linear(self.input_seq, self.hid_dim)

        self.encoder_layer = nn.ModuleList([
            PairMixerLayer(cfg, logger) for _ in range(self.layer_nums)
        ])

        # self.encoder_layer = nn.ModuleList([
        #     AttentionLayer(cfg, logger) for _ in range(self.layer_nums)
        # ])
        self.trend_layer = nn.ModuleList([
            InformationCoSelectModule(self.hid_dim, self.hid_dim) for _ in range(self.layer_nums)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.PReLU(),
            nn.Linear(self.hid_dim // 2, 1)
        )

    
    def forward(self, sequence):
        # iod = sequence['incomplete_od']
        # b, t, n, m = iod.shape
        
        # uod = (sequence['unfinished_od_yest'] + sequence['unfinished_od_week']) / 2
        # od = iod + uod

        od = sequence['od']
        b, t, n, m = od.shape

        od = od.permute(0, 2, 3, 1)
        od_feat = self.emb_layer(od)

        prev_od = sequence['prev_od']
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
    
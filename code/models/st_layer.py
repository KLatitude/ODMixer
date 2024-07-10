import torch
from torch import nn
from models.graph_module import GCN
from models.transformers import MultiHeadAttention, NormalTransformerBlock
import math
from torch.nn import functional as F

class SingleInteractModule(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(SingleInteractModule, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, output_dim)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, output_dim)
        )
        self.conv1d = nn.Conv1d(2, 1, 1)

    def forward(self, x, y):
        shape = x.shape
        # x_to_y, y_to_x = x.clone(), y.clone()
        # x_to_y, y_to_x = x_to_y.reshape(shape[0], -1), y_to_x.reshape(shape[0], -1)
        x_to_y, y_to_x = x.reshape(shape[0], -1), y.reshape(shape[0], -1)
        z = torch.cat((x_to_y.unsqueeze(-1), y_to_x.unsqueeze(-1)), dim=-1)
        z = torch.squeeze(self.conv1d(z.permute(0, 2, 1)))
        z = z.reshape(shape)
        gate = torch.sigmoid(self.linear1(z))
        output = self.linear2(x) * gate
        return output + x
    

class InformationCoSelectModule(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(InformationCoSelectModule, self).__init__()

        self.up_interact = SingleInteractModule(input_dim, hid_dim, input_dim)
        self.down_interact = SingleInteractModule(input_dim, hid_dim, input_dim)

    def forward(self, x, y):
        output_x = self.up_interact(x, y)
        output_y = self.down_interact(y, x)
        return output_x, output_y


class SpatioTemporalSimModule(nn.Module):
    def __init__(self, cfg, logger):
        super(SpatioTemporalSimModule, self).__init__()

        self.cfg = cfg
        self.logger = logger

        self.input_seq = cfg['model']['input_seq']
        self.hid_dim = cfg['model']['hidden_dim']
        self.num_nodes = cfg['model']['num_nodes']
        self.dropout = cfg['model']['dropout']
        self.head_num = cfg['model']['head']

        self.shift_gcn = GCN(self.hid_dim, self.hid_dim, self.hid_dim)
        self.shift_transformer = MultiHeadAttention(self.hid_dim, self.hid_dim, self.hid_dim, self.hid_dim,
                                                    self.hid_dim, self.head_num, self.dropout, bias=True)
        self.shift_nodes = None
        self.shift_steps = None
        self.shift_rate = cfg['model']['shift_rate']
        self.shift_node_num = 0

        self.ln = nn.LayerNorm([self.num_nodes, self.hid_dim])

    def node_shift(self, x):
        b, t, n, d = x.shape
        y = x.clone()
        shift_node_num = int(n * self.shift_rate)
        self.shift_node_num = shift_node_num

        perm = torch.randperm(n, device=x.device)
        self.shift_nodes = perm[:shift_node_num]
        self.shift_steps = torch.randint(low=1, high=t, size=(shift_node_num,), device=x.device)

        shift_matrix = torch.arange(0, t, 1, device=x.device)
        shift_matrix = shift_matrix.unsqueeze(0).repeat(n, 1)
        shift_steps = self.shift_steps.unsqueeze(1).repeat(1, t)
        shift_matrix[self.shift_nodes, :] = (shift_matrix[self.shift_nodes, :] - shift_steps) % t

        shift_matrix = shift_matrix.unsqueeze(2).repeat(1, 1, d)
        shift_matrix = shift_matrix.unsqueeze(0).repeat(b, 1, 1, 1)
        y = y.permute(0, 2, 1, 3)
        y = torch.gather(y, dim=2, index=shift_matrix)
        y = y.permute(0, 2, 1, 3)
        return y

    def node_back_shift(self, x):
        b, t, n, d = x.shape
        y = x.clone()
        shift_matrix = torch.arange(0, t, 1, device=x.device)
        shift_matrix = shift_matrix.unsqueeze(0).repeat(n, 1)
        shift_steps = self.shift_steps.unsqueeze(1).repeat(1, t)
        shift_matrix[self.shift_nodes, :] = (shift_matrix[self.shift_nodes, :] + shift_steps) % t

        shift_matrix = shift_matrix.unsqueeze(2).repeat(1, 1, d)
        shift_matrix = shift_matrix.unsqueeze(0).repeat(b, 1, 1, 1)
        y = y.permute(0, 2, 1, 3)
        y = torch.gather(y, dim=2, index=shift_matrix)
        y = y.permute(0, 2, 1, 3)
        return y
    
    def forward(self, x, time_feat, adj_mx):
        b, t, n, d = x.shape
        z = self.shift_gcn(x, adj_mx)

        time_feat = torch.unsqueeze(time_feat, dim=2).repeat(1, 1, self.num_nodes, 1)
        z = z + time_feat

        qkv = z.reshape(b, t * n, d)
        feat = self.shift_transformer(qkv, qkv, qkv)
        output_feat = feat.reshape(b, t, n, d)

        return self.ln(output_feat + x)


class SpatioTemporalSepModule(nn.Module):
    def __init__(self, cfg, logger):
        super(SpatioTemporalSepModule, self).__init__()

        self.cfg = cfg
        self.logger = logger

        self.input_seq = cfg['model']['input_seq']
        self.hid_dim = cfg['model']['hidden_dim']
        self.num_nodes = cfg['model']['num_nodes']
        self.dropout = cfg['model']['dropout']
        self.head_num = cfg['model']['head']

        self.linear = nn.Linear(self.hid_dim, 3 * self.hid_dim)

        self.gcn = GCN(self.hid_dim // 2, self.hid_dim, self.hid_dim // 2)
        self.spa_transformer = MultiHeadAttention(self.hid_dim // 2, self.hid_dim // 2, self.hid_dim // 2,
                                                  self.hid_dim // 2, self.hid_dim // 2, self.head_num, self.dropout,
                                                  bias=True)

        self.time_emb = nn.Linear(self.hid_dim, self.hid_dim // 2)
        self.tem_transformer = MultiHeadAttention(self.hid_dim // 2, self.hid_dim // 2, self.hid_dim // 2,
                                                  self.hid_dim // 2, self.hid_dim // 2, self.head_num, self.dropout,
                                                  bias=True)

        self.interact_layer = InformationCoSelectModule(self.hid_dim // 2, self.hid_dim // 2)
        self.ln = nn.LayerNorm([self.num_nodes, self.hid_dim])

    def forward(self, x, time_feat, adj_mx):
        q, k, v = self.linear(x).chunk(3, dim=3)
        q_s, q_t = q.chunk(2, dim=3)
        k_s, k_t = k.chunk(2, dim=3)
        v_s, v_t = v.chunk(2, dim=3)

        shape = q_s.shape
        q_s = self.gcn(q_s, adj_mx)
        k_s = self.gcn(k_s, adj_mx)
        v_s = self.gcn(v_s, adj_mx)
        q_s = q_s.reshape(-1, self.num_nodes, self.hid_dim // 2)
        k_s = k_s.reshape(-1, self.num_nodes, self.hid_dim // 2)
        v_s = v_s.reshape(-1, self.num_nodes, self.hid_dim // 2)
        y_s = self.spa_transformer(q_s, k_s, v_s)
        y_s = y_s.reshape(shape)

        time_feat = self.time_emb(time_feat)
        time_feat = torch.unsqueeze(time_feat, dim=2).repeat(1, 1, self.num_nodes, 1)
        q_t = q_t + time_feat
        k_t = k_t + time_feat
        v_t = v_t + time_feat
        q_t = q_t.permute(0, 2, 1, 3).reshape(-1, shape[1], self.hid_dim // 2)
        k_t = k_t.permute(0, 2, 1, 3).reshape(-1, shape[1], self.hid_dim // 2)
        v_t = v_t.permute(0, 2, 1, 3).reshape(-1, shape[1], self.hid_dim // 2)
        y_t = self.tem_transformer(q_t, v_t, k_t)
        y_t = y_t.reshape(shape[0], self.num_nodes, shape[1], self.hid_dim // 2).permute(0, 2, 1, 3).reshape(shape)

        output_s, output_t = self.interact_layer(y_s, y_t)

        output = torch.cat((output_s, output_t), dim=3)
        return self.ln(output + x)


class STLayer(nn.Module):
    def __init__(self, cfg, logger):
        super(STLayer, self).__init__()

        self.cfg = cfg
        self.logger = logger

        self.num_nodes = cfg['model']['num_nodes']
        self.hid_dim = cfg['model']['hidden_dim']

        self.ln = nn.LayerNorm([self.num_nodes, self.hid_dim])
        self.st_sep_layer = SpatioTemporalSepModule(cfg, logger)
        self.st_sim_layer = SpatioTemporalSimModule(cfg, logger)
        self.st_interact_layer = InformationCoSelectModule(self.hid_dim, self.hid_dim)

        self.output_layer = nn.Sequential(
            nn.Linear(2 * self.hid_dim, self.hid_dim),
            nn.PReLU(),
            nn.Linear(self.hid_dim, self.hid_dim)
        )

    def forward(self, x, time_feat, adj_mx):
        y = x.clone()
        output_sep = self.st_sep_layer(x, time_feat, adj_mx)
        output_sim = self.st_sim_layer(x, time_feat, adj_mx)

        after_sep, after_sim = self.st_interact_layer(output_sep, output_sim)
        final_output = torch.cat((after_sep, after_sim), dim=3)
        final_output = self.output_layer(final_output)
        
        return self.ln(final_output + y)


class ShiftAttentionLayer(nn.Module):
    def __init__(self, cfg, logger):
        super(ShiftAttentionLayer, self).__init__()

        self.hid_dim = cfg['model']['hidden_dim']
        self.num_nodes = cfg['model']['num_nodes']
        self.input_seq = cfg['model']['input_seq']

        self.ln = nn.LayerNorm([self.num_nodes, self.hid_dim])

        self.w_q = nn.ModuleList([
            nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.input_seq)
        ])
        self.w_k = nn.ModuleList([
            nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.input_seq)
        ])
        self.w_v = nn.ModuleList([
            nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.input_seq)
        ])
        
        # self.ww_q = nn.ModuleList([
        #     nn.Linear(self.num_nodes, self.hid_dim) for _ in range(self.input_seq - 1)
        # ])
        # self.ww_k = nn.ModuleList([
        #     nn.
        # ])
        self.ffn = nn.Sequential(
            nn.Linear(self.hid_dim, 2 * self.hid_dim),
            nn.PReLU(),
            nn.Linear(2 * self.hid_dim, self.hid_dim)
        )

    def forward(self, x, time_feat, adj_mx):
        y = x.clone()
        time_feat = torch.unsqueeze(time_feat, dim=2).repeat(1, 1, self.num_nodes, 1)
        x = x + time_feat
        q = []
        k = []
        v = []
        for i in range(self.input_seq):
            q.append(self.w_q[i](x[:, i, :, :]))
            k.append(self.w_k[i](x[:, i, :, :]))
            v.append(self.w_v[i](x[:, i, :, :]))
        
        d = self.hid_dim
        weights = []
        for i in range(self.input_seq):
            score_t = torch.bmm(q[i], k[i].transpose(1, 2)) / math.sqrt(d)
            score_t = F.softmax(score_t, dim=-1)
            weights.append(score_t)
        
        
        ud = weights[0].shape[-1]
        updatad_weights = []
        updatad_weights.append(weights[0])
        for i in range(1, self.input_seq):
            score_weight = torch.bmm(weights[i - 1], weights[i].transpose(1, 2)) / math.sqrt(ud)
            score_weight = F.softmax(score_weight, dim=-1)
            new_weight = torch.bmm(score_weight, weights[i])
            new_weight = new_weight + weights[i]
            # new_weight = F.softmax(new_weight, dim=-1)
            updatad_weights.append(new_weight)
        
        output = []
        for i in range(self.input_seq):
            output_t = torch.bmm(updatad_weights[i], v[i])
            output.append(output_t)
        final_output = torch.stack(output, dim=1).to(x.device)
        
        ffn_input = self.ln(final_output + y)
        ffn_feat = self.ffn(ffn_input)
        ffn_output = self.ln(ffn_feat + ffn_input)
        return ffn_output


import torch
from torch import nn
import torch.nn.functional as F


"""
class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        # print(input_dim // num_heads, output_dim // num_heads)
        assert input_dim % num_heads == 0 and output_dim % num_heads == 0

        self.input_dim = input_dim // num_heads
        self.output_dim = output_dim // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        self.W = nn.ModuleList([nn.Linear(self.input_dim, self.output_dim) for _ in range(num_heads)])
        self.a = nn.ModuleList([nn.Linear(2 * self.output_dim, 1) for _ in range(num_heads)])

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.activation = SwiGLU(input_dim, output_dim)

    def forward(self, x, adj):
        batch, num_nodes, _ = x.shape

        x = x.reshape(batch, num_nodes, self.num_heads, -1)
        h = torch.cat([self.W[i](x[:, :, i:i + 1, :]) for i in range(self.num_heads)], dim=2)
        attention = [self.a[i](torch.cat([h[:, :, i, :].repeat(1, 1, num_nodes).view(batch, num_nodes * num_nodes, -1),
                                          h[:, :, i, :].repeat(1, num_nodes, 1)], dim=-1)).squeeze(2) for i in
                     range(self.num_heads)]
        # (head, batch, num_nodes, num_nodes)
        attention = torch.cat(attention, dim=0).view(batch * self.num_heads, num_nodes, num_nodes)
        attention = self.leakyrelu(attention)

        mask = -9e15 * torch.ones_like(attention)
        # attention = torch.where(adj > 0, attention, mask)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout)

        h = h.permute(0, 2, 1, 3).reshape(batch * self.num_heads, num_nodes, -1)
        y = torch.matmul(attention, h).reshape(batch, self.num_heads, num_nodes, -1)
        # print(y.shape)
        y = y.permute(0, 2, 1, 3).reshape(batch, num_nodes, -1)

        return self.activation(y)


class GAT(nn.Module):
    def __init__(self, cfg, logger, input_dim, output_dim):
        super(GAT, self).__init__()

        self.layer_nums = cfg['model']['layer_nums']
        hid_dim = cfg['model']['hidden_dim']
        head_nums = cfg['model']['head']

        # print(hid_dim, head_nums)
        self.gat_layers = nn.ModuleList([
            GATLayer(input_dim, output_dim, head_nums) for _ in range(self.layer_nums)
        ])

    def forward(self, x, adj):
        for layer in self.gat_layers:
            x = layer(x, adj)
        return x
"""


class GCN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(GCN, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, output_dim)
        )

    def forward(self, x, adj_matrix):
        # self.logger.info('x shape: {} adj_shape: {}'.format(x.shape, adj_matrix.shape))
        # y = self.mlp(torch.matmul(adj_matrix, x))
        # self.logger.info('shape: {}'.format(y.shape))
        return self.mlp(torch.matmul(adj_matrix, x))

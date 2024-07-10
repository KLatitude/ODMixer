import torch
from torch import nn
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
    

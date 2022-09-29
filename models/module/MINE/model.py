import torch 
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        # self._init_weights()

    def forward(self, input):
        output = F.leaky_relu(self.fc1(input))
        output = F.leaky_relu(self.fc2(output))
        output = F.leaky_relu(self.fc3(output))
        output = torch.sigmoid(self.out(output))
        return output

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
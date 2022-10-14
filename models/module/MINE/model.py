import torch 
import torch.nn as nn
import torch.nn.functional as F


class M(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # print(input_dim)
        hidden_dim = [4096, 512, 64, 64, 10]
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc4 = nn.Linear(hidden_dim[2], hidden_dim[3])
        self.fc5 = nn.Linear(hidden_dim[3], hidden_dim[4])
        self.out = nn.Linear(hidden_dim[4], 1)
        # self._init_weights()

    def forward(self, input):
        output = F.leaky_relu(self.fc1(input))
        output = F.leaky_relu(self.fc2(output))
        output = F.leaky_relu(self.fc3(output))
        output = F.leaky_relu(self.fc4(output))
        output = F.leaky_relu(self.fc5(output))
        output = torch.sigmoid(self.out(output))
        return output

    # def _init_weights(self):
    #     nn.init.normal_(self.fc1.weight, std=0.02)
    #     nn.init.constant_(self.fc1.bias, 0)
    #     nn.init.normal_(self.fc2.weight, std=0.02)
    #     nn.init.constant_(self.fc2.bias, 0)
    #     nn.init.normal_(self.fc3.weight, std=0.02)
    #     nn.init.constant_(self.fc3.bias, 0)
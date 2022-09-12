import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    input:
        x:
    outputs:
        e
    """

    def __init__(self, e_dim=100, nz="tanh"):
        super(Encoder, self).__init__()
        # self.input_h, self.input_w, self.input_dep = input_shape
        self.hidden_dim = 16 * 24 * 24

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        self.enc_e = nn.Linear(self.hidden_dim, e_dim)

        if nz == "tanh":
            self.nz = nn.Tanh()
        elif nz == "sigmoid":
            self.nz = nn.Sigmoid()
        elif nz == "relu":
            self.nz = nn.ReLU()

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))

        bs, dim, h, w = x.size()
        x = x.view(bs, -1)
        # print(x.size())

        e = self.nz(self.enc_e(x))

        return e


if __name__ == "__main__":
    import numpy as np

    enc = Encoder()
    x = torch.randn(1000, 59)
    e1, e2 = enc(x)
    print(e1.size(), e2.size())
    model_parameters = filter(lambda p: p.requires_grad, enc.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable params:", num_params)

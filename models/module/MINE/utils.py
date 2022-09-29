import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import numpy as np



def mi_criterion(a, z, mine_net):
    # print(a.size())
    # print(z.size())
    # print('adsjlald')
    index, joint = sample_batch_joint(z, a)
    marginal = sample_batch_marginal(z, a, index)

    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_loss = -(torch.mean(t) - torch.log(torch.mean(et)))
    return mi_loss

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def sample_batch_joint(x, y):
    index = np.random.choice(range(x.shape[0]), size=x.shape[0], replace=False)
    # print(index)
    batch = torch.cat([x[index], y[index]], dim=1)
    return index, batch


def sample_batch_marginal(x, y, y_index):
    x_marginal_index = np.random.choice(range(y.shape[0]), size=x.shape[0], replace=False)
    batch = torch.cat([x[x_marginal_index], y[y_index]], dim=1)
    return batch

    
# def sample_batch(data, batch_size=100, sample_mode='joint'):
#     if sample_mode == 'joint':
#         index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         batch = data[index]
#     else:
#         joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         batch = np.concatenate([data[joint_index][:,0].reshape(-1,1),
#                                          data[marginal_index][:,1].reshape(-1,1)],
#                                        axis=1)
#     return batch
from asyncore import file_dispatcher
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from models.module.basenet import ResNet18 
from torchsummary import summary
import wandb

MAX_DIM = 64 * 16  # 1024

class Predictor(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        # ========= create models ===========
        self.predictor = ResNet18(n_classes=1, pretrained=True, hidden_size=self.hidden_size, dropout=0.5)
        # if opt['predictor'] == 'ResNet50':
        #     self.predictor = basenet.ResNet50(n_classes=2, pretrained=True, hidden_size=self.hidden_size, dropout=opt['dropout']).to(self.device)
        # elif opt['predictor'] == 'ResNet18':
        #     self.predictor = basenet.ResNet18(n_classes=2, pretrained=True, hidden_size=self.hidden_size, dropout=opt['dropout']).to(self.device)
        # elif opt['predictor'] == 'VGG16':
        #     self.predictor = basenet.Vgg16(n_classes=2, pretrained=True, dropout=opt['dropout']).to(self.device)

    def forward(self, x):
        out, feature = self.predictor(x)
        return out
    
    def load_weights(self, file_path, optim_pred=None):
        ckpt = torch.load(file_path)
        self.predictor.load_state_dict(ckpt["predictor"])
    
    def _criterion_pred(self, output, target):
        # output is logits and target is 0 or 1
        return F.binary_cross_entropy_with_logits(output, target)
        

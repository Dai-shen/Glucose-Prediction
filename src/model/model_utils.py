import torch.nn as nn
import torch

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        diff = torch.abs(y_true - y_pred)
        delta = self.delta
        if delta > 0:
            # 使用Huber Loss
            loss = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
        else:
            # 当delta等于0时，退化为MSE Loss
            loss = 0.5 * diff ** 2
        return loss.mean()

def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'transformer_encoder' in name:
                # For Transformer encoder layer, initialize the weight parameters as 2D tensor
                nn.init.xavier_uniform_(param.view(param.size(0), -1))
            elif 'fc_layer' in name:
                # For fully connected layer, initialize the weight parameters as 2D tensor
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
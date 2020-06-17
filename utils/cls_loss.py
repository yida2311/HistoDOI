import torch.nn as nn
import torch.nn.functional as F
import torch



class CrossEntropyLoss(nn.Module):
    """CE Loss
        - no one-hot
        - +log_softmax
        - label smooth
    """
    def __init__(self, weight=None, size_average=True, ignore_index=-100, sigma=1e-3):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.sigma = sigma
    
    def forward(self, inputs, targets):
        # smooth label
        labels = torch.argmax(targets, dim=1)
        res = self.sigma * torch.ones_like(targets)
        res[labels] = - 2 * self.sigma
        targets += res 

        logits = F.log_softmax(inputs, dim=1)
        loss = torch.mul(logits, targets)

        if self.size_average:
            loss = loss.mean()
        
        return loss


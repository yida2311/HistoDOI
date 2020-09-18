import torch.nn as nn
import torch.nn.functional as F 
import torch 
import numpy as np 


class globalBranch(nn.Module):
    def __init__(self, n_class):
        super(globalBranch, self).__init__()
        #
'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.segmentor.SemanticFlow.fpn_semantic_flow import FPN_SF
from models.backbone.ResNet.resnet_dilated import resnet_dilated_50


class FPN_Bilinear(nn.Module):
    def __init__(self, num_classes, backbone):
        super(FPN_Bilinear, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.head = FPN_SF(num_classes, expansion=4, mode='Bilinear')
    
    def forward(self, img):
        _, _, H, W = img.size()
        c2, c3, c4, c5 = self.backbone(img)
        output = self.head(c2, c3, c4, c5)
        output = F.interpolate(output, size=(H, W), mode='bilinear')

        return output


def fpn_bilinear_resnet50(num_classes):
    backbone = resnet_dilated_50(pretrained=True)
    model = FPN_Bilinear(num_classes=num_classes, backbone=backbone)
    
    return model





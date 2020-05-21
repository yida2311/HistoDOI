import torch 
import os
from torch import nn

from .EfficientNet_model import EfficientNet as EffNet


class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        model = EffNet.from_pretrained('efficientnet-b0')
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

        self.conv3 = nn.Conv2d(40, 64, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(80, 128, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

    def extract_features(self, x):
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        feature_maps = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(x)

        return feature_maps[1:]

    def forward(self, x):
        features = self.extract_features(x)
        c3 = self.conv3(features[0])
        c4 = self.conv4(features[1])
        c5 = self.conv5(features[2])
        c6 = self.conv6(c5)

        return c3, c4, c5, c6
        

def efficientnet_b0():
    bN = 0
    num_classes = 4
    ds_low = 16
    ds_high = 32
    pretrained = True
    norm_type = "batchnorm"
    return effnet(bN, num_classes, ds_low, ds_high, pretrained=pretrained, norm_type=norm_type)



if __name__ == '__main__':
    model_path = '/remote-home/ldy/OSCC_Seg/pretrained_model/EfficientNet/efficientnet-b0-355c32eb.pth'
    state_dict = torch.load(model_path)
    print(state_dict.keys())
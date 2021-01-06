import torch
from typing import Optional, Union, List
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

from ..dynamicUNet.decoder import UnetDecoder
from ...encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
from segmentation_models_pytorch.base import modules as md
import segmentation_models_pytorch.base.initialization as init


class WeakUnet(nn.Module):
    """ WeakUnet is a fully convolution neural network for weakly-supervised image semantic segmentation derived from UNet

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **WeakUnet**


    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        unary_mid_channels: int = 4,
        unary_out_channels: int = 1,
        classes: int = 3,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super(WeakUnet, self).__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth-1,
            use_batchnorm=decoder_use_batchnorm,
            center=True, # attention\conv\Identity
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.unary_head = UnaryHead(
            in_channels=decoder_channels,
            mid_channels=unary_mid_channels,
            out_channels=unary_out_channels,
            use_batchnorm=decoder_use_batchnorm,
        )

        # self.fr = FillingRate(1)

        self.name = "u-weak-{}".format(encoder_name)
        self.initialize()
    
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_decoder(self.unary_head)
        init.initialize_head(self.segmentation_head)
    

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        outputs = self.decoder.extract_features(*features)
        masks = self.segmentation_head(outputs[-1])
        unary, fr = self.unary_head(outputs)

        return masks, unary, fr
    


class UnaryHead(nn.Module):
    def __init__(
        self, 
        in_channels,
        mid_channels,
        out_channels,
        use_batchnorm=True
    ):
        super(UnaryHead, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.smooth = self.get_smooth_layers(in_channels, mid_channels)
        self.conv = md.Conv2dReLU(len(in_channels)*mid_channels, mid_channels, 3, 1, use_batchnorm=self.use_batchnorm)
        self.unary = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fr = nn.Sequential(
            nn.Linear(mid_channels, out_channels, bias=True),
            nn.Sigmoid(),
        )
    
    def get_smooth_layers(self, in_channels, mid_channels):
        blocks = []
        for in_channel in in_channels:
            blocks.append(md.Conv2dReLU(in_channel, mid_channels, 1, use_batchnorm=self.use_batchnorm))
        return nn.ModuleList(blocks)
    
    def forward(self, features):
        _, _, h, w = features[-1].size() 
        feat = []
        for i, feature in enumerate(features):
            x = self.smooth[i](feature)
            x = F.interpolate(x, size=[h, w], mode='nearest')
            feat.append(x)
        feat = torch.cat(feat, dim=1)
        feat = self.conv(feat)

        unary = self.unary(feat)
        fr = self.pooling(feat)
        fr = torch.flatten(fr, 1)
        fr = self.fr(fr)
        return unary, fr




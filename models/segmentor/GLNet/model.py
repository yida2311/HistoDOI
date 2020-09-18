import torch.nn as nn
import torch.nn.functional as F 
import torch 
import numpy as np 

from ..dynamicUNet.decoder import DecoderBlock, DAN_Module


class globalBranch(nn.Module):
    def __init__(
            self, 
            n_class, 
            encoder_channels=[512, 256, 128, 64],
            deocder_channels=[256, 128, 64, 64],
            attention_type=None,
            center=False,
            ):
        super(globalBranch, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.encoder_channels = encoder_channels
        self.deocder_channels = deocder_channels
        self.hidden_channel = 256
        # Top layer
        if center:
            self.toplayer = DAN_Module(self.encoder_channels[0], self.decoder_channels[0])
        else:
            self.toplayer = nn.Conv2d(self.encoder_channels[0], self.deocder_channels[0], kernel_size=1, stride=1, padding=1)
        # Upsample layers
        self.upsample1 = DecoderBlock(self.decoder_channels[0], self.encoder_channels[1], self.deocder_channels[1], attention_type=attention_type)
        self.upsample2 = DecoderBlock(self.decoder_channels[1], self.encoder_channels[2], self.deocder_channels[2], attention_type=attention_type)
        self.upsample3 = DecoderBlock(self.decoder_channels[2], self.encoder_channels[3], self.deocder_channels[3], attention_type=attention_type)
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(self.decoder_channels[0], self.decoder_channels[0], kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(self.decoder_channels[1], self.decoder_channels[1], kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(self.decoder_channels[2], self.decoder_channels[2], kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(self.decoder_channels[3], self.decoder_channels[3], kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(self.decoder_channels[0], self.decoder_channels[0]//2, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(self.decoder_channels[1], self.decoder_channels[1]//2, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(self.decoder_channels[2], self.decoder_channels[2]//2, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(self.decoder_channels[3], self.decoder_channels[3]//2, kernel_size=3, stride=1, padding=1)
        # Classify layers
        self.classifier = nn.Conv2d(sum(self.deocder_channels)//2, n_class, kernel_size=3, stride=1, padding=1)

        # Local2Global: double channels  ***
        # Lateral layers
        self.latlayer1_ext = nn.Conv2d(self.encoder_channels[0]*2, self.encoder_channels[0], kernel_size=3, stride=1, padding=1)
        self.latlayer2_ext = nn.Conv2d(self.encoder_channels[1]*2, self.encoder_channels[1], kernel_size=3, stride=1, padding=1)
        self.latlayer3_ext = nn.Conv2d(self.encoder_channels[2]*2, self.encoder_channels[2], kernel_size=3, stride=1, padding=1)
        self.latlayer4_ext = nn.Conv2d(self.encoder_channels[3]*2, self.encoder_channels[3], kernel_size=3, stride=1, padding=1)
        # Smooth layers
        self.smooth1_1_ext = nn.Conv2d(self.decoder_channels[0]*2, self.decoder_channels[0], kernel_size=3, stride=1, padding=1)
        self.smooth2_1_ext = nn.Conv2d(self.decoder_channels[1]*2, self.decoder_channels[1], kernel_size=3, stride=1, padding=1)
        self.smooth3_1_ext = nn.Conv2d(self.decoder_channels[2]*2, self.decoder_channels[2], kernel_size=3, stride=1, padding=1)
        self.smooth4_1_ext = nn.Conv2d(self.decoder_channels[3]*2, self.decoder_channels[3], kernel_size=3, stride=1, padding=1)
        self.smooth1_2_ext = nn.Conv2d(self.decoder_channels[0]*2, self.decoder_channels[0]//2, kernel_size=3, stride=1, padding=1)
        self.smooth2_2_ext = nn.Conv2d(self.decoder_channels[1]*2, self.decoder_channels[1]//2, kernel_size=3, stride=1, padding=1)
        self.smooth3_2_ext = nn.Conv2d(self.decoder_channels[2]*2, self.decoder_channels[2]//2, kernel_size=3, stride=1, padding=1)
        self.smooth4_2_ext = nn.Conv2d(self.decoder_channels[3]*2, self.decoder_channels[3]//2, kernel_size=3, stride=1, padding=1)
        self.smooth = nn.Conv2d(sum(self.deocder_channels), sum(self.deocder_channels)//2, kernel_size=3, stride=3, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)
    

    def forward(self, feat, feat_ext=None, ps0_ext=None, ps1_ext=None, ps2_ext=None):
        # feat (list): [c5, c4, c3, c2]
        if feat_ext is None:
            p5 = self.toplayer(feat[0])
            p4 = self.upsample1(p5, skip=feat[1]) 
            p3 = self.upsample2(p4, skip=feat[2])
            p2 = self.upsample3(p3, skip=feat[3])
        else:
            p5 = self.toplayer(self.latlayer1_ext(torch.cat((feat[0], feat_ext[0]), dim=1)))
            p4 = self.upsample1(p5, skip=self.latlayer2_ext(torch.cat((feat[1], feat_ext[1]), dim=1)))
            p3 = self.upsample2(p4, skip=self.latlayer3_ext(torch.cat((feat[2], feat_ext[2]), dim=1)))
            p2 = self.upsample3(p3, skip=self.latlayer4_ext(torch.cat((feat[3], feat_ext[3]), dim=1)))
        ps0 = [p5, p4, p3, p2]

        if ps0_ext is None:
            p5 = self.smooth1_1(p5)
            p4 = self.smooth2_1(p4)
            p3 = self.smooth3_1(p3)
            p2 = self.smooth4_1(p2)
        else:
            p5 = self.smooth1_1_ext(torch.cat((p5, ps0_ext[0]), dim=1))
            p4 = self.smooth2_1_ext(torch.cat((p4, ps0_ext[1]), dim=1))
            p3 = self.smooth3_1_ext(torch.cat((p3, ps0_ext[2]), dim=1))
            p2 = self.smooth4_1_ext(torch.cat((p2, ps0_ext[3]), dim=1))
        ps1 = [p5, p4, p3, p2]

        if ps1_ext is None:
            p5 = self.smooth1_2(p5)
            p4 = self.smooth2_2(p4)
            p3 = self.smooth3_2(p3)
            p2 = self.smooth4_2(p2)
        else:
            p5 = self.smooth1_2_ext(torch.cat((p5, ps1_ext[0]), dim=1))
            p4 = self.smooth2_2_ext(torch.cat((p4, ps1_ext[1]), dim=1))
            p3 = self.smooth3_2_ext(torch.cat((p3, ps1_ext[2]), dim=1))
            p2 = self.smooth4_2_ext(torch.cat((p2, ps1_ext[3]), dim=1))
        ps2 = [p5, p4, p3, p2]

        # Classify
        if ps2_ext is None:
            ps3 = self._concatenate(p5, p4, p3, p2)
            output = self.classifier(ps3)
        else:
            ps3 = self.smooth(self._concatenate(
                torch.cat((p5, ps2_ext[0]), dim=1),
                torch.cat((p4, ps2_ext[1]), dim=1),
                torch.cat((p3, ps2_ext[2]), dim=2),
                torch.cat((p2, ps2_ext[3]), dim=3)
            ))
            output = self.classifier(ps3)
        
        return output, ps0, ps1, ps2, ps3


class localBranch(nn.Module):
    def __init__(
            self, 
            n_class, 
            encoder_channels=[512, 256, 128, 64],
            deocder_channels=[256, 128, 64, 64],
            attention_type=None,
            center=False,
            ):
        super(localBranch, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.encoder_channels = encoder_channels
        self.deocder_channels = deocder_channels
        self.hidden_channel = 256
        fold = 2
        # Top layer
        if center:
            self.toplayer = DAN_Module(self.encoder_channels[0], self.decoder_channels[0])
        else:
            self.toplayer = nn.Conv2d(self.encoder_channels[0], self.deocder_channels[0], kernel_size=1, stride=1, padding=1)
        # Upsample layers
        self.upsample1 = DecoderBlock(self.decoder_channels[0], self.encoder_channels[1], self.deocder_channels[1], attention_type=attention_type)
        self.upsample2 = DecoderBlock(self.decoder_channels[1], self.encoder_channels[2], self.deocder_channels[2], attention_type=attention_type)
        self.upsample3 = DecoderBlock(self.decoder_channels[2], self.encoder_channels[3], self.deocder_channels[3], attention_type=attention_type)
        # Classify layers
        self.classifier = nn.Conv2d(sum(self.deocder_channels)//2, n_class, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(self.encoder_channels[0]*fold, self.encoder_channels[0], kernel_size=3, stride=1, padding=1)
        self.latlayer2 = nn.Conv2d(self.encoder_channels[1]*fold, self.encoder_channels[1], kernel_size=3, stride=1, padding=1)
        self.latlayer3 = nn.Conv2d(self.encoder_channels[2]*fold, self.encoder_channels[2], kernel_size=3, stride=1, padding=1)
        self.latlayer4 = nn.Conv2d(self.encoder_channels[3]*fold, self.encoder_channels[3], kernel_size=3, stride=1, padding=1)
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(self.decoder_channels[0]*fold, self.decoder_channels[0], kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(self.decoder_channels[1]*fold, self.decoder_channels[1], kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(self.decoder_channels[2]*fold, self.decoder_channels[2], kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(self.decoder_channels[3]*fold, self.decoder_channels[3], kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(self.decoder_channels[0]*fold, self.decoder_channels[0]//2, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(self.decoder_channels[1]*fold, self.decoder_channels[1]//2, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(self.decoder_channels[2]*fold, self.decoder_channels[2]//2, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(self.decoder_channels[3]*fold, self.decoder_channels[3]//2, kernel_size=3, stride=1, padding=1)
        self.smooth = nn.Conv2d(sum(self.deocder_channels), sum(self.deocder_channels)//2, kernel_size=3, stride=3, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)
    

    def forward(self, feat, feat_ext=None, ps0_ext=None, ps1_ext=None, ps2_ext=None):
        # feat (list): [c5, c4, c3, c2]
        p5 = self.toplayer(self.latlayer1(torch.cat((feat[0], F.interpolate(feat_ext[0], size=feat[0].size()[2:], **self._up_kwargs)), dim=1)))
        p4 = self.upsample1(p5, skip=self.latlayer2(torch.cat((feat[1], F.interpolate(feat_ext[1], size=feat[1].size()[2:], **self._up_kwargs)), dim=1)))
        p3 = self.upsample1(p4, skip=self.latlayer2(torch.cat((feat[2], F.interpolate(feat_ext[2], size=feat[2].size()[2:], **self._up_kwargs)), dim=1)))
        p2 = self.upsample1(p4, skip=self.latlayer2(torch.cat((feat[3], F.interpolate(feat_ext[3], size=feat[3].size()[2:], **self._up_kwargs)), dim=1)))
        ps0 = [p5, p4, p3, p2]

        p5 = self.smooth1_1(torch.cat((p5, F.interpolate(ps0_ext[0][0], size=p5.size()[2:], **self._up_kwargs)), dim=1))
        p4 = self.smooth2_1(torch.cat((p4, F.interpolate(ps0_ext[1][0], size=p4.size()[2:], **self._up_kwargs)), dim=1))
        p3 = self.smooth3_1(torch.cat((p3, F.interpolate(ps0_ext[2][0], size=p3.size()[2:], **self._up_kwargs)), dim=1))
        p2 = self.smooth4_1(torch.cat((p2, F.interpolate(ps0_ext[3][0], size=p2.size()[2:], **self._up_kwargs)), dim=1))
        ps1 = [p5, p4, p3, p2]

        p5 = self.smooth1_2(torch.cat((p5, F.interpolate(ps1_ext[0][0], size=p5.size()[2:], **self._up_kwargs)), dim=1))
        p4 = self.smooth2_2(torch.cat((p4, F.interpolate(ps1_ext[1][0], size=p4.size()[2:], **self._up_kwargs)), dim=1))
        p3 = self.smooth3_2(torch.cat((p3, F.interpolate(ps1_ext[2][0], size=p3.size()[2:], **self._up_kwargs)), dim=1))
        p2 = self.smooth4_2(torch.cat((p2, F.interpolate(ps1_ext[3][0], size=p2.size()[2:], **self._up_kwargs)), dim=1))
        ps2 = [p5, p4, p3, p2]

        ps3 = self.smooth(self._concatenate(
            torch.cat((p5, F.interpolate(ps2_ext[0][0], size=p5.size()[2:], **self._up_kwargs)), dim=1),
            torch.cat((p4, F.interpolate(ps2_ext[1][0], size=p4.size()[2:], **self._up_kwargs)), dim=1),
            torch.cat((p3, F.interpolate(ps2_ext[2][0], size=p3.size()[2:], **self._up_kwargs)), dim=1),
            torch.cat((p2, F.interpolate(ps2_ext[3][0], size=p2.size()[2:], **self._up_kwargs)), dim=1)
        ))

        # Classify
        output = self.classifier(ps3)
        
        return output, ps0, ps1, ps2, ps3






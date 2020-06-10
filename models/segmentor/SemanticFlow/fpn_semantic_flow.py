import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def get_fpn_sf_global(num_classes, mode):
    return FPN_SF(num_classes, expansion=4, mode=mode)


def get_fpn_sf_local(num_classes, mode):
    return FPN_SF(num_classes, expansion=1, mode=mode) 


class FPN_SF(nn.Module):
    def __init__(self, num_classes, expansion=1, mode='SemanticFlow'):
        """Meta model for FPN based on Semantci Flow
        Params:
            expansion: 1(resnet18, resenet34), 4(resnet50, resnet101)
            mode: upsample mode
        """
        super(FPN_SF, self).__init__()
        self.mode = mode  # SemanticFlow, Bilinear
        # PPM
        self.toplayer = PSPModule(features=512*expansion, out_features=256)
        
        # lateral layers
        self.laterlayer1 = nn.Conv2d(256*expansion, 256, kernel_size=1, stride=1, padding=0)
        self.laterlayer2 = nn.Conv2d(128*expansion, 256, kernel_size=1, stride=1, padding=0)
        self.laterlayer3 = nn.Conv2d(64*expansion, 256, kernel_size=1, stride=1, padding=0)
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # FAM layers
        if self.mode == 'SemanticFlow':
            self.fam1 = FAM(features=256)
            self.fam2 = FAM(features=256)
            self.fam3 = FAM(features=256)
            self.fam1_ext = FAM(features=256)
            self.fam2_ext = FAM(features=256)
            self.fam3_ext = FAM(features=256) 
        
        # classify layer
        self.fuse = nn.Sequential(nn.Conv2d(256*4, 256, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(True))
        self.classify = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1)

        # init
        self._init_params()

    def _init_params(self):
        for m in self.children():
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias, 0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        '''
        _, _, H, W = y.size()
        if self.mode == 'Bilinear':
            return F.interpolate(x, size=(H, W), mode='bilinear') + y
     
    def forward(self, c2, c3, c4, c5):
        # upsample
        p5 = self.toplayer(c5)
        if self.mode == 'SemanticFlow':
            p4 = self.fam1(p5, self.laterlayer1(c4))
            p3 = self.fam2(p4, self.laterlayer2(c3))
            p2 = self.fam3(p3, self.laterlayer3(c2))
            # deep supervision
            p5 = self.fam1_ext(p5, p2)
            p4 = self.fam2_ext(p4, p2)
            p3 = self.fam3_ext(p3, p2)
        else:
            p4 = self._upsample_add(p5, self.laterlayer1(c4))
            p3 = self._upsample_add(p4, self.laterlayer2(c3))
            p2 = self._upsample_add(p3, self.laterlayer3(c2))
            # Smooth
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            p2 = self.smooth3(p2)
            # # deep supervision
            _, _, H, W = p2.size()
            p5 = F.interpolate(p5, size=(H, W), mode='bilinear')
            p4 = F.interpolate(p4, size=(H, W), mode='bilinear')
            p3 = F.interpolate(p3, size=(H, W), mode='bilinear')
        ensemble = self.fuse(torch.cat([p5, p4, p3, p2], dim=1))
        output = self.classify(ensemble)
        
        return output
    
    def get_fusion_feature_pyramid(self, c2, c3, c4, c5):
        p5 = self.toplayer(c5)
        p4 = self.laterlayer1(c4)
        p3 = self.laterlayer2(c3)
        p2 = self.laterlayer3(c2)


        return [p2, p3, p4, p5]

    def forward_with_lateral(self, ps):
        p5 = ps[3]
        if self.mode == 'SemanticFlow':
            p4 = self.fam1(p5, ps[2])
            p3 = self.fam2(p4, ps[1])
            p2 = self.fam3(p3, ps[0])
            p5 = self.fam1_ext(p5, p2)
            p4 = self.fam2_ext(p4, p2)
            p3 = self.fam3_ext(p3, p2)
        else:
            # Smooth
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            p2 = self.smooth3(p2)
            
            p4 = self._upsample_add(p5, ps[2])
            p3 = self._upsample_add(p4, ps[1])
            p2 = self._upsample_add(p3, ps[0])
            # # deep supervision
            _, _, H, W = p2.size()
            p5 = F.interpolate(p5, size=(H, W), mode='bilinear')
            p4 = F.interpolate(p4, size=(H, W), mode='bilinear')
            p3 = F.interpolate(p3, size=(H, W), mode='bilinear')
        ensemble = self.fuse(torch.cat([p5, p4, p3, p2], dim=1))
        output = self.classify(ensemble)
        
        return output, ensemble

    def ensemble_classifier(self, ps):
        p2 = ps[0]
        if self.mode == 'SemanticFlow':
            p5 = self.fam1_ext(ps[3], p2)
            p4 = self.fam2_ext(ps[2], p2)
            p3 = self.fam3_ext(ps[1], p2)
        else:
            _, _, H, W = p2.size()
            p5 = F.interpolate(p5, size=(H, W), mode='bilinear')
            p4 = F.interpolate(p4, size=(H, W), mode='bilinear')
            p3 = F.interpolate(p3, size=(H, W), mode='bilinear')
        ensemble = self.fuse(torch.cat([p5, p4, p3, p2], dim=1))
        output = self.classify(ensemble)

        return output, ensemble


class FAM(nn.Module):
    def __init__(self, features=256):
        super().__init__()
        self.smooth_up = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.smooth_d = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.flow = nn.Conv2d(features*2, 2, kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        '''
        _, _, H, W = y.size()
        x_smooth = self.smooth_up(x)
        x_smooth = F.interpolate(x, size=(H, W), mode='bilinear')
        y_smooth = self.smooth_d(y)
        
        flow = self.flow(torch.cat([x_smooth, y_smooth], dim=1))
        x_warp = self.stn(x, flow)
        return x_warp + y
    
    def stn(self, x, flow):
        _, _, H, W = flow.size()
        grid = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        grid = torch.stack(grid)
        grid = torch.unsqueeze(grid, 0).float().cuda()
        flow += grid
        flow[:, 0, ...] = 2 * (flow[:, 0, ...]/(H-1) - 0.5)
        flow[:, 1, ...] = 2 * (flow[:, 1, ...]/(W-1) - 0.5)

        flow = flow.permute(0, 2, 3, 1)
        
        return F.grid_sample(x, flow)


class PSPModule(nn.Module):
    def __init__(self, features, out_features=256, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
                        nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1),
                        nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1))
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)





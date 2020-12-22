import torch.nn as nn
import torch.nn.functional as F 
import torch 
import numpy as np 

from ..dynamicUNet.decoder import DecoderBlock, DAN_Module
from ...encoders import get_encoder, encoders


class globalBranch(nn.Module):
    def __init__(
            self, 
            n_class, 
            encoder_channels=[512, 256, 128, 64],
            decoder_channels=[256, 128, 64, 64],
            attention_type=None,
            center=False,
            ):
        super(globalBranch, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.hidden_channel = 256
        # Top layer
        if center:
            self.toplayer = DAN_Module(self.encoder_channels[0], self.decoder_channels[0])
        else:
            self.toplayer = nn.Conv2d(self.encoder_channels[0], self.decoder_channels[0], kernel_size=1, stride=1, padding=1)
        # Upsample layers
        self.upsample1 = DecoderBlock(self.decoder_channels[0], self.encoder_channels[1], self.decoder_channels[1], attention_type=attention_type)
        self.upsample2 = DecoderBlock(self.decoder_channels[1], self.encoder_channels[2], self.decoder_channels[2], attention_type=attention_type)
        self.upsample3 = DecoderBlock(self.decoder_channels[2], self.encoder_channels[3], self.decoder_channels[3], attention_type=attention_type)
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
        self.classifier = nn.Conv2d(sum(self.decoder_channels)//2, n_class, kernel_size=3, stride=1, padding=1)

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
        self.smooth1_2_ext = nn.Conv2d(self.decoder_channels[0]*2, self.decoder_channels[0], kernel_size=3, stride=1, padding=1)
        self.smooth2_2_ext = nn.Conv2d(self.decoder_channels[1]*2, self.decoder_channels[1], kernel_size=3, stride=1, padding=1)
        self.smooth3_2_ext = nn.Conv2d(self.decoder_channels[2]*2, self.decoder_channels[2], kernel_size=3, stride=1, padding=1)
        self.smooth4_2_ext = nn.Conv2d(self.decoder_channels[3]*2, self.decoder_channels[3], kernel_size=3, stride=1, padding=1)
        self.smooth = nn.Conv2d(sum(self.decoder_channels)//2, sum(self.decoder_channels), kernel_size=3, stride=3, padding=1)

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


def globalNet(nn.Module):
    def __init__(
        self,
        n_class,
        encoder_name: str = 'resnet18',
        decoder_channels = [256, 128, 64, 64],
        attention_type = None,
    ):
        super(globalNet, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.encoder_channels = encoders[encoder_name]["params"]["out_channels"][::-1]
        self.decoder_channels = decoder_channels
        self.encoder = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        self.decoder = globalBranch(
            n_class,
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
            attention_type=attention_type,
            center=True,
        )

        # init decoder
        for m in self.decoder.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)
    

    def forward(self, image):
        feat = self.encoder_global.forward(image)
        output, ps0, ps1, ps2, ps3 = self.decoder.forward(feat)



class localBranch(nn.Module):
    def __init__(
            self, 
            n_class, 
            encoder_channels=[512, 256, 128, 64],
            decoder_channels=[256, 128, 64, 64],
            attention_type=None,
            center=False,
            ):
        super(localBranch, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.hidden_channel = 256
        fold = 2
        # Top layer
        if center:
            self.toplayer = DAN_Module(self.encoder_channels[0], self.decoder_channels[0])
        else:
            self.toplayer = nn.Conv2d(self.encoder_channels[0], self.decoder_channels[0], kernel_size=1, stride=1, padding=1)
        # Upsample layers
        self.upsample1 = DecoderBlock(self.decoder_channels[0], self.encoder_channels[1], self.decoder_channels[1], attention_type=attention_type)
        self.upsample2 = DecoderBlock(self.decoder_channels[1], self.encoder_channels[2], self.decoder_channels[2], attention_type=attention_type)
        self.upsample3 = DecoderBlock(self.decoder_channels[2], self.encoder_channels[3], self.decoder_channels[3], attention_type=attention_type)
        # Classify layers
        self.classifier = nn.Conv2d(sum(self.decoder_channels)//2, n_class, kernel_size=3, stride=1, padding=1)

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
        self.smooth1_2 = nn.Conv2d(self.decoder_channels[0]*fold, self.decoder_channels[0], kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(self.decoder_channels[1]*fold, self.decoder_channels[1], kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(self.decoder_channels[2]*fold, self.decoder_channels[2], kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(self.decoder_channels[3]*fold, self.decoder_channels[3], kernel_size=3, stride=1, padding=1)
        self.smooth = nn.Conv2d(sum(self.decoder_channels), sum(self.decoder_channels)//2, kernel_size=3, stride=3, padding=1)

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


class GLNet(nn.Module):
    def __init__(
        self, 
        n_class,
        encoder_name: str = 'resnet18',
        decoder_channels = [256, 128, 64, 64],
        attention_type = None,

    ):
        super(GLNet, self).__init__()
        self._up_kwargs = {'mode': 'bilinear'}
        self.encoder_channels = encoders[encoder_name]["params"]["out_channels"][::-1]
        self.decoder_channels = decoder_channels
        # Encoders
        self.encoder_local = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        self.encoder_global = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights="imagenet",
        )
        # Decoders
        self.decoder_local = localBranch(
            n_class,
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels,
            attention_type=attention_type,
            center=True,
        )
        self.decoder_global = globalBranch(
            n_class,
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels,
            attention_type=attention_type,
            center=True,
        )

        # Intermediate features (cache)
        self.feat_g = [None]*4; self.output_g = None
        self.ps0_g = None; self.ps1_g = None; self.ps2_g = None
        self.ps3_g = None

        self.feat_l = [[] for _ in range(4)]
        self.ps0_l = [[] for _ in range(4)]; self.ps1_l = [[] for _ in range(4)]; self.ps2_l = [[] for _ in range(4)]
        self.ps3_l = []; self.output_l = []

        self.feat_b = [None]*4
        self.ps0_b = [None]*4; self.ps1_b = [None]*4; self.ps2_b = [None]*4
        self.ps3_b = []

        self.patch_n = 0

        self.mse = nn.MSELoss()
        self.ensemble_conv = nn.Conv2d(sum(self.decoder_channels), n_class, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.ensemble_conv.weight, mean=0, std=0.01)

        # init decoder
        for m in self.decoder_global.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)
        for m in self.decoder_local.children():
            if hasattr(m, 'weight'): nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, 'bias'): nn.init.constant_(m.bias, 0)
    
    def clear_cache(self):
        self.feat_g = [None]*4; self.output_g = None
        self.ps0_g = None; self.ps1_g = None; self.ps2_g = None
        self.ps3_g = None

        self.feat_l = [[] for _ in range(4)]
        self.ps0_l = [[] for _ in range(4)]; self.ps1_l = [[] for _ in range(4)]; self.ps2_l = [[] for _ in range(4)]
        self.ps3_l = []; self.output_l = []

        self.feat_b = [None]*4
        self.ps0_b = [None]*4; self.ps1_b = [None]*4; self.ps2_b = [None]*4
        self.ps3_b = []

        self.patch_n = 0
    
    def _sample_grid(self, fm, bbox, sampleSize):
        """
        :param fm: tensor(b,c,h,w) the global feature map
        :param bbox: list [b* nparray(x1, y1, x2, y2)] the (x1,y1) is the left_top of bbox, (x2, y2) is the right_bottom of bbox
        there are in range [0, 1]. x is corresponding to width dimension and y is corresponding to height dimension
        :param sampleSize: (oH, oW) the point to sample in height dimension and width dimension
        :return: tensor(b, c, oH, oW) sampled tensor
        """
        b, c, h, w = fm.shape
        b_bbox = len(bbox)
        bbox = [x*2 - 1 for x in bbox] # range transform
        if b != b_bbox and b == 1:
            fm = torch.cat([fm,]*b_bbox, dim=0)
        grid = np.zeros((b_bbox,) + sampleSize + (2,), dtype=np.float32)
        gridMap = np.array([[(cnt_w/(sampleSize[1]-1), cnt_h/(sampleSize[0]-1)) for cnt_w in range(sampleSize[1])] for cnt_h in range(sampleSize[0])])
        for cnt_b in range(b_bbox):
            grid[cnt_b, :, :, 0] = bbox[cnt_b][0] + (bbox[cnt_b][2] - bbox[cnt_b][0])*gridMap[:, :, 0]
            grid[cnt_b, :, :, 1] = bbox[cnt_b][1] + (bbox[cnt_b][3] - bbox[cnt_b][1])*gridMap[:, :, 1]
        grid = torch.from_numpy(grid).cuda()
        return F.grid_sample(fm, grid)

    def _crop_global(self, f_global, top_lefts, ratio):
        '''
        top_lefts: [(top, left)] * b
        '''
        _, c, H, W = f_global.size()
        b = len(top_lefts)
        h, w = int(np.round(H * ratio[0])), int(np.round(W * ratio[1]))

        # bbox = [ np.array([left, top, left + ratio, top + ratio]) for (top, left) in top_lefts ]
        # crop = self._sample_grid(f_global, bbox, (H, W))

        crop = []
        for i in range(b):
            top, left = int(np.round(top_lefts[i][0] * H)), int(np.round(top_lefts[i][1] * W))
            # # global's sub-region & upsample
            # f_global_patch = F.interpolate(f_global[0:1, :, top:top+h, left:left+w], size=(h, w), mode='bilinear')
            f_global_patch = f_global[0:1, :, top:top+h, left:left+w]
            crop.append(f_global_patch[0])
        crop = torch.stack(crop, dim=0) # stack into mini-batch
        return [crop] # return as a list for easy to torch.cat

    def _merge_local(self, f_local, merge, f_global, top_lefts, oped, ratio, template):
        '''
        merge feature maps from local patches, and finally to a whole image's feature map (on cuda)
        f_local: a sub_batch_size of patch's feature map
        oped: [start, end)
        '''
        b, _, _, _ = f_local.size()
        _, c, H, W = f_global.size() # match global feature size
        if merge is None:
            merge = torch.zeros((1, c, H, W)).cuda()
        h, w = int(np.round(H * ratio[0])), int(np.round(W * ratio[1]))
        for i in range(b):
            index = oped[0] + i
            top, left = int(np.round(H * top_lefts[index][0])), int(np.round(W * top_lefts[index][1]))
            merge[:, :, top:top+h, left:left+w] += F.interpolate(f_local[i:i+1], size=(h, w), **self._up_kwargs)
        if oped[1] >= len(top_lefts):
            template = F.interpolate(template, size=(H, W), **self._up_kwargs)
            template = template.expand_as(merge)
            # template = Variable(template).cuda()
            merge /= template
        return merge

    def ensemble(self, f_local, f_global):
        return self.ensemble_conv(torch.cat((f_local, f_global), dim=1))
    
    def collect_local_fm(self, image_global, patches, ratio, top_lefts, oped, batch_size, global_model=None, template=None, n_patch_all=None):
        '''
        patches: 1 patch
        top_lefts: all top-left
        oped: [start, end)
        '''
        with torch.no_grad():
            if self.patch_n == 0:
                self.feat_g = global_model.module.encoder_global.forward(image_global)
                self.output_g, self.ps0_g, self.ps1_g, self.ps2_g, self.ps3_g = global_model.module.decoder_global.forward(self.feat_g)
                # self.output_g = F.interpolate(self.output_g, image_global.size()[2:], mode='nearest')
            self.patch_n += patches.size()[0]
            self.patch_n %= n_patch_all

            self.encoder_local.eval()
            self.decoder_local.eval()
            feat = self.encoder_local.forward(patches)
            # global's 1x patch cat
            output, ps0, ps1, ps2, ps3 = self.decoder_local.forward(
                feat,
                feat_ext=[ self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.feat_g ],
                ps0_ext=[ self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.ps0_g ],
                ps1_ext=[ self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.ps1_g ],
                ps2_ext=[ self._crop_global(f, top_lefts[oped[0]:oped[1]], ratio) for f in self.ps2_g ]
            )
            # output = F.interpolate(output, patches.size()[2:], mode='nearest')

            self.feat_b = [self._merge_local(feat[i], self.feat_b[i], self.feat_g[i], top_lefts, oped, ratio, template) for i in range(4)]
            self.ps0_b = [self._merge_local(ps0[i], self.ps0_b[i], self.ps0_g[i], top_lefts, oped, ratio, template) for i in range(4)]
            self.ps1_b = [self._merge_local(ps1[i], self.ps1_b[i], self.ps1_g[i], top_lefts, oped, ratio, template) for i in range(4)]
            self.ps2_b = [self._merge_local(ps2[i], self.ps2_b[i], self.ps2_g[i], top_lefts, oped, ratio, template) for i in range(4)]
            self.ps3_b.append(ps3.cpu())
            # self.output_b.append(output.cpu()) # each output is 1, 7, h, w

            if self.patch_n == 0:
                # merged all patches into an image
                self.feat_l = [self.feat_l[i].append(self.feat_b[i]) for i in range(4)]
                self.ps0_l = [self.ps0_l[i].append(self.ps0_b[i]) for i in range(4)]
                self.ps1_l = [self.ps1_l[i].append(self.ps1_b[i]) for i in range(4)]
                self.ps2_l = [self.ps2_l[i].append(self.ps2_b[i]) for i in range(4)]

                # collected all ps3 and output of patches as a (b) tensor, append into list
                self.ps3_l.append(torch.cat(self.ps3_b, dim=0)); # a list of tensors
                # self.output_l.append(torch.cat(self.output_b, dim=0)) # a list of 36, 7, h, w tensors

                self.feat_b = [None]*4
                self.ps0_b = [None]*4; self.ps1_b = [None]*4; self.ps2_b = [None]*4
                self.ps3_b = []# ; self.output_b = []

            if len(self.feat_l[0]) == batch_size:
                self.feat_l = [torch.cat(c, dim=0) for c in self.feat_l] # .cuda()
                self.ps0_l = [torch.cat(c, dim=0) for c in self.ps0_l] # .cuda()
                self.ps1_l = [torch.cat(c, dim=0) for c in self.ps1_l] # .cuda()
                self.ps2_l = [torch.cat(c, dim=0) for c in self.ps2_l] # .cuda()
                # self.ps3_l = torch.cat(self.ps3_l, dim=0)# .cuda()
            return self.ps3_l, output# self.output_l
    

    def forward(self, image_global, patches, top_lefts, ratio, mode=1, global_model=None, n_patch=None):
        if mode == 1:
            # train global model
            feat_g = self.encoder_global.forward(image_global)
            output_g, ps0_g, ps1_g, ps2_g, ps3_g = self.decoder_global.forward(feat_g)
            # imsize = image_global.size()[2:]
            # output_g = F.interpolate(output_g, imsize, mode='nearest')
            return output_g, None
        elif mode == 2:
            # train global2local model
            with torch.no_grad():
                if self.patch_n == 0:
                    # calculate global images only if patches belong to a new set of global images (when self.patch_n % n_patch == 0)
                    self.feat_g = self.encoder_global.forward(image_global)
                    self.output_g, self.ps0_g, self.ps1_g, self.ps2_g, self.ps3_g = self.decoder_global.forward(self.feat_g)
                    # imsize_glb = image_global.size()[2:]
                    # self.output_g = F.interpolate(self.output_g, imsize_glb, mode='nearest')
                self.patch_n += patches.size()[0]
                self.patch_n %= n_patch

            # train local model #######################################
            feat_l = self.encoder_local.forward(patches)
            # global's 1x patch cat
            output_l, ps0_l, ps1_l, ps2_l, ps3_l = self.decoder_local.forward(
                feat_l,
                feat_ext=[ self._crop_global(self.f, top_lefts, ratio) for f in self.feat_g ],
                ps0_ext=[ self._crop_global(f, top_lefts, ratio) for f in self.ps0_g ],
                ps1_ext=[ self._crop_global(f, top_lefts, ratio) for f in self.ps1_g ],
                ps2_ext=[ self._crop_global(f, top_lefts, ratio) for f in self.ps2_g ],
            )
            # imsize = patches.size()[2:]
            # output_l = F.interpolate(output_l, imsize, mode='nearest')
            ps3_g2l = self._crop_global(self.ps3_g, top_lefts, ratio)[0] # only calculate loss on 1x
            ps3_g2l = F.interpolate(ps3_g2l, size=ps3_l.size()[2:], **self._up_kwargs)

            output = self.ensemble(ps3_l, ps3_g2l)
            # output = F.interpolate(output, imsize, mode='nearest')
            return output, self.output_g, output_l, self.mse(ps3_l, ps3_g2l)
        else:
            # train local2global model
            feat_g = self.encoder_global.forward(image_global)
            # local patch cat into global
            output_g, ps0_g, ps1_g, ps2_g, ps3_g = self.decoder_global.forward(feat_g, feat_ext=self.feat_l, ps0_ext=self.ps0_l, ps1_ext=self.ps1_l, ps2_ext=self.ps2_l)
            # imsize = image_global.size()[2:]
            # output_g = F.interpolate(output_g, imsize, mode='nearest')
            self.clear_cache()
            return output_g, ps3_g











import torch.nn as nn
import torch.nn.functional as F
import torch
from .lovasz_loss import LovaszSoftmax

__all__ = ['FRSegLoss', 'SchpLoss', 'CrossEntropyLoss', 'SymmetricCrossEntropyLoss', 
            'NormalizedSymmetricCrossEntropyLoss', 'FocalLoss', 'SoftCrossEntropyLoss2d']


#===========================================================================================================================
class FRSegLoss(nn.Module):
    """Filling Rate combined segmentation loss, contains TopKSegLoss, UnaryLoss and FillingRate term
        Args:
            num_classes: num of classes,
            alpha: weight of unary term,
            beta: weight of filling rate term,
            reduction: "mean" for averge, "sum" for sum,
            momentum: update weight for filling rate term
        
        Return:
            loss,
            filling rates,
            """
    def __init__(self, num_classes, alpha=1.0, beta=3, momentum=0.8, reduction='mean'):
        super(FRSegLoss, self).__init__()

        self.momentum = momentum
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

        self.unary_loss = nn.MSELoss(reduction=reduction)
        self.topk_seg_loss = TopKSegLoss(num_classes=self.num_classes)


    def forward(self, inputs, targets, unarys, frs, old_frs):
        """
        Args:
            inputs: b x c x h x w
            targets: b x h x w
            unarys: b x (1/2) x h x w
            frs: b x (1/2)    
        """
        b, c, h, w = inputs.size()
        if self.num_classes == 3:
            targets_bin = torch.stack([targets==2], dim=1)  #　binary mask, b x 1 x h x w
        else:
            targets_bin = torch.stack([targets==2, targets==3], dim=1) # b x 2 x h x w
        targets_bin = targets_bin.clone().float().cuda()
        unarys_bin = unarys * targets_bin #　focused unarys
        num_unary = torch.sum(targets_bin, dim=(2, 3)) # b x (1/2)
        filling_rates =  frs * h * w  / (num_unary+10)
        filling_rates = torch.clamp(self.momentum * filling_rates + (1-self.momentum) * old_frs, max=1)
        topk = (filling_rates * num_unary).clone().detach()

        topk_term = self.topk_seg_loss(inputs, targets, unarys_bin, topk, num_unary)
        unary_term = self.unary_loss(unarys, targets_bin)
        fr_term = torch.mean(filling_rates)

        loss = topk_term + self.alpha * unary_term + self.beta * fr_term
        return loss, filling_rates


class TopKSegLoss(nn.Module):
    """TopK Segmentation Loss where topk is focousing on foreground region and background region remain the same.
    Args
    """
    def __init__(self, num_classes):
        super(TopKSegLoss, self).__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    
    def forward(self, inputs, targets, unarys, topk, num_unary):
        b, c, h, w = inputs.size()
        targets_bg = targets 
        targets_bg[targets_bg==2] = - 1
        if self.num_classes != 3:
            targets_bg[targets_bg==3] = -1
        loss_bg = self.ce(inputs, targets) / (b*h*w - torch.sum(num_unary) + 1)

        loss_fg = 0
        if self.num_classes == 3:
            for i in range(b):
                if topk[i] > 0:
                    _, indices = torch.topk(unarys[i].view(-1), int(topk[i]))
                    anchor = inputs[i].view(1, c, -1)[:, :, indices]
                    targets_fg = targets[i].view(1, -1)[:, indices]
                    loss_fg += self.ce(anchor, targets_fg)
        else:
            for i in ranage(b):
                for j in range(2):
                    if topk[i][j] > 0:
                        _, indices = torch.topk(unarys[i][j].view(-1), int(topk[i][j]))
                        anchor = inputs[i].view(1, c, -1)[:, :, indices]
                        targets_fg = targets[i].view(1, -1)[:, indices]
                        loss_fg += self.ce(anchor, targets_fg)
        loss_fg /= torch.sum(topk)

        return (loss_bg + loss_fg)/2


#============================================================================================================================
class DecoupledSegLoss_v1(nn.Module):
    """Decoupled Segmentation loss 
        clean set: CrossEntropy, noisy set: SymmetricCrossEntropy
    """
    def __init__(self, num_classes=4, alpha=1, beta=1, ignore_index=-1, reduction='mean'):
        super(DecoupledSegLoss_v1, self).__init__()
        self.clean_loss = CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.noisy_loss = SymmetricCrossEntropyLoss(alpha=alpha, beta=beta, num_classes=num_classes, ignore_index=ignore_index, reduction=reduction)
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        clean_targets, noisy_targets = self.decoupleTargets(targets)
        loss_clean = self.clean_loss(inputs, clean_targets)
        loss_noisy = self.noisy_loss(inputs, noisy_targets)
        return loss_clean + loss_noisy
        
    def decoupleTargets(self, targets):
        clean_mask = (targets==0) + (targets==1)
        noisy_mask = (targets==2) + (targets==3)
        clean_targets = targets
        clean_targets[noisy_mask] = self.ignore_index
        noisy_targets = targets
        noisy_targets[clean_mask] = self.ignore_index

        return clean_targets, noisy_targets


class DecoupledSegLoss_v2(nn.Module):
    """Decoupled Segmentation loss 
        clean set: Focal, noisy set: SymmetricCrossEntropy
    """
    def __init__(self, num_classes=4, alpha=1, beta=1, gamma=1, ignore_index=-1, reduction='mean'):
        super(DecoupledSegLoss_v2, self).__init__()
        self.clean_loss = FocalLoss(gamma=gamma, ignore_index=ignore_index, reduction=reduction)
        self.noisy_loss = SymmetricCrossEntropyLoss(alpha=alpha, beta=beta, num_classes=num_classes, ignore_index=ignore_index, reduction=reduction)
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        B, C, H, W = inputs.size()
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        targets = targets.view(-1)

        clean_mask, noisy_mask = self.decoupleTargets(targets)
        if targets[clean_mask].size(0) == 0:
            loss_clean = 0
        else:
            loss_clean = self.clean_loss(inputs[clean_mask], targets[clean_mask]) 
        if targets[noisy_mask].size(0) == 0:
            loss_noisy = 0
        else:
            loss_noisy = self.noisy_loss(inputs[noisy_mask], targets[noisy_mask])
        # print(loss_clean, loss_noisy)
        return loss_noisy+loss_clean
        
    def decoupleTargets(self, targets):
        clean_mask = (targets==0) + (targets==1)
        noisy_mask = (targets==2) + (targets==3)
        # clean_targets = targets.clone()
        # clean_targets[noisy_mask] = self.ignore_index
        # noisy_targets = targets.clone()
        # noisy_targets[clean_mask] = self.ignore_index

        # clean_targets = targets[clean_mask]
        # noisy_targets = targets[noisy_mask]

        # return clean_targets, noisy_targets
        return clean_mask, noisy_mask


#======================= SCHP Loss ======================================
class SchpLoss(nn.Module):
    """ For self-correction human parsing"""
    def __init__(self, use_class_weight=False, lamda=1, num_classes=4):
        super(SchpLoss, self).__init__()
        self.use_class_weight = use_class_weight
        self.criterion = nn.CrossEntropyLoss()
        self.lovasz = LovaszSoftmax()
        self.kldiv = KLDivergenceLoss(T=1)
        self.lamda = lamda
        self.num_classes = num_classes
    
    def parsing_loss(self, preds, soft_preds, masks, cycle_n=None):
        """
        Loss function definition.
        Args:
            preds: preds of model
            soft_preds: preds of schp model
            masks: gt masks
        Returns:
            Calculated Loss.
        """
        h, w = masks.size(1), masks.size(2)

        # loss = self.lovasz(preds, masks)
        loss = self.criterion(preds, masks)
        if soft_preds is not None:
            soft_preds = F.softmax(soft_preds)
            soft_preds = moving_average(soft_preds, one_hot(masks, self.num_classes), alpha=1.0/(cycle_n+1))
            loss += self.lamda * self.kldiv(preds, soft_preds)
            loss /= (self.lamda + 1)

        return loss 
    
    def forward(self, preds, soft_preds, masks, cycle_n=None):
        loss = self.parsing_loss(preds, soft_preds, masks, cycle_n=cycle_n)
        return loss 

    def _generate_weights(self, masks, num_classes):
        """
        masks: torch.Tensor with shape [B, H, W]
        """
        masks_label = masks.data.cpu().numpy().astype(np.int64)
        pixel_nums = []
        tot_pixels = 0
        for i in range(num_classes):
            pixel_num_of_cls_i = np.sum(masks_label == i).astype(np.float)
            pixel_nums.append(pixel_num_of_cls_i)
            tot_pixels += pixel_num_of_cls_i
        weights = []
        for i in range(num_classes):
            weights.append(
                (tot_pixels - pixel_nums[i]) / tot_pixels / (num_classes - 1)
            )
        weights = np.array(weights, dtype=np.float)
        # weights = torch.from_numpy(weights).float().to(masks.device)
        return weights


class KLDivergenceLoss(nn.Module):
    def __init__(self, T=1):
        super(KLDivergenceLoss, self).__init__()
        self.T = T

    def forward(self, input, target):
        log_input_prob = F.log_softmax(input / self.T, dim=1)
        # target_porb = F.softmax(target / self.T, dim=1)
        loss = F.kl_div(log_input_prob, target_porb)
        return self.T*self.T*loss # balanced


#================== Symmetric Cross Entropy Loss =========================
class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=4, ignore_index=-100, reduction='mean'):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.rce_loss = ReverseCrossEntropyLoss(num_classes=num_classes, reduction=reduction)

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        rce = self.rce_loss(inputs, targets)

        loss = self.alpha * ce + self.beta * rce 
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class ReverseCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4, ignore_index=-100, reduction='mean'):
        super(ReverseCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        if inputs.dim() == 4:
            B, C, H, W = inputs.size()
            pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
            targets = targets.view(-1)

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            pred = pred[valid]
            targets = targets[valid]

        targets_one_hot = one_hot(targets, self.num_classes)
        targets_one_hot = torch.clamp(targets_one_hot, min=1e-4, max=1.0)

        loss = -1 * torch.sum(pred * torch.log(targets_one_hot), dim=1)
        # print(loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


#===================== Normalized Symmetric Cross Entropy Loss =====================
class NormalizedSymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=4, ignore_index=-100, reduction='mean'):
        super(NormalizedSymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nce_loss = NormalizedCrossEntropyLoss(num_classes=num_classes, ignore_index=ignore_index, reduction=reduction)
        self.nrce_loss = NormalizedReverseCrossEntropyLoss(num_classes=num_classes, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, targets):
        nce = self.nce_loss(inputs, targets)
        nrce = self.nrce_loss(inputs, targets)
        
        loss = self.alpha * nce + self.beta * nrce 
        return loss


class NormalizedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4, ignore_index=-100, reduction='mean'):
        super(NormalizedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        pred = F.log_softmax(inputs, dim=1)
        B, C, H, W = inputs.size()
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        targets = targets.view(-1)

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            pred = pred[valid]
            targets = targets[valid]

        targets_one_hot = one_hot(targets, self.num_classes)
        ce =  -1 * torch.sum(targets_one_hot * pred, dim=1)
        C = -1 * torch.sum(pred, dim=1)
        nce = torch.div(ce, C)

        if self.reduction == 'mean':
            nce = nce.mean()
        elif self.reduction == 'sum':
            nce = nce.sum()

        return nce


class NormalizedReverseCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=4, ignore_index=-100, reduction='mean'):
        super(NormalizedReverseCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        pred = F.softmax(inputs, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        B, C, H, W = inputs.size()
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        targets = targets.view(-1)

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            pred = pred[valid]
            targets = targets[valid]

        targets_one_hot = one_hot(targets, self.num_classes)
        targets_one_hot = torch.clamp(targets_one_hot, min=1e-4, max=1.0)

        rce = -1 * torch.sum(pred * torch.log(targets_one_hot), dim=1)
        nrce = rce / (self.num_classes-1) / 4

        if self.reduction == 'mean':
            nrce = nrce.mean()
        elif self.reduction == 'sum':
            nrce = nrce.sum()

        return nrce


#======================================= Focal Loss ======================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, one_hot=True, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        self.one_hot = one_hot
        self.ignore_index = ignore_index

    def forward(self, input, target):
        '''
        only support ignore at 0
        '''
        if input.dim() == 4:
            B, C, H, W = input.size()
            input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
            target = target.view(-1)

        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            input = input[valid]
            target = target[valid]

        if self.one_hot: 
            target = one_hot(target, input.size(1))

        probs = F.softmax(input, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)
        log_p = probs.log()
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        
        return loss


#=============== Soft Cross Entropy Loss ========================
class SoftCrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss2d, self).__init__()

    def forward(self, inputs, targets):
        loss = 0
        inputs = -F.log_softmax(inputs, dim=1)
        for index in range(inputs.size()[0]):
            loss += F.conv2d(inputs[range(index, index+1)], targets[range(index, index+1)])/(targets.size()[2] *
                                                                                             targets.size()[3])
        return loss

#-------------------------------- helper function ----------------------------------#

def moving_average(target1, target2, alpha=1.0):
    target = 0
    target += (1.0 - alpha) * target1
    target += target2 * alpha
    return target


def one_hot(index, classes):
    # index is not flattened (pypass ignore) ############
    size = index.size()[:1] + (classes,) + index.size()[1:]
    view = index.size()[:1] + (1,) + index.size()[1:]
    #####################################################
    # index is flatten (during ignore) ##################
    # size = index.size()[:1] + (classes,)
    # view = index.size()[:1] + (1,)
    #####################################################

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)






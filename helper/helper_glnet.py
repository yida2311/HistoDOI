import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from .utils import ConfusionMatrixSeg


transformer = transforms.Compose([
    transforms.ToTensor(),
])

def resize(images, shape, label=False):
    '''
    resize PIL images
    shape: (w, h)
    '''
    resized = list(images)
    for i in range(len(images)):
        if label:
            resized[i] = images[i].resize(shape, Image.NEAREST)
        else:
            resized[i] = images[i].resize(shape, Image.BILINEAR)
    return resized

def _mask_transform(mask):
    target = np.array(mask).astype('int32')
    target[target == 255] = -1
    # target -= 1 # in DeepGlobe: make class 0 (should be ignored) as -1 (to be ignored in cross_entropy)
    return target

def masks_transform(masks, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = []
    for m in masks:
        targets.append(_mask_transform(m))
    targets = np.array(targets)
    if numpy:
        return targets
    else:
        return torch.from_numpy(targets).long().cuda()

def images_transform(images):
    '''
    images: list of PIL images
    '''
    inputs = []
    for img in images:
        inputs.append(transformer(img))
    inputs = torch.stack(inputs, dim=0).cuda()
    return inputs

def get_patch_info(shape, p_size):
    '''
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    '''
    x = shape[0]
    y = shape[1]
    n = m = 1
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < 50:
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < 50:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)

def global2patch(images, p_size):
    '''
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    '''
    patches = []; coordinates = []; templates = []; sizes = []; ratios = [(0, 0)] * len(images); patch_ones = np.ones(p_size)
    for i in range(len(images)):
        w, h = images[i].size
        size = (h, w)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        template = np.zeros(size)
        n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0])
        patches.append([images[i]] * (n_x * n_y))
        coordinates.append([(0, 0)] * (n_x * n_y))
        for x in range(n_x):
            if x < n_x - 1: top = int(np.round(x * step_x))
            else: top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1: left = int(np.round(y * step_y))
                else: left = size[1] - p_size[1]
                template[top:top+p_size[0], left:left+p_size[1]] += patch_ones
                coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])
                patches[i][x * n_y + y] = transforms.functional.crop(images[i], top, left, p_size[0], p_size[1])
        templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)).cuda())
    return patches, coordinates, templates, sizes, ratios

def patch2global(patches, n_class, sizes, coordinates, p_size):
    '''
    predicted patches (after classify layer) => predictions
    return: list of np.array
    '''
    predictions = [ np.zeros((n_class, size[0], size[1])) for size in sizes ]
    for i in range(len(sizes)):
        for j in range(len(coordinates[i])):
            top, left = coordinates[i][j]
            top = int(np.round(top * sizes[i][0])); left = int(np.round(left * sizes[i][1]))
            predictions[i][:, top: top + p_size[0], left: left + p_size[1]] += patches[i][j]
    return predictions

def template_patch2global(size_g, size_p, n, step):
    template = np.zeros(size_g)
    coordinates = [(0, 0)] * n ** 2
    patch = np.ones(size_p)
    step = (size_g[0] - size_p[0]) // (n - 1)
    x = y = 0
    i = 0
    while x + size_p[0] <= size_g[0]:
        while y + size_p[1] <= size_g[1]:
            template[x:x+size_p[0], y:y+size_p[1]] += patch
            coordinates[i] = (1.0 * x / size_g[0], 1.0 * y / size_g[1])
            i += 1
            y += step
        x += step
        y = 0
    return Variable(torch.Tensor(template).expand(1, 1, -1, -1)).cuda(), coordinates

def one_hot_gaussian_blur(index, classes):
    '''
    index: numpy array b, h, w
    classes: int
    '''
    mask = np.transpose((np.arange(classes) == index[..., None]).astype(float), (0, 3, 1, 2))
    b, c, _, _ = mask.shape
    for i in range(b):
        for j in range(c):
            mask[i][j] = cv2.GaussianBlur(mask[i][j], (0, 0), 8)

    return mask


def create_model_load_weights(model, global_fixed, device, mode=1, local_rank=0, evaluation=False, path_g=None, path_g2l=None, path_l2g=None):
    # to cuda and dirstributed
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    if global_fixed:
        global_fixed = nn.SyncBatchNorm.convert_sync_batchnorm(global_fixed)
        global_fixed.to(device)
        global_fixed = nn.parallel.DistributedDataParallel(global_fixed, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
  
    # load checkpoint
    if (mode == 2 and not evaluation) or (mode == 1 and evaluation):
        # load fixed basic global branch
        partial = torch.load(path_g)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    if (mode == 3 and not evaluation) or (mode == 2 and evaluation):
        partial = torch.load(path_g2l)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    if mode == 3:
        # load fixed basic global branch
        partial = torch.load(path_g)
        state = global_fixed.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        global_fixed.load_state_dict(state)
        global_fixed.eval()

    if mode == 3 and evaluation:
        partial = torch.load(path_l2g)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    if mode == 1 or mode == 3:
        model.module.encoder_local.eval()
        model.module.decoder_local.eval()
    else:
        model.module.encoder_global.eval()
        model.module.decoder_global.eval()
    
    return model, global_fixed


def get_optimizer(model, mode=1, learning_rate=2e-5):
    if mode == 1 or mode == 3:
        # train global
        optimizer = torch.optim.Adam([
                {'params': model.module.encoder_global.parameters(), 'lr': learning_rate},
                {'params': model.module.encoder_local.parameters(), 'lr': 0},
                {'params': model.module.decoder_global.parameters(), 'lr': learning_rate},
                {'params': model.module.decoder_local.parameters(), 'lr': 0},
                {'params': model.module.ensemble_conv.parameters(), 'lr': learning_rate},
            ], weight_decay=5e-4)
    else:
        # train local
        optimizer = torch.optim.Adam([
                {'params': model.module.encoder_global.parameters(), 'lr': 0},
                {'params': model.module.encoder_local.parameters(), 'lr': learning_rate},
                {'params': model.module.decoder_global.parameters(), 'lr': 0},
                {'params': model.module.decoder_local.parameters(), 'lr': learning_rate},
                {'params': model.module.ensemble_conv.parameters(), 'lr': learning_rate},
            ], weight_decay=5e-4)
    return optimizer


class Trainer(object):
    def __init__(self, criterion, optimizer, n_class, size_g, size_p, sub_batch_size=6, mode=1, lamb_fmreg=0.15):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_global = ConfusionMatrixSeg(n_class)
        self.metrics_local = ConfusionMatrixSeg(n_class)
        self.metrics = ConfusionMatrixSeg(n_class)
        self.n_class = n_class
        self.size_g = size_g
        self.size_p = size_p
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.lamb_fmreg = lamb_fmreg
    
    def set_train(self, model):
        model.module.ensemble_conv.train()
        if self.mode == 1 or self.mode == 3:
            model.module.encoder_global.train()
            model.module.decoder_global.train()
        else:
            model.module.encoder_local.train()
            model.module.decoder_local.train()

    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()

    def train(self, sample, model, global_fixed):
        images, labels = sample['image'], sample['mask'] # PIL images
        labels_npy = masks_transform(labels, numpy=True) # label of origin size in numpy

        images_glb = resize(images, self.size_g) # list of resized PIL images
        images_glb = images_transform(images_glb)
        labels_glb = resize(labels, (self.size_g[0] // 4, self.size_g[1] // 4), label=True) # FPN down 1/4, for loss
        labels_glb = masks_transform(labels_glb)

        if self.mode == 2 or self.mode == 3:
            patches, coordinates, templates, sizes, ratios = global2patch(images, self.size_p)
            label_patches, _, _, _, _ = global2patch(labels, self.size_p)
            predicted_patches = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
            predicted_ensembles = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
            outputs_global = [ None for i in range(len(images)) ]

        if self.mode == 1:
            # training with only (resized) global image #########################################
            outputs_global, _ = model.forward(images_glb, None, None, None)
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            ##############################################

        if self.mode == 2:
            # training with patches ###########################################
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                    label_patches_var = masks_transform(resize(label_patches[i][j : j+self.sub_batch_size], (self.size_p[0] // 4, self.size_p[1] // 4), label=True)) # down 1/4 for loss

                    output_ensembles, output_global, output_patches, fmreg_l2 = model.forward(images_glb[i:i+1], patches_var, coordinates[i][j : j+self.sub_batch_size], ratios[i], mode=self.mode, n_patch=len(coordinates[i]))
                    loss = self.criterion(output_patches, label_patches_var) + self.criterion(output_ensembles, label_patches_var) + self.lamb_fmreg * fmreg_l2
                    loss.backward()

                    # patch predictions
                    predicted_patches[i][j:j+output_patches.size()[0]] = F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                    predicted_ensembles[i][j:j+output_ensembles.size()[0]] = F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                    j += self.sub_batch_size
                outputs_global[i] = output_global
            outputs_global = torch.cat(outputs_global, dim=0)

            self.optimizer.step()
            self.optimizer.zero_grad()
            #####################################################################################

        if self.mode == 3:
            # train global with help from patches ##################################################
            # go through local patches to collect feature maps
            # collect predictions from patches
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                    fm_patches, output_patches = model.module.collect_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, template=templates[i], n_patch_all=len(coordinates[i]))
                    predicted_patches[i][j:j+output_patches.size()[0]] = F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                    j += self.sub_batch_size
            # train on global image
            outputs_global, fm_global = model.forward(images_glb, None, None, None, mode=self.mode)
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward(retain_graph=True)
            # fmreg loss
            # generate ensembles & calc loss
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    label_patches_var = masks_transform(resize(label_patches[i][j : j+self.sub_batch_size], (self.size_p[0] // 4, self.size_p[1] // 4), label=True))
                    fl = fm_patches[i][j : j+self.sub_batch_size].cuda()
                    fg = model.module._crop_global(fm_global[i:i+1], coordinates[i][j:j+self.sub_batch_size], ratios[i])[0]
                    fg = F.interpolate(fg, size=fl.size()[2:], mode='bilinear')
                    output_ensembles = model.module.ensemble(fl, fg)
                    loss = self.criterion(output_ensembles, label_patches_var)# + 0.15 * mse(fl, fg)
                    if i == len(images) - 1 and j + self.sub_batch_size >= len(coordinates[i]):
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)

                    # ensemble predictions
                    predicted_ensembles[i][j:j+output_ensembles.size()[0]] = F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                    j += self.sub_batch_size
            self.optimizer.step()
            self.optimizer.zero_grad()

        # global predictions ###########################
        outputs_global = outputs_global.cpu()
        predictions_global = [F.interpolate(outputs_global[i:i+1], images[i].size[::-1], mode='nearest').argmax(1).detach().numpy() for i in range(len(images))]
        self.metrics_global.update(labels_npy, predictions_global)

        if self.mode == 2 or self.mode == 3:
            # patch predictions ###########################
            scores_local = np.array(patch2global(predicted_patches, self.n_class, sizes, coordinates, self.size_p)) # merge softmax scores from patches (overlaps)
            predictions_local = scores_local.argmax(1) # b, h, w
            self.metrics_local.update(labels_npy, predictions_local)
            ###################################################
            # combined/ensemble predictions ###########################
            scores = np.array(patch2global(predicted_ensembles, self.n_class, sizes, coordinates, self.size_p)) # merge softmax scores from patches (overlaps)
            predictions = scores.argmax(1) # b, h, w
            self.metrics.update(labels_npy, predictions)
        return loss


class Evaluator(object):
    def __init__(self, n_class, size_g, size_p, sub_batch_size=6, mode=1, test=False):
        self.metrics_global = ConfusionMatrixSeg(n_class)
        self.metrics_local = ConfusionMatrixSeg(n_class)
        self.metrics = ConfusionMatrixSeg(n_class)
        self.n_class = n_class
        self.size_g = size_g
        self.size_p = size_p
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.test = test

        if test:
            self.flip_range = [False, True]
            self.rotate_range = [0, 1, 2, 3]
        else:
            self.flip_range = [False]
            self.rotate_range = [0]
    
    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()

    def eval_test(self, sample, model, global_fixed):
        with torch.no_grad():
            images = sample['image']
            if not self.test:
                labels = sample['mask'] # PIL images
                labels_npy = masks_transform(labels, numpy=True)

            images_global = resize(images, self.size_g)
            outputs_global = np.zeros((len(images), self.n_class, self.size_g[0] // 4, self.size_g[1] // 4))
            if self.mode == 2 or self.mode == 3:
                images_local = [ image.copy() for image in images ]
                scores_local = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]
                scores = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]

            for flip in self.flip_range:
                if flip:
                    # we already rotated images for 270'
                    for b in range(len(images)):
                        images_global[b] = transforms.functional.rotate(images_global[b], 90) # rotate back!
                        images_global[b] = transforms.functional.hflip(images_global[b])
                        if self.mode == 2 or self.mode == 3:
                            images_local[b] = transforms.functional.rotate(images_local[b], 90) # rotate back!
                            images_local[b] = transforms.functional.hflip(images_local[b])
                for angle in self.rotate_range:
                    if angle > 0:
                        for b in range(len(images)):
                            images_global[b] = transforms.functional.rotate(images_global[b], 90)
                            if self.mode == 2 or self.mode == 3:
                                images_local[b] = transforms.functional.rotate(images_local[b], 90)

                    # prepare global images onto cuda
                    images_glb = images_transform(images_global) # b, c, h, w

                    if self.mode == 2 or self.mode == 3:
                        patches, coordinates, templates, sizes, ratios = global2patch(images, self.size_p)
                        predicted_patches = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
                        predicted_ensembles = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]

                    if self.mode == 1:
                        # eval with only resized global image ##########################
                        if flip:
                            outputs_global += np.flip(np.rot90(model.forward(images_glb, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global += np.rot90(model.forward(images_glb, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2))
                        ################################################################

                    if self.mode == 2:
                        # eval with patches ###########################################
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                                output_ensembles, output_global, output_patches, _ = model.forward(images_glb[i:i+1], patches_var, coordinates[i][j : j+self.sub_batch_size], ratios[i], mode=self.mode, n_patch=len(coordinates[i]))

                                # patch predictions
                                predicted_patches[i][j:j+output_patches.size()[0]] += F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                                predicted_ensembles[i][j:j+output_ensembles.size()[0]] += F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                                j += patches_var.size()[0]
                            if flip:
                                outputs_global[i] += np.flip(np.rot90(output_global[0].data.cpu().numpy(), k=angle, axes=(2, 1)), axis=2)
                                scores_local[i] += np.flip(np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                                scores[i] += np.flip(np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                            else:
                                outputs_global[i] += np.rot90(output_global[0].data.cpu().numpy(), k=angle, axes=(2, 1))
                                scores_local[i] += np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                scores[i] += np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                        ###############################################################

                    if self.mode == 3:
                        # eval global with help from patches ##################################################
                        # go through local patches to collect feature maps
                        # collect predictions from patches
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size]) # b, c, h, w
                                fm_patches, output_patches = model.module.collect_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, template=templates[i], n_patch_all=len(coordinates[i]))
                                predicted_patches[i][j:j+output_patches.size()[0]] += F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                                j += self.sub_batch_size
                        # go through global image
                        tmp, fm_global = model.forward(images_glb, None, None, None, mode=self.mode)
                        if flip:
                            outputs_global += np.flip(np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global += np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2))
                        # generate ensembles
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                fl = fm_patches[i][j : j+self.sub_batch_size].cuda()
                                fg = model.module._crop_global(fm_global[i:i+1], coordinates[i][j:j+self.sub_batch_size], ratios[i])[0]
                                fg = F.interpolate(fg, size=fl.size()[2:], mode='bilinear')
                                output_ensembles = model.module.ensemble(fl, fg) # include cordinates

                                # ensemble predictions
                                predicted_ensembles[i][j:j+output_ensembles.size()[0]] += F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                                j += self.sub_batch_size
                            if flip:
                                scores_local[i] += np.flip(np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)), axis=3)[0] # merge softmax scores from patches (overlaps)
                                scores[i] += np.flip(np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)), axis=3)[0] # merge softmax scores from patches (overlaps)
                            else:
                                scores_local[i] += np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                scores[i] += np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                        ###################################################

            # global predictions ###########################
            outputs_global = torch.Tensor(outputs_global)
            predictions_global = [F.interpolate(outputs_global[i:i+1], images[i].size[::-1], mode='nearest').argmax(1).detach().numpy()[0] for i in range(len(images))]
            if not self.test:
                self.metrics_global.update(labels_npy, predictions_global)

            if self.mode == 2 or self.mode == 3:
                # patch predictions ###########################
                predictions_local = [ score.argmax(1)[0] for score in scores_local ]
                if not self.test:
                    self.metrics_local.update(labels_npy, predictions_local)
                ###################################################
                # combined/ensemble predictions ###########################
                predictions = [ score.argmax(1)[0] for score in scores ]
                if not self.test:
                    self.metrics.update(labels_npy, predictions)
                return predictions, predictions_global, predictions_local
            else:
                return None, predictions_global, None


def save_ckpt_model(model, cfg, score, score_global, best_pred, epoch):
    if cfg.mode == 1:
        if score_global["iou_mean"] >  best_pred:
            best_pred = score_global["iou_mean"]
            save_path = os.path.join(cfg.model_path, "%s-%d-%.5f.pth"%(cfg.model+'-'+cfg.backbone + '-' + cfg.mode_str[cfg.mode], epoch, best_pred))
    else:
        if score['iou_mean'] > best_pred:
            best_pred = score['iou_mean']
            save_path = os.path.join(cfg.model_path, "%s-%d-%.5f.pth"%(cfg.model+'-'+cfg.backbone + '-' + cfg.mode_str[cfg.mode], epoch, best_pred))
            torch.save(model.state_dict(), save_path)
    
    return best_pred


def update_log(f_log, cfg, scores_train, scores_val, epoch):
    log = ""
    if cfg.mode == 1:
        log = log + 'epoch [{}/{}] global mIoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, cfg.num_epochs, scores_train[1]['iou_mean'], scores_val[1]['iou_mean']) + "\n"
        log = log + "[train] global IoU = " + str(scores_train[1]['iou']) + "\n"
        log = log + "[train] global Dice = " + str(scores_train[1]['dice']) + "\n"
        log = log + "[train] global Dice_mean = " + str(scores_train[1]['dice_mean']) + "\n"
        log = log + "[train] global Accuracy = " + str(scores_train[1]['accuracy'])  + "\n"
        log = log + "[train] global Accuracy_mean = " + str(scores_train[1]['accuracy_mean'])  + "\n"
        log = log + "------------------------------------ \n"
        log = log + "[val] global IoU = " + str(scores_val[1]['iou']) + "\n"
        log = log + "[val] global Dice = " + str(scores_val[1]['dice']) + "\n"
        log = log + "[val] global Dice_mean = " + str(scores_val[1]['dice_mean']) + "\n"
        log = log + "[val] global Accuracy = " + str(scores_val[1]['accuracy'])  + "\n"
        log = log + "[val] global Accuracy_mean = " + str(scores_val[1]['accuracy_mean'])  + "\n"
        log += "================================\n"
    else:
        log = log + 'epoch [{}/{}] mIoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, cfg.num_epochs, scores_train[0]['iou_mean'], scores_val[0]['iou_mean']) + "\n"
        log = log + 'epoch [{}/{}] global mIoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, cfg.num_epochs, scores_train[1]['iou_mean'], scores_val[1]['iou_mean']) + "\n"
        log = log + 'epoch [{}/{}] local mIoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, cfg.num_epochs, scores_train[2]['iou_mean'], scores_val[2]['iou_mean']) + "\n"
        log = log + "[train] IoU = " + str(scores_train[0]['iou']) + "\n"
        log = log + "[train] Dice = " + str(scores_train[0]['dice']) + "\n"
        log = log + "[train] Dice_mean = " + str(scores_train[0]['dice_mean']) + "\n"
        log = log + "[train] Accuracy = " + str(scores_train[0]['accuracy'])  + "\n"
        log = log + "[train] Accuracy_mean = " + str(scores_train[0]['accuracy_mean'])  + "\n"
        log = log + "------------------------------------ \n"
        log = log + "[val] IoU = " + str(scores_val[0]['iou']) + "\n"
        log = log + "[val] Dice = " + str(scores_val[0]['dice']) + "\n"
        log = log + "[val] Dice_mean = " + str(scores_val[0]['dice_mean']) + "\n"
        log = log + "[val] Accuracy = " + str(scores_val[0]['accuracy'])  + "\n"
        log = log + "[val] Accuracy_mean = " + str(scores_val[0]['accuracy_mean'])  + "\n"
        log += "================================\n"
        log = log + "[train] global IoU = " + str(scores_train[1]['iou']) + "\n"
        log = log + "[train] global Dice = " + str(scores_train[1]['dice']) + "\n"
        log = log + "[train] global Dice_mean = " + str(scores_train[1]['dice_mean']) + "\n"
        log = log + "[train] global Accuracy = " + str(scores_train[1]['accuracy'])  + "\n"
        log = log + "[train] global Accuracy_mean = " + str(scores_train[1]['accuracy_mean'])  + "\n"
        log = log + "------------------------------------ \n"
        log = log + "[val] global IoU = " + str(scores_val[1]['iou']) + "\n"
        log = log + "[val] global Dice = " + str(scores_val[1]['dice']) + "\n"
        log = log + "[val] global Dice_mean = " + str(scores_val[1]['dice_mean']) + "\n"
        log = log + "[val] global Accuracy = " + str(scores_val[1]['accuracy'])  + "\n"
        log = log + "[val] global Accuracy_mean = " + str(scores_val[1]['accuracy_mean'])  + "\n"
        log += "================================\n"
        log = log + "[train] local IoU = " + str(scores_train[2]['iou']) + "\n"
        log = log + "[train] local Dice = " + str(scores_train[2]['dice']) + "\n"
        log = log + "[train] local Dice_mean = " + str(scores_train[2]['dice_mean']) + "\n"
        log = log + "[train] local Accuracy = " + str(scores_train[2]['accuracy'])  + "\n"
        log = log + "[train] local Accuracy_mean = " + str(scores_train[2]['accuracy_mean'])  + "\n"
        log = log + "------------------------------------ \n"
        log = log + "[val] local IoU = " + str(scores_val[2]['iou']) + "\n"
        log = log + "[val] local Dice = " + str(scores_val[2]['dice']) + "\n"
        log = log + "[val] local Dice_mean = " + str(scores_val[2]['dice_mean']) + "\n"
        log = log + "[val] local Accuracy = " + str(scores_val[2]['accuracy'])  + "\n"
        log = log + "[val] local Accuracy_mean = " + str(scores_val[2]['accuracy_mean'])  + "\n"
        log += "================================\n"
        
    print(log)
    f_log.write(log)
    f_log.flush()


def update_writer(writer, writer_info, epoch):
    for k, v in writer_info.items():
        if isinstance(v, dict):
            writer.add_scalars(k, v, epoch)
        else:
            writer.add_scalar(k, v, epoch)











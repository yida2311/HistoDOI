import os 
import torch 
import torch.nn.functional as F
import numpy as np 
import random
import math 
import cv2

from utils.data import class_to_RGB

def slide2component(slide, size, min_area, num_nodes):
    """Transform slide to connected components to construct graph
        slide: numpy array, CxHxW
    """
    mask = np.array(np.argmax(slide, axis=0), dtype='uint8')
    _, thresh0 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    _, thresh1 = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(mask, 2, 255, cv2.THRESH_BINARY)

    normal = thresh0 - thresh1
    mucosa = thresh1 - thresh2
    tumor = thresh2

    n1, label1, stats1, centroids1 = cv2.connectedComponentsWithStats(normal)
    n2, label2, stats2, centroids2 = cv2.connectedComponentsWithStats(mucosa)
    n3, label3, stats3, centroids3 = cv2.connectedComponentsWithStats(tumor)
    print(n1, n2, n3)
    res = []
    res1, num1 = parseComponent(slide, n1, label1, stats1, centroids1, size=size, min_area=min_area, maxNum=num_nodes[0])
    res2, num2 = parseComponent(slide, n2, label2, stats2, centroids2, size=size, min_area=min_area, maxNum=num_nodes[1])
    res3, num3 = parseComponent(slide, n3, label3, stats3, centroids3, size=size, min_area=min_area, maxNum=num_nodes[2])
    res = res1 + res2 + res3
    print(num1, num2, num3)

    assert num2 > 1
    
    cmp = torch.stack([c[0] for c in res], dim=0) # Nx(size+6)
    spa = torch.stack([c[1] for c in res], dim=0) # Nx4
    info = [c[2] for c in res]
    cnt = (num1, num2, num3)
    return cmp, spa, info, cnt


def parseComponent(slide, n, label, stats, centroids, size=32, min_area=64, maxNum=30):
    """parse and filter connected components
    Args:
        slide (numpy array): slide feature
        n (int): number of connected components
        label (numpy array): marked connected components mask
        stats (numpy array): n x 5, connected components' status including x, y, h, w, area
        centroids (numpy array): n x 2, connected components' centroids
        size (int): size of Fourier descriptor of connected components
        min_area (int): threshold of connected component selection
        maxNum: maximum number of selected connected components
    Output:
        res (list): connected components feature, spatial feature, information
    """
    # slide = torch.from_numpy(slide).float()
    # label = torch.from_numpy(label)
    res = []
    area_hist = []
    for i in range(1, n):
        if stats[i][2] > 2 and stats[i][3] > 2 and stats[i][4] > min_area:
            area_hist.append((stats[i][4], i))
    
    area_hist.sort(key=lambda x: x[0], reverse=True)
    if len(area_hist) > maxNum:
        area_hist = area_hist[:maxNum]

    for area, i in area_hist:
        patch = slide[:, stats[i][1]:stats[i][1]+stats[i][3], stats[i][0]:stats[i][0]+stats[i][2]]  # component feature (4 x h x w)
        mask = label[stats[i][1]:stats[i][1]+stats[i][3], stats[i][0]:stats[i][0]+stats[i][2]] # component bbox mask (h x w)
        thresh = mask == i # component binary mask (h x w)
        patch = patch * thresh # masked feature

        shape_feat = fourierDescriptor(thresh, min_descriptor=size) # shape feature (size)
        cls_feat = np.sum(patch, axis=(1, 2)) / np.sum(thresh, axis=(0, 1)) # category feature (4)
        size_feat = np.log(stats[i][2:4])  # size feature (2)
        # print(shape_feat.shape, cls_feat.shape, size_feat.shape)
        cmp = np.concatenate((shape_feat, cls_feat, size_feat), axis=None) # cmp feature (size+6)
        
        cmp = torch.FloatTensor(cmp)
        spa = torch.FloatTensor([centroids[i][1], centroids[i][0], stats[i][3], stats[i][2]]) # row, col, h, w
        info = {"bbox":stats[i][:4], "area":stats[i][4], "centroid":centroids[i]}
        res.append([cmp, spa, info]) # component, spatial, info
    
    return res, len(res)


def edgeGeneration(spa, cnt, num_edges_per_class):
    k = num_edges_per_class
    num_normal = cnt[0]
    num_mucosa = cnt[1]
    num_tumor = cnt[2]
    num_nodes = cnt[0] + cnt[1] + cnt[2]

    k1 = min(k, num_normal)
    k2 = min(k, num_mucosa)
    k3 = min(k, num_tumor)
    large = list(range(k1)) + list(range(num_normal, num_normal+k2)) + list(range(num_normal+num_mucosa, num_normal+num_mucosa+k3))
    small = [c for c in range(num_nodes) if c not in large]

    edges_index = []
    edges = []

    for i in small:
        for j in large:
            edges_index.append([i, j])
            edges_index.append([j, i])
            edges.append(relation_encoding(spa, i, j))
            edges.append(relation_encoding(spa, j, i))

    for i in range(len(large)):
        for j in range(i+1, len(large)):
            edges_index.append([large[i], large[j]])
            edges_index.append([large[j], large[i]])
            edges.append(relation_encoding(spa, large[i], large[j]))
            edges.append(relation_encoding(spa, large[j], large[i]))
    
    edges = torch.stack(edges, dim=0)
    edges_index = torch.tensor(edges_index, dtype=torch.long)

    return edges, edges_index


# def scale_encoding(scale, z=100.0):
#     """ scale:w+h+area"""
#     scale[2] = math.sqrt(scale[2])
#     scale = np.arra(scale, dtype=float)

#     return torch.FloatTensor(np.sqrt(scale)/z)


# def position_encoding(pos, channel):
#     def cal_angle(pos, idx):
#         return pos / np.power(10000, 2*(idx//2)/channel)
    
#     angles = np.array([cal_angle(pos, i) for i in range(channel)])
#     angles[0::2] = np.sin(angles[0::2])
#     angles[1::2] = np.cos(angles[1::2])

#     return torch.FloatTensor(angles)

def relation_encoding(spa, i, j):
    "Relation encoding includes spatial relation and scale relation"
    s_i = spa[i]
    s_j = spa[j]
    edge = torch.zeros(4)
    edge[0] = s_j[0] - s_i[0]  # xj - xi 
    edge[0] = np.log(edge[0]+1) if edge[0] > 0 else -np.log(-edge[0]+1)
    edge[1] = s_j[1] - s_i[1]  # yj - yi
    edge[1] = np.log(edge[1]+1) if edge[1] > 0 else -np.log(-edge[1]+1)
    edge[2] = np.log(s_j[2]/s_i[2]) #　ln(hj/hi)
    edge[3] = np.log(s_j[3]/s_i[3]) #　ln(wj/wi)

    return edge


def labelGeneration(mask, info, edges_index, num_nodes, num_edges):
    """mask:numpy array"""
    nodes_label = torch.zeros(num_nodes, dtype=torch.long) # label of nodes
    cluster_label = torch.zeros(num_nodes, dtype=torch.long) # cluster label of nodes
    for i in range(num_nodes):
        bbox = info[i]["bbox"]
        patch_mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        label = getLabelFromMask(patch_mask)
        cluster_label[i] = label
        if label == 1 or label == 3:
            nodes_label[i] = 1
        else:
            nodes_label[i] = label
        
    edges_label = torch.zeros(num_edges, dtype=torch.float) # label of edges
    for j in range(num_edges):
        edge_index = edges_index[j]
        if cluster_label[edge_index[0]] == cluster_label[edge_index[1]]:
            edges_label[j] = 1
        
    return nodes_label, edges_label


def getLabelFromMask(mask):
    hist = [0]*4
    for i in range(4):
        hist[i] = np.sum(mask==i+1)
        
    return hist.index(max(hist))



def fourierDescriptor(mask, min_descriptor=32):
    """Ref: https://www.pythonf.cn/read/3418 """
    mask = cv2.convertScaleAbs(np.array(mask, dtype=np.int32))
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_array = cnt[0][:, 0, :]
    cnt_complex = np.empty(cnt_array.shape[:-1], dtype=complex)
    cnt_complex.real = cnt_array[:, 0] #　横坐标作为实数部分
    cnt_complex.imag = cnt_array[:, 1] #　纵坐标作为虚数部分
    fourier_res = np.fft.fft(cnt_complex)
    # print(fourier_res)
    #　截断傅里叶描述子
    descriptor = np.fft.fftshift(fourier_res)
    # print(descriptor)
    center_index = len(descriptor) // 2
    low, high = center_index - min_descriptor // 2, center_index + min_descriptor // 2
    descriptor = descriptor[low:high]
    # print(descriptor)
    descriptor = np.fft.ifftshift(descriptor)

    return abs(descriptor)



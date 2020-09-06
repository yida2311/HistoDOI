import os 
import torch 
import torch.nn.functional as F
import numpy as np 
import random
import math 


def slide2component(slide, size, min_size, num_nodes):
    """Transform slide to connected components to construct graph
        slide: torch.tensor, CxHxW
    """
    mask = np.array(torch.argmax(slide, dim=0)) 
    _, thresh0 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    _, thresh1 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(img, 2, 255, cv2.THRESH_BINARY)
    normal = thresh0 - thresh1
    mucosa = thresh1 - thresh2
    tumor = thresh2

    n1, label1, stats1, centroids1 = cv2.connectedComponentsWithStats(normal)
    n2, label2, stats2, centroids2 = cv2.connectedComponentsWithStats(mucosa)
    n3, label3, stats3, centroids3 = cv2.connectedComponentsWithStats(tumor)

    res = []
    res1, num1 = parseComponent(slide, n1, label1, stats1, centroids1, size=size, min_size=min_size, maxNum=num_nodes[0])
    res2, num2 = parseComponent(slide, n2, label2, stats2, centroids2, size=size, min_size=min_size, maxNum=num_nodes[1])
    res3, num3 = parseComponent(slide, n2, label2, stats3, centroids3, size=size, min_size=min_size, maxNum=num_nodes[2])
    res = res1 + res2 + res3

    cmp = torch.stack(res[:][0], dim=0) # Nx4xsizexsize
    spa = torch.stack(res[:][1], dim=0) # Nx4
    info = res[:][2]
    cnt = (num1, num2, num3)
    return cmp, spa, info, cnt


def parseComponent(slide, n, label, stats, centroids, size=16, min_size=32 maxNum=30):
    """parse and filter connected components"""
    res = []
    for i in range(1, n):
        if stats[i][2] < min_size and stas[i][3] < min_size:
            continue
        else:
            patch = slide[:, stats[i][1]:stats[i][1]+stats[i][3], stats[i][0]:stats[i][0]+stats[i][2]]
            mask = label[stats[i][1]:stats[i][1]+stats[i][3], stats[i][0]:stats[i][0]+stats[i][2]]
            cmp = F.interpolate(patch[mask==i], size=(size, size), mode='nearest')
            spa = torch.FloatTensor([centroids[i][1], centroids[i][0], stats[i][3], stats[i][2]]) # row, col, h, w
            info = {"bbox":stats[i][:4], "area":stats[i][4], "centroid":centroids[i]}
            res.append([cmp, spa, info]) # component, spatial, info
    
    res.sort(key=lambda x: x[2]['area'], reverse=True)
    if len(res) > maxNum:
        return res[:maxNum], maxNum
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
    large = list(range(k1)) + list(range(num_normal:num_normal+k2)) + 
            list(range(num_normal+num_mucosa, num_normal+num_mucosa+k3))
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
        
    edges = torch.tensor(edges, dtype=torch.float)
    edges_index = torch.tensor(edges_index, dtype=torch.float)
    edges_label = torch.tensor(edges_label, dtype=torch.float)

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
    edge[0] = ln(edge[0]) if edge[0] > 0 else -ln(-edge[0])
    edge[1] = s_j[1] - s_i[1]  # yj - yi
    edge[1] = ln(edge[1]) if edge[1] > 0 else -ln(-edge[1])
    edge[2] = torch.log(s_j[2]/s_i[2]) #　ln(hj/hi)
    edge[3] = torch.log(s_j[3]/s_i[3]) #　ln(wj/wi)

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
            nodes_label = 1
        else:
            nodes_label = label
        
    edges_label = torch.zeros(num_edges, dtype=torch.long) # label of edges
    for j in range(num_edges):
        edge_index = edges_index[j]
        if cluster_label[edge_index[0]] == cluster_label[edge_index[1]]:
            edges_label[j] = 1
        
    return nodes_label, edges_label


def getLabelFromMask(mask):
    hist = [0]*4
    for j in range(4):
        hist[j] = np.sum(mask==i+1)
        
    return hist.index(max(hist))




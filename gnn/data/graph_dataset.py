import os
import cv2
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from .slide_parse import slide2component, edgeGeneration, labelGeneration


class DoiDataset(InMemoryDataset):
    """Dataset for OSCC HistoGraph
    Args:
        root: root directory where the dataset should be saved.
        config: Graph configuration.
        transform: A function/transform that takes  in a 'torch_geometric.data.Data' and return a tranformed version.
        pre_transform: Tranformed before save to disk.
        pre_filter: A function that takes in a 'torch_geometric.data.Data' and returns a boolean value, indicating whether the data should be included in the final dataset.    
    """
    def __init__(self, root, config, dataname='doi', train=True, root_mask=None, tranform=None, pre_transform=None, pre_filter=None):
        self.max_num_nodes = config["max_num_nodes"]  # [10,10,30]
        self.min_node_area = config["min_node_area"] #　30
        self.num_edges_per_class = config["num_edges_per_class"] # 4
        self.min_descriptor = config["min_descriptor"] #　16
        self.train = train
        self.root_mask = root_mask

        self.raw = os.path.join(root, dataname+'_raw')
        self.processed = os.path.join(root, dataname+'_processed')
        super(DoiDataset, self).__init__(root=root, transform=tranform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    #返回原始文件列表
    @property
    def raw_file_names(self):
        return []

    #返回需要跳过的文件列表
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data_list = []
        for slide_name in os.listdir(self.raw):
            slide = np.load(os.path.join(self.raw, slide_name))
            if self.train and self.root_mask:
                mask = cv2.imread(os.path.join(self.root_mask, slide_name.split('.')[0]+'.png'))
            else:
                mask = None
            print(slide_name)
            data = self.slide2graph(slide, mask=mask)
            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


    def slide2graph(self, slide, mask=None):
        nodes, spa, info, cnt = slide2component(slide, self.min_descriptor, self.min_node_area, self.max_num_nodes)
        # nodes: Nx(size+6); spa: Nx4
        num_nodes = cnt[0] + cnt[1] + cnt[2]

        edges, edges_index = edgeGeneration(spa, cnt, self.num_edges_per_class)
        num_edges = edges.size(0)

        if mask is not None:
            nodes_label, edges_label = labelGeneration(mask, info, edges_index, num_nodes, num_edges)
            data = Data(x=nodes, y=nodes_label, edge_index=edges_index.transpose(0,1), edge_attr=edges)
            data.edge_label = edges_label
        else:
            data = Data(x=nodes, edge_index=edges_index.transpose(0,1), edge_attr=edges)
        
        return data
    

    def __repr__(self):
        return '{}()'.format(self.dataname)

    
    











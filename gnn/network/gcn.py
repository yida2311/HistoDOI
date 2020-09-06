from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        n = x.size(0)
        x = self.conv1(x)
        x = F.relu(self.norm(x))
        x = self.pool(x)
        x = x.reshape((n, -1))

        return x
        

class EdgeGATConv(GATConv):
    """
    Args:
        in_channels: Size of each input node. A tuple corresponds to the sizes of source and target dimensionalities.
        out_channels: Size of each output node.
        edge_in_channels: Size of each input edge.
        edge_out_channels: Size of each output edge.
        heads: Number of multi-head-attentions.
        concat: If set to 'False', the multi-head attentions are averaged instead of concatenated.
        negative_slope: LeakyReLU angle of the negative slope.
        dropout: Dropout probability of the normalized attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training.
        add_self_loops: If set to 'False', the layer will not learn an addictive bias
        **kwargs: Additional arguments of class 'torch_geometric.nn.conv.MessagePassing'.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int,
                edge_in_channels: int=4, edge_out_channels: int=8,  
                heads: int=1, concat: bool=True, negative_slope: float=0.2, 
                dropout: float=0, add_self_loops: bool=True, bias: bool=True, **kwargs):
        super(EdgeGATConv, self).__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, add_self_loops, bias, **kwargs)
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        self.lin_edge1 == Linear(edge_in_channels, out_channels)
        self.lin_edge2 = nn.ModuleList([Linear(3*out_channels, edge_out_channels, bias=True), nn.ReLU()])
    
    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin_edge1.weight)
        glorot(self.lin_edge2.weight)
        glorot(self.lin_edge2.bias)
        glorot(self.bias)


    def forward(self, data, size=None, return_attention_weights=None):
        """
        Args:
            return_attention_weights (bool, optional): If set to 'True', will additionally return the 
                tuple (edge_index, attention_weights), holding the computed attention weights for each edge
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        H, C = self.head, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None # attention coefficients
        alpha_r: OptTensor = None

        # node linear transform
        if isinstance(x, Tensor):
            assert x.dim == 2, "Static graphs not supported in 'EdgeGATConv'."
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = alpha_r = (x_l * self.att_l).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim == 2, "Static graphs not supported in 'EdgeGATConv'."
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r*self.att_r).sum(dim=-1)
        
        assert x_l is not None
        assert alpha_l is not None

        # edge propagate
        edge_attr = self.lin_edge1(edge_attr)
        edge_attr = self.edge_propagate(edge_index, edge_attr, x_l, size=size)

        # add self loops
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                num_nodes = size[1] is size is not None else num_nodes
                num_nodes  = x_r.size(0) if x_r is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        
        # node propagate typeL (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)
        alpha = self._alpha
        self._alpha = None
        
        # mutil-head attention aggregation
        if self.concat:
            out = out.view(-1, self.heads*self.out_channels)
        else:
            out = out.mean(dim=-1)
        
        if self.bias is not None:
            out += self.bias
        
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, edge, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge, edge_index.set_value(alpha, layout='coo')
        else:
            return out, edge
    

    def edge_propagate(self, edge_index: Adj, edge_attr: Tensor, x: Tensor, size: Size=None):
        edge_index, _ = remove_self_loops(edge_index)
        size = self.__check_input__(edge_index, size)
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        x_i = self.__lift__(x, edge_index, i)
        x_j = self.__lift__(x, edge_index, j)

        out = self.lin_edge2(torch.cat([x_i, edge_attr, x_j]))
        out = F.relu(out)

        return out

    
    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor, ptr: OptTensor, size_i:Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_i+alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = nn.functional.dropout(alpha, p=self.dropout, training=self.training)
        return x_j*alpha.unsqueeze(-1)
    

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, edge_in_channels={}, edge_out_channels={}, heads={})'.format(
                                            self.__class__.__name__, 
                                            self.in_channels,
                                            self.out_channels,
                                            self.edge_in_channels,
                                            self.edge_out_channels
                                            self.heads)



class DoiNet(nn.Module):
    def __init__(self, num_classes):
        super(DoiNet, self).__init__()
        self.stem = Stem(in_channels=4, out_channels=64) #　out shape: N x 64
        self.gat_conv_1 = EdgeGATConv(64, 32, 4, 8, heads=3, concat=False, dropout=0, add_self_loops=True)
        self.gat_conv_2 = EdgeGATConv(32, num_classes, 8, 1, heads=1, concat=True, dropout=0, add_self_loops=True)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.stem()
        x, edge_attr = self.gat_conv_1(x, edge_index, edge_attr)
        x = F.relu(x)
        x, edge_attr = self.gat_conv_2(x, edge_index, edge_attr)
        # x = self.softmax(x)
        edge_attr = self.sigmoid(edge_attr)

        return x, edge_attr
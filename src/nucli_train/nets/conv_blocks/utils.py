

import numpy.random as random
import numpy as np 


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse import SparseTensor


import torch
import torch.nn as nn
from torchsparse import SparseTensor
import torchsparse


def decompose(x: SparseTensor) -> torch.Tensor:
    coords = x.coords  # (N, 4)
    batch_indices = coords[:, 0]  # (N,)

    decomposed = []
    for b in batch_indices.unique():
        decomposed.append((batch_indices == b).nonzero(as_tuple=False).squeeze())
    return decomposed

def to_sparse(x):
    """
    Convert a 5D dense tensor [B, C, D, H, W] into a torchsparse SparseTensor.
    Args:
        x (torch.Tensor): Dense tensor of shape [B, C, D, H, W]
    Returns:
        torchsparse.SparseTensor: Sparse tensor representation.
    """
    assert x.ndim == 5, "Input tensor must be 5D (B, C, D, H, W)"
    B, C, D, H, W = x.shape
    x_mask = x.sum(dim=1)  
    coords = x_mask.nonzero(as_tuple=False) 
    feats = x[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]]  # [N, C]
    coords = coords.int()
    return torchsparse.SparseTensor(coords=coords, feats=feats)




class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1)) # [1, C, 1, 1, 1] we suppose 3D volumes (PET Scans)
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1)) # [1, C, 1, 1, 1] we suppose 3D volumes (PET Scans)

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x



class SparseGRN(nn.Module):
    """GRN layer adapted for torchsparse SparseTensor.
    
    For the formula, see: https://arxiv.org/pdf/2301.00808

    Adam. 
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: SparseTensor):
        # x.feat: [N, C]
        Gx = torch.norm(x.feats, p=2, dim=0, keepdim=True)  # [1, C]
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)   # [1, C]

        out_feat = self.gamma * (x.feats * Nx) + self.beta + x.feats

        return SparseTensor(feats=out_feat, coords=x.coords)

class SparseDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    Args:
        drop_prob (float): Probability of dropping a path.
        scale_by_keep (bool): If True, scales the output by 1/(1 - drop_prob) to keep the expected value the same as without dropping.
    Reference
    Adam.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(SparseDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: SparseTensor):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob

        mask = torch.cat([ # The masking is done per batch index
            torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
            else torch.zeros(len(_)) for _ in decompose(x)
        ]).view(-1, 1)

        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)

        print(mask)

        return SparseTensor(
            feats=x.feats * mask,
            coords=x.coords
        )

class SparseDepthWise3D(nn.Module): 
    """
    Depthwise Convolution for SparseTensors.
    This first implementation is not the most efficient one, but it is for testing. 
    The goal is to apply a single convolution per channel, thus leveraging the usage of 
    a loop. 
    This implementation is applicable only for 3D volumes, not 2D images.
    Args: 
        in_channels (int): Number of input channels. 
        kernel_size (int): Kernel size.
        stride (int): Stride.
        padding (int): Padding.
        bias (bool): If True, adds a learnable bias to the output. Default: False.

    Adam.
    """
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SparseDepthWise3D, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        convolution_list = []
        for c in range(in_channels): 
            convolution_list.append(
                torchsparse.nn.Conv3d(
                    in_channels=1, 
                    out_channels=1, 
                    kernel_size=kernel_size, 
                    stride=stride, 
                    padding=padding, 
                    bias=bias
                )
            )
        self.convolutions = nn.ModuleList(convolution_list)

    def combine(self, convolved_tensors_coords, convolved_tensors_feats): 
        """
        We computed a convolution per channel, we now need to combine each of these SparseTensors into a single one. 
        Args: 
            convolved_tensors (List[SparseTensor]): List of SparseTensors, each corresponding to a convolved channel.
        Returns: 
            SparseTensor: Combined SparseTensor.

        Adam.
        """
        num_channels = len(convolved_tensors_coords)

        merged_coords = torch.cat(convolved_tensors_coords, dim=0)  
        merged_coords = torch.unique(merged_coords, dim=0)

        merged_np = merged_coords.cpu().numpy()
        sorted_indices_np = np.lexsort([merged_np[:, i] for i in reversed(range(4))])
        sorted_indices = torch.from_numpy(sorted_indices_np).to(merged_coords.device) # The goal here is to sort by last dimension, then third, then second, then first.

        merged_coords = merged_coords[sorted_indices]

        merged_feats = torch.zeros((merged_coords.shape[0], num_channels), device=convolved_tensors_feats[0].device)
        for c in range(num_channels):
            coords = convolved_tensors_coords[c]
            feats = convolved_tensors_feats[c]
            indices = torch.nonzero((merged_coords[:, None] == coords).all(-1), as_tuple=False)[:, 1]
            merged_feats[indices, c] = feats.squeeze()
        
        return SparseTensor(feats=merged_feats, coords=merged_coords)

    
    def forward(self, x: SparseTensor) -> SparseTensor:
        # x.feat: [N, C] => We need to extract each channel individually and apply the corresponding convolution
        convolved_tensors_coords = []
        convolved_tensors_feats = []
        for c in range(self.in_channels):
            channel_feat = x.feats[:, c].unsqueeze(1)  # [N, 1] => Need to keep the 2D shape for the convolution
            channel_tensor = SparseTensor(feats=channel_feat, coords=x.coords)
            result = self.convolutions[c](channel_tensor)
            convolved_tensors_coords.append(result.coords)
            convolved_tensors_feats.append(result.feats)

        
        # We now have a list of SparseTensors that correspond each to a convolved channel
        # We need to combine them into a single SparseTensor
        return self.combine(convolved_tensors_coords, convolved_tensors_feats)


        


if __name__ == "__main__":
    import torch
    from torchsparse import nn as spnn

    B, C, D, H, W = 2, 4, 16, 16, 16
    x = torch.randn(B, C, D, H, W)

    x[torch.rand_like(x) < 0.7] = 0

    x = x.to('cuda')
    sparse_x = to_sparse(x).cuda()

    my_depthwise = SparseDepthWise3D(in_channels=C, kernel_size=2, stride=2).cuda()
    classical_depthwise = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=2, stride=2, groups=C, bias=False).cuda()
    for c in range(C):
        with torch.no_grad():
            my_depthwise.convolutions[c].kernel.copy_(classical_depthwise.weight[c, 0].flatten().view(8, 1, 1))



    sparse_y = my_depthwise(sparse_x)
    classical_y = classical_depthwise(x)

    y = torchsparse.utils.to_dense(sparse_y.feats, sparse_y.coords, spatial_range=(B, D//2, H//2, W//2)).permute(0, 4, 1, 2, 3)

    print(y.shape)
    print(classical_y.shape)

    print(y - classical_y)

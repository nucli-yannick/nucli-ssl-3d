import torchsparse
import torch

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

if __name__ == "__main__":
    x = torch.randn(2, 4, 32, 32, 32).cuda()
    sparse_x = to_sparse(x)
    densified_x = torchsparse.utils.to_dense(sparse_x.feats, sparse_x.coords, (2, 32, 32, 32)).permute(0, 4, 1, 2, 3)
    print(x - densified_x)
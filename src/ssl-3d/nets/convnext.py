from typing import Type
import torchsparse

import torch.nn as nn
import torch 
import os


import math

from .conv_blocks.conv_blocks import (
    SparseConvNeXtV2Block3D, 
    ConvNeXtV2Block3D, 
    SparseResBlock3D, 
    DepthWiseSparseConvNeXtV2Block3D, 
    ResBlock3D
    )


from .builders import ARCHITECTURE_BUILDERS_REGISTRY

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



class ConvNeXtV2(nn.Module): 
    """ Sparse ConvNeXtV2 
    Args:
        in_chans (int): Number of input channels. (default: 1 because PET scans)
        num_classes (int): Number of output classes. (default: 1)
        depths (tuple(int)): Number of blocks at each stage. (default: [3, 3, 9, 3])
        dims (int): Feature dimensions at each stage. (default: [96, 192, 384, 768])
        drop_path_rate (float): Stochastic depth rate. (default: 0.0)
        head_init_scale (float): Initial scaling value for classifier weights and biases
        block: block type to use
    Adam.
    """

    def __init__(
        self, 
        block,
        in_chans=1, # PET or CT scans
        num_classes=1, 
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.0, 
        head_init_scale=1.0
        ):
        super(ConvNeXtV2, self).__init__()

        self.Block = block

        self.depths = depths

        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv3d(
                in_channels=in_chans,
                out_channels=dims[0],
                kernel_size=4,
                stride=4
            ),
            nn.GroupNorm(1, dims[0], 1e-6)  # GroupNorm with 1 group to match LayerNorm behavior
        ) # First block of the architecture. His goal is to reduce the spatial dimensions by 4x.
          # This is actually a 4x4 patch embedding (like in ViT).

        self.downsample_layers.append(stem)

        for i in range(3): 
            downsample_layer = nn.Sequential(
                nn.GroupNorm(1, dims[i], 1e-6),  #
                nn.Conv3d(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    kernel_size=2,
                    stride=2
                )
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList() # For now, we suppose the number of stages is 4 (hard coded)
        # in a future, this could be a parameter of the class.

        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # We do this because we want the first layers to be more probably dropped than the last ones. 
        cur = 0
        for i in range(4): 
            stage = nn.Sequential(
                *[self.Block(dim=dims[i], drop_path=drop_rates[cur + j]) for j in range(depths[i])]
            ) # We apply the Block depths[i] times, with the corresponding drop path rate.
            self.stages.append(stage)
            cur += depths[i]
            # While we advance through the network, we increase the drop rate

        self.norm = nn.GroupNorm(1, dims[-1], 1e-6)  # Final normalization layer
        self.head = nn.Linear(dims[-1], num_classes)  # Final classification layer


        # We don't want the head to be initialized with too big values and ruin the learnt embeddings during pre-training => scale them
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m): 
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    

    def forward_features(self, x): 
        for i in range(4): 
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-3, -2, -1]))  # Global average pooling over the spatial dimensions (N, C, H, W) -> (N, C)

    def forward(self, x): 
        x = self.forward_features(x)
        x = self.head(x)
        return x



class SparseConvNeXtV2(nn.Module): 
    """ Sparse ConvNeXtV2 
    Args:
        in_chans (int): Number of input channels. (default: 1 because PET scans)
        depths (tuple(int)): Number of blocks at each stage. (default: [3, 3, 9, 3])
        dims (int): Feature dimensions at each stage. (default: [96, 192, 384, 768])
        drop_path_rate (float): Stochastic depth rate. (default: 0.0)
        head_init_scale (float): Initial scaling value for classifier weights and biases
        block: block type to use
    This is the sparse version of ConvNeXtV2, again for 3D convolutions.
    Adam.
    """

    def __init__(
        self, 
        encoder_block,
        decoder_block,
        in_chans=1, # PET or CT scans
        depths=[3, 3, 3, 3], 
        dims=[96, 192, 192, 384], 
        drop_path_rate=0.0, 
        head_init_scale=1.0, 
        decoder_embed_dim=256,
        decoder_depth=1, 
        patch_size=32, 
        lr=3e-4, 
        learnable_mask_token=False
        ):

        super(SparseConvNeXtV2, self).__init__()

        self.Block = encoder_block

        self.dims = dims

        self.depths = depths

        self.downsample_layers = nn.ModuleList()

        self.patch_size = patch_size

        # Encoder Part
        self.num_stages = len(depths)


        # This part is done densely 
        stem = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_chans,
                out_channels=dims[0],
                kernel_size=4,  # To change, normally 4
                stride=4 # To change, normally 4
            ),
            torch.nn.GroupNorm(1, dims[0], 1e-6)  # GroupNorm with 1 group to match LayerNorm behavior
        )

        self.downsample_layers.append(stem)

        for i in range(self.num_stages - 1): 
            downsample_layer = nn.Sequential(
                torchsparse.nn.GroupNorm(1, dims[i], 1e-6),  #
                torchsparse.nn.Conv3d(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    kernel_size=2,
                    stride=2
                )
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # For now, we suppose the number of stages is 4 (hard coded)
        # in a future, this could be a parameter of the class.

        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # We do this because we want the first layers to be more probably dropped than the last
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[self.Block(dim=dims[i], drop_path=drop_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            # While we advance through the network, we increase the drop rate

        


        # Decoder Part
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.DecoderBlock = decoder_block

        self.proj = nn.Conv3d(
            in_channels=dims[-1],
            out_channels=decoder_embed_dim,
            kernel_size=1
        )

        if learnable_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1, 1)) # Each channel has its own mask token
        else: 
            self.mask_token = torch.zeros(1, decoder_embed_dim, 1, 1, 1, requires_grad=False).cuda()
            

        decoder = [self.DecoderBlock(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)

        # This is the patch size used in the encoder
        self.pred = nn.Conv3d(
            in_channels=decoder_embed_dim,
            out_channels=self.patch_size ** 3 * in_chans,
            kernel_size=1
        )




        
    def _init_weights(self, m): 
        if isinstance(m, (torchsparse.nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)

    def upsample_mask(self, mask, scale):
        """
        The goal of this function is to upsample a 3D mask tensor to the desired scale.
        Args:
            mask (torch.Tensor): 2D tensor of shape [B, p*p*p] where N is the vectorization of a pxpxp patch.
            scale (int): The scale factor to upsample the mask.
        Returns: 
            torch.Tensor: Upsampled mask of shape [B, p*scale, p*scale, p*scale].
        """ 
        assert len(mask.shape) == 2, "Mask must be a 2D tensor because we are working with 3D Volumes"
        epsilon = 1e-3
        p = int(mask.shape[1] ** (1/3) + epsilon) 
        return mask.reshape(-1, p, p, p).\
            repeat_interleave(scale, axis=1).\
            repeat_interleave(scale, axis=2).\
            repeat_interleave(scale, axis=3)

    def forward_decoder(self, x, mask): 
        print(f"Decoder input x shape: {x.shape}")
        x = self.proj(x)
        print(f"x shape after projection: {x.shape}")

        #mask = self.upsample_mask(mask, 2**(- 1)) # So we scale by 8

        B, C, D, H, W = x.shape
        mask = mask.reshape(-1, D, H, W).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(B, 1, D, H, W)
        print(f"Mask shape after reshaping: {mask.shape}")
        print(f"Mask token shape: {mask_token.shape}")


        x = x * (1. - mask) + mask_token * mask

        x = self.decoder(x)
        print(f"x shape after decoder: {x.shape}")

        pred = self.pred(x)
        print(f"pred shape: {pred.shape}")


        return pred



    def forward(self, x, mask): 
        """
        Forward pass of the Sparse ConvNeXtV2 model.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
            mask (torch.Tensor): Mask tensor of shape [B, N] where N is
                the vectorization of a pxpxp patch.
        Returns:
            torch.Tensor: Output tensor of shape [B, C, D, H, W].
        The usage of sparse tensors is done in the intermidiate layers.

        Adam.
        """
        print(f"Original x shape: {x.shape}")
        original_mask = mask.clone()

        epsilon = 1e-3
        p = int(mask.shape[1] ** (1/3) + epsilon)

        print(f"Original mask shape: {mask.shape}")
        num_stages = len(self.stages) 

        # patch embedding
        # At this point, this is still dense
        x = self.downsample_layers[0](x)  # Application of the stem layer
        print(f"x shape after stem: {x.shape}")

        special_size = x.shape[2] # D, H, W should be the same

        mask = self.upsample_mask(mask, int(special_size // p)) 
        print(f"Mask shape after upsampling: {mask.shape}")
        mask = mask.unsqueeze(1).type_as(x) # [B, 1, D, H, W]


        x *= (1.-mask)  # Apply the mask to the input tensor

        self.original_shape = list(x.shape)

        # We sparse only here 
        x = to_sparse(x)  # Convert to sparse tensor
        for i in range(self.num_stages): 
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages[i](x)
            if i > 0: 
                ### We update the shape
                self.original_shape[1] = self.dims[i]
                self.original_shape[2] = self.original_shape[2] // 2
                self.original_shape[3] = self.original_shape[3] // 2
                self.original_shape[4] = self.original_shape[4] // 2
            
            

        # densification
        x = torchsparse.utils.to_dense(x.feats, x.coords, spatial_range=(self.original_shape[0], self.original_shape[2], self.original_shape[3], self.original_shape[4])).permute(0, 4, 1, 2, 3)
        print(f"x shape after encoder: {x.shape}")
        # We return a dense tensor of shape [B, C, D, H, W]
        x = self.forward_decoder(x, original_mask)
        print(f"x shape after decoder: {x.shape}")

        return x


    def get_optimizer(self):
        k = 1/16
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=3e-4
            #lr=math.sqrt(k)*1e-4
        )
        return optimizer

    def parts_to_save(self):
        """
        Returns a dict with only the trainable parts of the model.
        """
        return {'sparse-convnext': self.state_dict()}

    def load_checkpoint(self, checkpoint_dir, epoch):
        self.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'sparse-convnext_epoch_{epoch}.pt')))





@ARCHITECTURE_BUILDERS_REGISTRY.register('convnextv2-3d')
def build_convnextv2_3d_from_cfg(cfg): 
    if cfg.get('encoder_block') == "SparseConvNeXtV2Block3D": 
        encoder_block = SparseConvNeXtV2Block3D
    elif cfg.get('encoder_block') == "SparseResBlock3D":
        encoder_block = SparseResBlock3D
    elif cfg.get('encoder_block') == "DepthWiseSparseConvNeXtV2Block3D": 
        encoder_block = DepthWiseSparseConvNeXtV2Block3D
    else:
        raise ValueError(f"Unknown encoder block type: {cfg.get('encoder_block')}")
    
    if cfg.get('decoder_block') == "ConvNeXtV2Block3D": 
        decoder_block = ConvNeXtV2Block3D
    elif cfg.get('decoder_block') == "ResBlock3D":
        decoder_block = SparseResBlock3D
    
    model = SparseConvNeXtV2(
        encoder_block = encoder_block, 
        decoder_block = ConvNeXtV2Block3D, 
        in_chans=cfg.get('in_chans', 1),
        depths=cfg.get('depths', [3, 3, 3, 3]),
        dims=cfg.get('dims', [96, 192, 192, 384]),
        drop_path_rate=cfg.get('drop_path_rate', 0.0),
        head_init_scale=cfg.get('head_init_scale', 1.0),
        decoder_embed_dim=cfg.get('decoder_embed_dim', 256),
        decoder_depth=cfg.get('decoder_depth', 1),
        patch_size=cfg.get('patch_size', 32), 
        lr=cfg.get('lr', 3e-4), 
        learnable_mask_token=cfg.get('learnable_mask_token', False)
    )

    return model







from __future__ import annotations

import torch.nn as nn
import torch

from .builder import LOSSES_REGISTRY
import torch.nn.functional as F


@LOSSES_REGISTRY.register('L1Loss')
def l1_loss():
    """L1 loss function."""
    class L1Loss(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = nn.L1Loss()

        def forward(self, input, target):
            return self.loss_fn(input, target)

    return L1Loss()


# The following loss will be used by VoCo
@LOSSES_REGISTRY.register('VoCoLoss')
def voco_loss(): 
    """VoCo loss function."""
    class VoCoLoss(nn.Module): 

        def __init__(self, pred_weight=1, reg_weight=1): 
            print("VoCo loss initialized")
            super().__init__()
            self.pred_weight = pred_weight
            self.reg_weight = reg_weight
        
        def prediction_loss(self, base_embeddings, target_embeddings, gt_overlaps):
            """
            Computes the prediction loss based on cosine similarity between base and target embeddings.
            """

       
            base_embeddings_de = base_embeddings.detach() 
            pred_similarity = F.cosine_similarity(
                base_embeddings_de[:, None, :, :], target_embeddings[:, :, None, :], dim=-1
            ) 
            logits = F.relu(pred_similarity)

            print("Logits shape:", logits.shape)

            pos_dist = torch.abs(gt_overlaps - logits)
            pos_pos = torch.where(gt_overlaps > 0, torch.ones_like(gt_overlaps), torch.zeros_like(gt_overlaps))
            # pos_loss = ((-torch.log(1 - pos_dist + 1e-6)) * pos_pos).sum() / (pos_pos.sum())
            pos_loss = ((-torch.log(1 - pos_dist + 1e-6)) * gt_overlaps).sum() / (gt_overlaps.sum())    # use overlap factor
            neg_loss = ((logits**2) * (1 - pos_pos)).sum() / (1 - pos_pos + 1e-6).sum()

            l_pred = pos_loss + neg_loss
            return {"loss": l_pred, "matrix": logits}

        def regularization_loss(self, base_embeddings: torch.Tensor) -> float:
            """
            Computes the regularization loss based on cosine similarity between base embeddings.
            Args:
                base_embeddings (torch.Tensor): Base embeddings of shape [B, N_base, D]
            """
            inter_crop_similarity = F.cosine_similarity(
                base_embeddings[:, None, :, :],
                base_embeddings[:, :, None, :],
                dim=-1,
            )
            
            inter_crop_sim_relu = F.relu(inter_crop_similarity)

            up_tri = torch.ones(
                inter_crop_sim_relu.shape[-2], inter_crop_sim_relu.shape[-1], device=inter_crop_sim_relu.device
            ).triu(diagonal=1)[None, ...]

            upper_triangular = up_tri * inter_crop_sim_relu
            N = upper_triangular.shape[-1]


            l_reg = torch.mean(torch.sum(upper_triangular, dim=(-2, -1)) * 2 / (N * (N - 1)))
            return {
                "loss": l_reg, 
                "matrix": inter_crop_similarity
            }
        
        def forward(self, base_embeddings: torch.Tensor, target_embeddings: torch.Tensor, gt_overlaps: torch.Tensor):
            """
            Computes the VoCo loss based on prediction and regularization losses.
            Args:
                base_embeddings (torch.Tensor): Base embeddings of shape [B, N_base, D]
                target_embeddings (torch.Tensor): Target embeddings of shape [B, N_target, D]
                gt_overlaps (torch.Tensor): Ground truth overlaps of shape [B, N_target, N_base]
            Returns:
                torch.Tensor: Computed VoCo loss.
            """
            l_pred_all = self.prediction_loss(base_embeddings, target_embeddings, gt_overlaps)
            l_pred = l_pred_all["loss"]
            l_reg_all = self.regularization_loss(base_embeddings)
            l_reg = l_reg_all["loss"]

            return {
                "main": self.pred_weight * l_pred + self.reg_weight * l_reg, 
                "pred_loss": l_pred_all,
                "reg_loss": l_reg_all, 
                }
        
    return VoCoLoss()




@LOSSES_REGISTRY.register('ConvnextMAELoss')
def mae_convnext_loss(patch_size):
    class MAEConvnextLoss(nn.Module):
        def __init__(self, patch_size): 
            super().__init__()
            self.patch_size = patch_size

        def patchify(self, imgs): 
            """
            imgs: (B, 1, D, H, W)
            x: (B, L, patch_size**3 * 1) because 1 is the channel number$
            The goal of this function is to convert he 3D image into a sequence ot patches
            """
            p = self.patch_size
            assert imgs.shape[2] == imgs.shape[3] and imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0

            
            batch_size, _, axis_size, H, W = imgs.shape
            d = h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(batch_size, 1, d, p, h, p, w, p))
            x = torch.einsum('bcdefghi->bcdfhegi', x)
            x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3)) # L = d * h * w
            return x

        def unpatchify(self, x): 
            """
            x: (B, L, patch_size**3 * 1)
            imgs: (B, 1, D, H, W)
            """
            epsilon = 1e-5
            print("Unpatchify input shape: ", x.shape)
            p = self.patch_size
            h = w = d = int(x.shape[1]**(1/3) + epsilon)
            assert h * w * d == x.shape[1]

            x = x.reshape(shape=(x.shape[0], 1, d, h, w, p, p, p))
            x = torch.einsum('bcdhwpqs->bcdphqws', x)
            imgs = x.reshape(shape=(x.shape[0], 1, d * p, h * p, w * p))
            return imgs

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

        
        def forward(self, inputs, preds, mask): 
            if len(preds.shape) == 5: 
                B, C, D, H, W = preds.shape
                preds = preds.reshape(B, C, -1)
                preds = preds.permute(0, 2, 1) 

            preds = self.unpatchify(preds)
            target = inputs
            mask = self.upsample_mask(mask, self.patch_size).unsqueeze(1)
            print("Pred shape: ", preds.shape)
            print("Target shape: ", target.shape)
            print("Mask shape: ", mask.shape)
            loss = (preds - target) ** 2
            loss = loss.mean(dim=-1)  # [B, L], mean loss per
            loss = (loss * mask).sum() / mask.sum() # Do a run without mask to see 
            return {
                "main": loss
            }

    return MAEConvnextLoss(patch_size)
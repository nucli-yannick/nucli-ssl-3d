from __future__ import annotations

from nucli_train.models import ImageTranslationModel
from nucli_train.utils.grad import set_grad
import torch.nn as nn
import torch

class TranslationModelWithDiscriminator(ImageTranslationModel):
    def __init__(self, model, discriminator, discr_loss_coef=0.05, loss_functions=None, metrics={}, optimizer=None):
        super().__init__(model, loss_functions, metrics, optimizer)
        self.discriminator = discriminator
        self.discr_obj = nn.BCEWithLogitsLoss()

        self.coef = discr_loss_coef
        self.D_opt = torch.optim.Adam(self.discriminator.parameters(),
            lr=1e-4)

    def cuda(self):
        super().cuda()
        self.discriminator.cuda()

    def get_params(self):
        params = super().get_params()
        params["paramsD"] = sum(p.numel() for p in self.discriminator.parameters()) / 10**6
        return params

    def train_step(self, batch):
        



        inputs, targets = batch["input"].cuda(), batch["target"].cuda()

        outputs = self.network(inputs)

        losses = self.get_losses(outputs, targets)

        set_grad(self.discriminator, False)
        
        discriminator_pred = self.discriminator(torch.cat((outputs, inputs), dim=1))

        adv_loss = self.discr_obj(discriminator_pred, torch.ones_like(discriminator_pred).cuda())

        losses['adv_loss'] = adv_loss

        if "value" not in losses.keys():
            losses["value"] = 0.0

        losses["value"] += self.coef*adv_loss


        yield losses

        set_grad(self.discriminator, True)

        # Discriminator loss
        d_fake = self.discriminator(torch.cat((outputs.detach(), inputs), dim=1))
        d_loss_fake = self.discr_obj(d_fake, torch.zeros_like(d_fake))
        d_real = self.discriminator(torch.cat((targets, inputs), dim=1))
        d_loss_real = self.discr_obj(d_real, torch.ones_like(d_real))
        losses = {}
        losses['d_loss_real'] = d_loss_real
        losses['d_loss_fake'] = d_loss_fake
        losses['value'] = losses['d_loss_real'] + losses['d_loss_fake']

        yield losses

    def get_optimizers(self):
        return [self.opt, self.D_opt]
    
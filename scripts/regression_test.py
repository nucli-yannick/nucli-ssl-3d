from __future__ import annotations

from nucli_train.models.builders import build_image_translation_model
import yaml

from nucli_train.training import Trainer
from nucli_train.data_management.dataset import Blosc2DatasetRegressionTest

from nucli_train.val.evaluators import RegressionEvaluator

from nucli_train.models.nuclarity import Nuclarity

from torch.utils.data import DataLoader

checkpoints = '/home/vicde/nucli-train/experiments/nuclarity_data/basic-wide'

model_cfg = yaml.safe_load(open('./configs/wide_nuclarity_model.yaml'))

data_dir = '/home/vicde/blosc2_64'

dataloader = DataLoader(Blosc2DatasetRegressionTest(data_dir, meta_path='regression.yaml'), batch_size=32, num_workers=8)

val_loader = { 'regression' : { 'loader' : dataloader, 'evaluators' : [RegressionEvaluator('/home/vicde/nucli-train/experiments/nuclarity_data/unet_wide/results')], 'interval' : 1}}

model = build_image_translation_model(model_cfg['model'])

model.network.load_checkpoint('/home/vicde/nucli-train/experiments/nuclarity_data/basic-wide/', 1500)

model.cuda()
model.eval()





trainer = Trainer(model, val_loaders=val_loader)

trainer.validate(1500)

dataloader = DataLoader(Blosc2DatasetRegressionTest(data_dir, meta_path='regression.yaml'), batch_size=16, num_workers=8)

val_loader = { 'regression' : { 'loader' : dataloader, 'evaluators' : [RegressionEvaluator('/home/vicde/nucli-train/experiments/nuclarity_data/nuclarity/results')], 'interval' : 1}}

model = Nuclarity('/home/vicde/models/nuclarity/model-checkpoints/low-noise/checkpoints/epoch=359-val_nucli_score=15.17528.ckpt', '/home/vicde/models/nuclarity/model-checkpoints/high-noise/checkpoints/epoch=314-val_nucli_score=41.96758.ckpt')

trainer = Trainer(model, val_loaders=val_loader)
trainer.validate(0)
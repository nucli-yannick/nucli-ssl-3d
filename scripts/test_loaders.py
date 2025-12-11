from __future__ import annotations

from nucli_train.data_management.builders import build_dataloaders

import yaml
build_dataloaders(yaml.safe_load(open('/home/ec2-user/volume/nucli-train/configs/example_data.yaml')))
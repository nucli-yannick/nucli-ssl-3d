from __future__ import annotations

from nucli_train.models.builders import build_image_translation_model

import yaml

if __name__=='__main__':
    cfg = yaml.safe_load(open('configs/example.yaml', 'r'))
    build_image_translation_model(cfg['model'])
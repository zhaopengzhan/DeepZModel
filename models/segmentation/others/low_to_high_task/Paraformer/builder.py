import warnings

import numpy as np

from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling_L2HNet import L2HNet

from pathlib import Path

from models import DeepZMODELS

@DeepZMODELS.register_module('Paraformer')
def build_paraformer(vit_patches_size=16,
                     img_size=224,
                     width=64,
                     num_classes=17,
                     in_channels=3,
                     pretrained=True):
    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    net = ViT_seg(config_vit,
                  backbone=L2HNet(width=width, image_band=in_channels),
                  img_size=img_size,
                  num_classes=num_classes)
    root_dir = Path(__file__).resolve().parent
    pretrain_path = root_dir / 'pre-train_model'/'imagenet21k' / 'ViT-B_16.npz'
    if pretrained:
        if not pretrain_path.exists():
            warnings.warn('Pretrained model not found. Please download at: '
                          'https://drive.google.com/file/d/10Ao75MEBlZYADkrXE4YLg6VObvR0b2Dr/view?usp=sharing.'
                          f'and put it in this {pretrain_path}')

        net.load_from(weights=np.load(pretrain_path))

    return net

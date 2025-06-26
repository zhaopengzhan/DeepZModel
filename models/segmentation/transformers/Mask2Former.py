import os
import warnings

import torch
from torch import nn
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from models import DeepZMODELS

# @DeepZMODELS.register_module('Mask2Former')
class Mask2Former(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, model_id: str = "facebook/mask2former-swin-tiny-ade-semantic"):
        super(Mask2Former, self).__init__()
        self.model_id = model_id
        self.in_chans = in_channels
        self.num_classes = num_classes

        root_dir = r'F:/cache_hf'
        local_dir = os.path.join(root_dir, model_id)
        self.processor, self.model = self.load_model(model_id, local_dir)

        self._init()
        pass

    def _init(self):
        # 1. in
        old_conv = self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection
        self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=self.in_chans,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode
        )

        # 2. out
        old_cls = self.model.class_predictor
        self.model.class_predictor = nn.Linear(
            in_features=old_cls.in_features,
            out_features=old_cls.out_features,
            bias=(old_cls.bias is not None)
        )

        pass

    def convert_semantic_to_mask2former_labels(self, label_map: torch.Tensor, ignore_index: int = 0):
        """
        将语义分割的 `one batch` per-pixel label map 转换为 Mask2Former 所需格式：
        mask_labels: [num_classes_found, H, W]
        class_labels: [num_classes_found]
        """
        unique_classes = label_map.unique()
        mask_list = []
        class_list = []

        for cls in unique_classes:
            cls = cls.item()
            if cls == ignore_index:
                continue
            binary_mask = (label_map == cls).float()  # [H, W]
            mask_list.append(binary_mask)
            class_list.append(torch.tensor([cls]).long())

        # return mask_list, class_list
        mask_labels = torch.stack(mask_list, dim=0)  # [num_classes_found, H, W]
        class_labels = torch.tensor(class_list).long()  # [num_classes_found]
        return mask_labels, class_labels

    def forward(self, image, label_map=None):
        b, c, h, w = image.shape

        if label_map is None:
            outputs = self.model(pixel_values=image)

        else:
            mask_labels_list = []
            class_labels_list = []
            for i in range(b):
                mask_labels, class_labels = self.convert_semantic_to_mask2former_labels(label_map[i],0)
                mask_labels_list.append(mask_labels.to(image.device))
                class_labels_list.append(class_labels.to(image.device))

            outputs = self.model(pixel_values=image,
                                 mask_labels=mask_labels_list,  # list, length=batch_size
                                 class_labels=class_labels_list  # list, length=batch_size
                                 )

        # ===
        mask = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h, w)] * b
        )
        mask = torch.stack(mask, dim=0)
        return mask, outputs.loss

    def load_model(self, model_id, local_dir):
        # 检查本地路径是否存在 config 和权重文件
        if os.path.exists(os.path.join(local_dir, "config.json")) and \
                os.path.exists(os.path.join(local_dir, "preprocessor_config.json")):
            print(f"✔ 正在从本地加载模型：{local_dir}")
            processor = Mask2FormerImageProcessor.from_pretrained(local_dir, use_fast=False)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
        else:
            print(f"⚠ 本地未找到模型，联网下载并保存至：{local_dir}")
            processor = Mask2FormerImageProcessor.from_pretrained(model_id, use_fast=False)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
            os.makedirs(local_dir, exist_ok=True)
            processor.save_pretrained(local_dir)
            model.save_pretrained(local_dir)

        return processor, model

    @classmethod
    def get_model_ids(self) -> list[str]:
        '''
            from huggingface_hub import list_models
            models = list_models(search="facebook/mask2former")
            for m in models:
                print(m.modelId)
        '''
        return [
            "facebook/mask2former-swin-tiny-ade-semantic",
            "facebook/mask2former-swin-tiny-cityscapes-semantic",
            "facebook/mask2former-swin-small-ade-semantic",
            "facebook/mask2former-swin-small-cityscapes-semantic",
            "facebook/mask2former-swin-base-ade-semantic",
            "facebook/mask2former-swin-base-IN21k-ade-semantic",
            "facebook/mask2former-swin-base-IN21k-cityscapes-semantic",
            "facebook/mask2former-swin-large-ade-semantic",
            "facebook/mask2former-swin-large-cityscapes-semantic",
            "facebook/mask2former-swin-large-mapillary-vistas-semantic"
        ]


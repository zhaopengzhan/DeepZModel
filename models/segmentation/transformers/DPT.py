import os

import torch
from torch import nn
from transformers import DPTForSemanticSegmentation, \
    DPTImageProcessor

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings

warnings.filterwarnings("ignore")

from models import DeepZMODELS

# @DeepZMODELS.register_module('DPT')
class DPT(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, model_id: str = "Intel/dpt-large-ade"):
        super(DPT, self).__init__()
        self.model_id = model_id
        self.in_chans = in_channels
        self.num_classes = num_classes

        root_dir = r'F:/cache_hf'
        local_dir = os.path.join(root_dir, model_id)
        _, self.model = self.load_model(model_id, local_dir)

        self._init()
        pass

    def _init(self):
        # 1. in
        old_conv = self.model.dpt.embeddings.patch_embeddings.projection
        self.model.dpt.embeddings.patch_embeddings.projection = nn.Conv2d(
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
        old_cls = self.model.head.head[4]
        self.model.head.head[4] = nn.Conv2d(
            in_channels=old_cls.in_channels,
            out_channels=self.num_classes,
            kernel_size=old_cls.kernel_size,
            stride=old_cls.stride,
            padding=old_cls.padding,
            dilation=old_cls.dilation,
            groups=old_cls.groups,
            bias=(old_cls.bias is not None),
            padding_mode=old_cls.padding_mode
        )

        # 3. auxiliary
        old_cls = self.model.auxiliary_head.head[4]
        self.model.auxiliary_head.head[4] = nn.Conv2d(
            in_channels=old_cls.in_channels,
            out_channels=self.num_classes,
            kernel_size=old_cls.kernel_size,
            stride=old_cls.stride,
            padding=old_cls.padding,
            dilation=old_cls.dilation,
            groups=old_cls.groups,
            bias=(old_cls.bias is not None),
            padding_mode=old_cls.padding_mode
        )

        pass

    def forward(self, img):
        outputs = self.model(pixel_values=img)
        return outputs.logits

    def load_model(self, model_id, local_dir):
        # 检查本地路径是否存在 config 和权重文件
        if os.path.exists(os.path.join(local_dir, "config.json")) and \
                os.path.exists(os.path.join(local_dir, "preprocessor_config.json")):
            print(f"✔ 正在从本地加载模型：{local_dir}")
            processor = DPTImageProcessor.from_pretrained(local_dir, use_fast=False)
            # TODO: 由于img_size固定，必须改config
            t_model = DPTForSemanticSegmentation.from_pretrained(model_id)
            config = t_model.config
            config.num_channels = self.in_chans
            model = DPTForSemanticSegmentation.from_pretrained(model_id, config=config,
                                                               ignore_mismatched_sizes=True)
        else:
            print(f"⚠ 本地未找到模型，联网下载并保存至：{local_dir}")
            processor = DPTImageProcessor.from_pretrained(model_id, use_fast=False)

            # TODO: 由于img_size固定，必须改config
            t_model = DPTForSemanticSegmentation.from_pretrained(model_id)
            config = t_model.config
            config.num_channels = self.in_chans
            model = DPTForSemanticSegmentation.from_pretrained(model_id, config=config,
                                                               ignore_mismatched_sizes=True)

            os.makedirs(local_dir, exist_ok=True)
            processor.save_pretrained(local_dir)
            model.save_pretrained(local_dir)

        return processor, model

    @classmethod
    def get_model_ids(self) -> list[str]:
        '''
            from huggingface_hub import list_models
            models = list_models(search="nvidia/segformer")
            for m in models:
                print(m.modelId)
        '''
        return [
            "Intel/dpt-large",
            "Intel/dpt-large-ade",
            "Intel/dpt-hybrid-midas",
            "Intel/dpt-beit-large-512",
            "Intel/dpt-beit-large-384",
            "Intel/dpt-beit-base-384",
            "Intel/dpt-swinv2-base-384",
            "Intel/dpt-swinv2-tiny-256",
            "Intel/dpt-swinv2-large-384",
        ]


def test_build_model():
    model = DPT(model_id=DPT.get_model_ids()[-2],
                in_chans=8,
                num_classes=54)
    # print(model.model.config)
    # print(model)
    outputs = model(torch.randn(2, 8, 384, 384))
    print(outputs.shape)
    pass


if __name__ == '__main__':
    test_build_model()

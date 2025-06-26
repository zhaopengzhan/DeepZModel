import os

from peft import LoraConfig
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from models import DeepZMODELS


@DeepZMODELS.register_module('LORASegFormer')
class LORASegFormer(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        super(LORASegFormer, self).__init__()
        self.model_id = model_id
        self.in_chans = in_channels
        self.num_classes = num_classes
        # config = SegformerConfig(num_channels=in_chans, num_labels=num_classes)
        root_dir = r'F:/cache_hf'
        local_dir = os.path.join(root_dir, model_id)
        _, model = self.load_segformer(model_id, local_dir)

        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        # 2. 构建 LoRA 配置（可以根据显存和任务调整参数）
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "key", "value"],  # 注意力模块名称需匹配模型
            lora_dropout=0.05,
            bias="none",
            # task_type="SEMANTIC_SEGMENTATION",
            task_type="FEATURE_EXTRACTION"
        )
        # print(model)
        # 3. 添加 adapter，并命名
        model.add_adapter(lora_config, adapter_name="gid_lora")
        model.set_adapter("gid_lora")  # 激活刚加的 adapter

        # print(model.active_adapters())
        # print(model.peft_config)
        # model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()
        self.model = model
        self._init()
        pass

    def _init(self):
        # 1. in
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
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
        self.model.segformer.encoder.patch_embeddings[0].proj.requires_grad = False

        # 2. out
        old_cls = self.model.decode_head.classifier
        self.model.decode_head.classifier = nn.Conv2d(
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
        logits = self.up(outputs.logits)
        return logits

    def load_segformer(self, model_id, local_dir):
        # 检查本地路径是否存在 config 和权重文件
        if os.path.exists(os.path.join(local_dir, "config.json")) and \
                os.path.exists(os.path.join(local_dir, "preprocessor_config.json")):
            print(f"✔ 正在从本地加载模型：{local_dir}")
            processor = SegformerImageProcessor.from_pretrained(local_dir)
            model = SegformerForSemanticSegmentation.from_pretrained(local_dir)
        else:
            print(f"⚠ 本地未找到模型，联网下载并保存至：{local_dir}")
            processor = SegformerImageProcessor.from_pretrained(model_id)
            model = SegformerForSemanticSegmentation.from_pretrained(model_id)
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
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
            "nvidia/segformer-b0-finetuned-cityscapes-640-1280",
            "nvidia/segformer-b0-finetuned-cityscapes-768-768",
            "nvidia/segformer-b1-finetuned-ade-512-512",
            "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b2-finetuned-ade-512-512",
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b3-finetuned-ade-512-512",
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b4-finetuned-ade-512-512",
            "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b5-finetuned-ade-640-640",
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        ]


@DeepZMODELS.register_module('SegFormer')
class SegFormer(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        super(SegFormer, self).__init__()
        self.model_id = model_id
        self.in_chans = in_channels
        self.num_classes = num_classes
        # config = SegformerConfig(num_channels=in_chans, num_labels=num_classes)
        root_dir = r'F:/cache_hf'
        local_dir = os.path.join(root_dir, model_id)
        _, self.model = self.load_segformer(model_id, local_dir)

        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

        self._init()
        pass

    def _init(self):
        # 1. in
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
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
        self.model.segformer.encoder.patch_embeddings[0].proj.requires_grad = False

        # 2. out
        old_cls = self.model.decode_head.classifier
        self.model.decode_head.classifier = nn.Conv2d(
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
        logits = self.up(outputs.logits)
        return logits

    def load_segformer(self, model_id, local_dir):
        # 检查本地路径是否存在 config 和权重文件
        if os.path.exists(os.path.join(local_dir, "config.json")) and \
                os.path.exists(os.path.join(local_dir, "preprocessor_config.json")):
            print(f"✔ 正在从本地加载模型：{local_dir}")
            processor = SegformerImageProcessor.from_pretrained(local_dir)
            model = SegformerForSemanticSegmentation.from_pretrained(local_dir)
        else:
            print(f"⚠ 本地未找到模型，联网下载并保存至：{local_dir}")
            processor = SegformerImageProcessor.from_pretrained(model_id)
            model = SegformerForSemanticSegmentation.from_pretrained(model_id)
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
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
            "nvidia/segformer-b0-finetuned-cityscapes-640-1280",
            "nvidia/segformer-b0-finetuned-cityscapes-768-768",
            "nvidia/segformer-b1-finetuned-ade-512-512",
            "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b2-finetuned-ade-512-512",
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b3-finetuned-ade-512-512",
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b4-finetuned-ade-512-512",
            "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b5-finetuned-ade-640-640",
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        ]


@DeepZMODELS.register_module('DofSegFormer')
class DofSegFormer(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        super(DofSegFormer, self).__init__()
        self.model_id = model_id
        self.in_chans = in_channels
        self.num_classes = num_classes

        root_dir = r'F:/cache_hf'
        local_dir = os.path.join(root_dir, model_id)
        _, model = self.load_segformer(model_id, local_dir)

        # lora_config = LoraConfig(
        #     r=8,
        #     lora_alpha=16,
        #     target_modules=["query", "key", "value"],  # 注意力模块名称需匹配模型
        #     lora_dropout=0.05,
        #     bias="none",
        #     # task_type="SEMANTIC_SEGMENTATION",
        #     task_type="FEATURE_EXTRACTION"
        # )
        # # print(model)
        # # 3. 添加 adapter，并命名
        # model.add_adapter(lora_config, adapter_name="gid_lora")
        # model.set_adapter("gid_lora")  # 激活刚加的 adapter

        self.model = model
        self._init()

        pass

    def _init(self):
        # 1. in
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=self.in_chans,
            out_channels=old_conv.out_channels,
            kernel_size=1,
            bias=(old_conv.bias is not None),
        )

        # 2. out
        old_cls = self.model.decode_head.classifier
        self.model.decode_head.classifier = nn.Conv2d(
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

    def load_segformer(self, model_id, local_dir):
        # 检查本地路径是否存在 config 和权重文件
        if os.path.exists(os.path.join(local_dir, "config.json")) and \
                os.path.exists(os.path.join(local_dir, "preprocessor_config.json")):
            print(f"✔ 正在从本地加载模型：{local_dir}")
            processor = SegformerImageProcessor.from_pretrained(local_dir)
            model = SegformerForSemanticSegmentation.from_pretrained(local_dir)
        else:
            print(f"⚠ 本地未找到模型，联网下载并保存至：{local_dir}")
            processor = SegformerImageProcessor.from_pretrained(model_id)
            model = SegformerForSemanticSegmentation.from_pretrained(model_id)
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
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
            "nvidia/segformer-b0-finetuned-cityscapes-640-1280",
            "nvidia/segformer-b0-finetuned-cityscapes-768-768",
            "nvidia/segformer-b1-finetuned-ade-512-512",
            "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b2-finetuned-ade-512-512",
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b3-finetuned-ade-512-512",
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b4-finetuned-ade-512-512",
            "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
            "nvidia/segformer-b5-finetuned-ade-640-640",
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        ]


def test_build_model():
    model = LORASegFormer(model_id=SegFormer.get_model_ids()[0],
                          in_chans=4,
                          num_classes=32)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    pass


if __name__ == '__main__':
    test_build_model()

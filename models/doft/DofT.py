import os

import timm
import torch
from torch import nn


class DofTV1(nn.Module):
    def __init__(self, img_size=56, in_chans=3, num_classes=17, name='vit_tiny_patch16_384.augreg_in21k_ft_in1k'):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.in_chans = in_chans

        if name == 'vit_huge_patch14_224.mae':
            img_size = 16  # 196
            embed_dim = 1280
        if name == 'vit_huge_patch16_gap_448.in1k_ijepa':
            img_size = 28  # 784
            embed_dim = 1280
        if name == 'vit_tiny_patch16_384.augreg_in21k_ft_in1k':
            img_size = 24
            embed_dim = 192

        self.img_size = img_size
        self.embed_dim = embed_dim

        weights_path = fr'F:/cache_hf/{name}cls.pth'
        if os.path.exists(weights_path):
            ViT = timm.create_model(name, pretrained=False)
            checkpoint = torch.load(weights_path, weights_only=True)
            ViT.load_state_dict(checkpoint['state_dict'])
        else:
            ViT = timm.create_model(name, pretrained=True)
            torch.save({
                'state_dict': ViT.state_dict(),
            }, weights_path)

        ViT.patch_embed.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0)
        ViT.patch_embed.img_size = (img_size, img_size)
        self.vit = ViT
        self.classifer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.vit.forward_features(x)
        x = x[:, 1:, :].view(-1, self.img_size, self.img_size, self.embed_dim).permute(0, 3, 1, 2)
        # x = x.view(-1, self.img_size, self.img_size, self.embed_dim).permute(0, 3, 1, 2)
        x = self.classifer(x)
        return x

    def freeze_parameter(self, freeze=True):
        if freeze:
            for name, param in self.vit.named_parameters():
                if 'patch_embed' not in name:
                    param.requires_grad = False


if __name__ == '__main__':
    model = DofTV1(in_chans=300, num_classes=17, img_size=16)
    print(model(torch.randn(1, 300, 16, 16)).shape)

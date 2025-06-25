import os

import timm
import torch
from torch import nn


class Dof_VGG(nn.Module):
    def __init__(self, img_size=56, in_chans=3, num_classes=17, name='vgg16.tv_in1k'):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.in_chans = in_chans

        # if name == 'vgg16.tv_in1k':
        embed_dim = 512
        width = 64

        self.img_size = img_size
        self.embed_dim = embed_dim

        weights_path = fr'F:/cache_hf/{name}cls.pth'
        if os.path.exists(weights_path):
            model = timm.create_model(name, pretrained=False)
            checkpoint = torch.load(weights_path, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"Online load {name}")
            model = timm.create_model(name, pretrained=True)
            torch.save({
                'state_dict': model.state_dict(),
            }, weights_path)

        model.features[0] = nn.Sequential(
            nn.Conv2d(in_chans, width, kernel_size=1, stride=1, padding=0)
        )
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                # print(f"Replacing MaxPool2d at features[{i}] with Identity")
                model.features[i] = nn.Identity()

        self.extractor = model
        self.classifer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.extractor.forward_features(x)
        x = self.classifer(x)
        return x

    def freeze_parameter(self, freeze=True):
        if freeze:
            for name, param in self.vit.named_parameters():
                if 'patch_embed' not in name:
                    param.requires_grad = False


if __name__ == '__main__':
    model = Dof_VGG(in_chans=300, num_classes=17, img_size=16)
    print(model(torch.randn(1, 300, 16, 16)).shape)

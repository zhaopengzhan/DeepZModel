import timm
import torch
from ptflops import get_model_complexity_info
from torch import nn


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, use_transpose=True):
        super().__init__()

        if use_transpose:
            upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )
        self.up = upsample
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        pass

    def forward(self, x, skip):
        x = self.up(x)
        if skip != None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
        pass


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, use_transpose=True):
        super().__init__()
        self.layers = self.make_layers(encoder_channels, decoder_channels, use_transpose)

    def make_layers(self, encoder_channels, decoder_channels, use_transpose):
        num_layers = len(decoder_channels)
        layers = []

        for i in range(num_layers):
            in_ch = encoder_channels[-(i + 1)]
            skip_ch = encoder_channels[-(i + 2)] if i < len(encoder_channels) - 1 else 0
            out_ch = decoder_channels[i]
            if i != 0:
                in_ch = decoder_channels[i - 1]

            layers.append(UpLayer(in_ch, out_ch, skip_ch, use_transpose))

        return nn.ModuleList(layers)

    def forward(self, features):

        x = features[-1]  # 解码器初始输入
        for i, layer in enumerate(self.layers):
            skip = features[-(i + 2)] if i < len(features) - 1 else None
            x = layer(x, skip)
        return x


class MobileUNet(nn.Module):
    def __init__(self, num_classes, in_chans):
        super().__init__()
        self.mobile = timm.create_model('mobilenetv3_large_100',
                                        pretrained=True,
                                        features_only=True,
                                        out_indices=(0, 1, 2, 3, 4),
                                        )

        self.mobile.conv_stem = torch.nn.Conv2d(in_channels=in_chans, out_channels=self.mobile.conv_stem.out_channels,
                                                kernel_size=3, stride=2, padding=1, bias=False)

        encoder_channels = [16, 24, 40, 112, 960]
        # decoder_channels = [256, 128, 64, 32, 16]
        decoder_channels = [512, 256, 256, 256, 256]

        self.decoder = UNetDecoder(encoder_channels, decoder_channels)

        self.head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)  # 1x1卷积降低到类别数

    def forward(self, image_ts):
        features = self.mobile(image_ts)

        last_hidden_state = self.decoder(features)

        outputs = self.head(last_hidden_state)

        return outputs


class DualMobileUNet(nn.Module):
    def __init__(self, num_classes, in_chans1=3, in_chans2=256):
        super().__init__()
        self.backbone1 = timm.create_model('mobilenetv3_large_100',
                                           pretrained=True,
                                           features_only=True,
                                           out_indices=(0, 1, 2, 3, 4),
                                           )

        self.backbone1.conv_stem = torch.nn.Conv2d(in_channels=in_chans1,
                                                   out_channels=self.backbone1.conv_stem.out_channels,
                                                   kernel_size=3, stride=2, padding=1, bias=False)

        self.backbone2 = timm.create_model('mobilenetv3_large_100',
                                           pretrained=True,
                                           features_only=True,
                                           out_indices=(0, 1, 2, 3, 4),
                                           )

        self.backbone2.conv_stem = torch.nn.Conv2d(in_channels=in_chans2,
                                                   out_channels=self.backbone2.conv_stem.out_channels,
                                                   kernel_size=3, stride=2, padding=1, bias=False)

        # encoder_channels = [16, 24, 40, 112, 960]
        encoder_channels = [16, 24, 40, 112, 1920]

        # decoder_channels = [256, 128, 64, 32, 16]
        decoder_channels = [512, 256, 256, 256, 256]

        self.decoder = UNetDecoder(encoder_channels, decoder_channels)

        self.head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)  # 1x1卷积降低到类别数

    def forward(self, image_HR, image_LR):
        features_hr = self.backbone1(image_HR)
        features_lr = self.backbone2(image_LR)

        features_hr[-1] = torch.cat([features_hr[-1], features_lr[-1]], dim=1)

        last_hidden_state = self.decoder(features_hr)

        outputs = self.head(last_hidden_state)

        return outputs


if __name__ == '__main__':
    model = MobileUNet(num_classes=256, in_chans=256)
    outputs = model(torch.rand(size=(1, 256, 224, 224)))
    # print(model)
    # print(outputs.shape)

    macs, params = get_model_complexity_info(model, (256, 224, 224), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # model = DualMobileUNet(num_classes=256, in_chans1=3, in_chans2=256)
    # outputs = model(torch.rand(size=(1, 3, 224, 224)), torch.rand(size=(1, 256, 224, 224)))
    # print(model)
    # print(outputs.shape)
    # from torchinfo import summary
    #
    # # 支持多输入
    # summary(
    #     model,
    #     input_size=[(2, 3, 224, 224), (2, 256, 224, 224)],  # 输入形状
    #     # col_names=["input_size", "output_size", "num_params"],  # 显示的列
    #     # col_width=16,
    #     # depth=3  # 展示模型深度
    # )
    #
    #
    # # 构造自定义输入函数
    # def input_constructor(input_res):
    #     input1 = torch.randn((2,) + input_res[0])  # 第一个输入张量
    #     input2 = torch.randn((2,) + input_res[1])  # 第二个输入张量
    #     return dict(image_HR=input1, image_LR=input2)
    #
    #
    # macs, params = get_model_complexity_info(model.to('cpu'), input_res=((3, 224, 224), (256, 224, 224)),
    #                                          input_constructor=input_constructor, as_strings=True,
    #                                          print_per_layer_stat=False, verbose=False)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    pass

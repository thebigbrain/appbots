import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=None):
        super(UNet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d

        # 编码器
        self.encoders = nn.ModuleList()
        for feat in features:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feat, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feat),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feat),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = feat

        # 解码器
        self.decoders = nn.ModuleList()
        decoders_features = features[::-1]
        for i in range(len(decoders_features)):
            if i == 0:
                in_channels = decoders_features[i] + features[-1]
            else:
                in_channels = decoders_features[i] + decoders_features[i - 1]
            out_channels = decoders_features[i]
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)

        skip_connections = skip_connections[::-1]

        for decoder, skip_connection in zip(self.decoders, skip_connections):
            x = self.upsample(x)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = decoder(concat_skip)

        return self.final_conv(x)

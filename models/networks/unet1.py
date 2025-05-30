import torch
import torch.nn as nn


class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, multitask=False):
        super(U_Net, self).__init__()

        self.multitask = multitask
        features = 64

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_channels=in_channels, out_channels=64)
        self.Conv2 = conv_block(in_channels=64, out_channels=128)
        self.Conv3 = conv_block(in_channels=128, out_channels=256)
        self.Conv4 = conv_block(in_channels=256, out_channels=512)
        self.Conv5 = conv_block(in_channels=512, out_channels=1024)
        self.Up5 = up_conv(in_channels=1024, out_channels=512)
        self.Up_conv5 = conv_block(in_channels=1024, out_channels=512)
        self.Up4 = up_conv(in_channels=512, out_channels=256)
        self.Up_conv4 = conv_block(in_channels=512, out_channels=256)
        self.Up3 = up_conv(in_channels=256, out_channels=128)
        self.Up_conv3 = conv_block(in_channels=256, out_channels=128)
        self.Up2 = up_conv(in_channels=128, out_channels=64)
        self.Up_conv2 = conv_block(in_channels=128, out_channels=64)
        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

        if self.multitask:
            self.hm_conv = nn.Sequential(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
            )

            self.size_conv = nn.Sequential(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1)
            )

    def forward(self, x):

        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decode
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # d1 = self.Conv_1x1(d2)
        # output = {'mask': d1}
        if self.multitask:
            output = {'mask': self.Conv_1x1(d2), 'heatmap': self.hm_conv(d2), 'blobsize': self.size_conv(d2)}
        else:
            output = {'mask': self.Conv_1x1(d2)}

        return output

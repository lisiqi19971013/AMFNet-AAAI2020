import torch
import torch.nn as nn
import torchvision.models as models


class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, conv_list):
        super(ASPP, self).__init__()
        self.conv_list = conv_list
        self.conv = nn.ModuleList(
            [nn.Conv2d(inplanes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in conv_list])
        self.bn = nn.ModuleList([nn.BatchNorm2d(planes) for dil in conv_list])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.bn[0](self.conv[0](x))
        for i in range(1, len(self.conv_list)):
            y += self.bn[i](self.conv[i](x))
        x = self.relu(y)

        return x


class DepthSeg(nn.Module):
    def __init__(self, model, num_classes):
        super(DepthSeg, self).__init__()

        self.num_classes = num_classes

        self.conv1 = model.conv1
        self.bn0 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.duc1 = DUC(2048, 2048 * 2)
        self.duc2 = DUC(1024, 1024 * 2)
        self.duc3 = DUC(512, 512 * 2)
        self.duc4 = DUC(128, 128 * 2)
        self.duc5 = DUC(64, 64 * 2)
        self.ASPP = ASPP(32, 64, [1, 3, 5, 7])
        self.transformer = nn.Conv2d(320, 128, kernel_size=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        dfm1 = fm3 + self.duc1(fm4)
        dfm2 = fm2 + self.duc2(dfm1)
        dfm3 = fm1 + self.duc3(dfm2)
        dfm3_t = self.transformer(torch.cat((dfm3, pool_x), 1))
        dfm4 = conv_x + self.duc4(dfm3_t)
        dfm5 = self.duc5(dfm4)
        out = self.ASPP(dfm5)
        return out,


class ColorSeg(nn.Module):
    def __init__(self, model, num_classes):
        super(ColorSeg, self).__init__()
        self.num_classes = num_classes
        self.conv1 = model.conv1
        self.bn0 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.duc1 = DUC(2048, 2048 * 2)
        self.duc2 = DUC(1024, 1024 * 2)
        self.duc3 = DUC(512, 512 * 2)
        self.duc4 = DUC(128, 128 * 2)
        self.duc5 = DUC(64, 64 * 2)

        self.ASPP = ASPP(32, 64, [1, 3, 5, 7])

        self.transformer = nn.Conv2d(320, 128, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        dfm1 = fm3 + self.duc1(fm4)
        dfm2 = fm2 + self.duc2(dfm1)
        dfm3 = fm1 + self.duc3(dfm2)
        dfm3_t = self.transformer(torch.cat((dfm3, pool_x), 1))
        dfm4 = conv_x + self.duc4(dfm3_t)
        dfm5 = self.duc5(dfm4)
        out = self.ASPP(dfm5)

        return out,


class Seg2DNet(nn.Module):
    def __init__(self, num_classes):
        super(Seg2DNet, self).__init__()

        self.num_classes = num_classes
        self.cs = ColorSeg(model=models.resnet101(False), num_classes=num_classes)
        self.ds = DepthSeg(model=models.resnet101(False), num_classes=num_classes)

        self.fuse = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 64, 1),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, num_classes, 1))

    def forward(self, color, depth):
        color = self.cs(color)
        depth = self.ds(depth)
        x = torch.cat((color[0], depth[0]), dim=1)
        out = self.fuse(x)
        return out
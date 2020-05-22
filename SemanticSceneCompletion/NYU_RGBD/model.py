import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import operator
from torch.autograd import Variable


class Attention_block(nn.Module):
    def __init__(self, spatialsize=[1, 64, 60, 36, 60]):
        super(Attention_block, self).__init__()
        self.fc1 = nn.Linear(spatialsize[1], int(spatialsize[1]/8))
        self.fc2 = nn.Linear(int(spatialsize[1]/8), int(spatialsize[1]/8))
        self.fc3 = nn.Linear(int(spatialsize[1]/8), spatialsize[1])
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=(spatialsize[2], spatialsize[3], spatialsize[4]), stride=1)
        self.avgpool = nn.AvgPool3d(kernel_size=(spatialsize[2], spatialsize[3], spatialsize[4]), stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv3d(2, 2, 5, bias=False, padding=2)
        self.bn = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 1, 1, bias=False)

    def forward(self, infeature):
        spatialsize = infeature.shape
        max_f = self.maxpool(infeature).reshape(spatialsize[0], 1, spatialsize[1])
        avg_f = self.avgpool(infeature).reshape(spatialsize[0], 1, spatialsize[1])
        cha_f = torch.cat([max_f, avg_f], dim=1)
        out1 = self.fc3(self.relu(self.fc2(self.relu(self.fc1(cha_f)))))
        channel_attention = self.sigmoid(out1[:, 0, :] + out1[:, 1, :])\
            .reshape(spatialsize[0], spatialsize[1], 1, 1, 1)
        feature_with_channel_attention = infeature * channel_attention

        channel_wise_avg_pooling = torch.mean(feature_with_channel_attention, dim=1)
        channel_wise_max_pooling = torch.max(feature_with_channel_attention, dim=1)[0]
        channel_wise_pooling = torch.stack((channel_wise_avg_pooling, channel_wise_max_pooling), dim=1)
        spatial_attention = self.sigmoid(self.conv2(self.relu(self.bn(self.conv1(channel_wise_pooling)))))
        feature_map_with_attention = feature_with_channel_attention * spatial_attention
        return feature_map_with_attention


class DUC(nn.Module):    # conv+bn+relu
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # Channal/4, H*2, W*2

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
        self.conv = nn.ModuleList([nn.Conv2d(inplanes, planes, kernel_size=3, padding=dil, dilation=dil, bias = False) for dil in conv_list])
        self.bn = nn.ModuleList([nn.BatchNorm2d(planes) for dil in conv_list])
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.bn[0](self.conv[0](x))
        for i in range(1, len(self.conv_list)):
            y = y + self.bn[i](self.conv[i](x))
        x = self.relu(y)
        return x


class DDR(nn.Module):
    def __init__(self, channel, dil=1, relu=True):
        super(DDR, self).__init__()
        n = int(channel/4)
        self.conv_in = nn.Conv3d(channel, n, 1, bias=False)
        self.conv1 = nn.Conv3d(n, n, (1, 1, 3), padding=(0, 0, dil), dilation=(1, 1, dil), bias=False)
        self.conv2 = nn.Conv3d(n, n, (1, 3, 1), padding=(0, dil, 0), dilation=(1, dil, 1), bias=False)
        self.conv3 = nn.Conv3d(n, n, (3, 1, 1), padding=(dil, 0, 0), dilation=(dil, 1, 1), bias=False)
        self.conv_out = nn.Conv3d(n, channel, 1, bias=False)
        self.Relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm3d(n)
        self.bn2 = nn.BatchNorm3d(n)
        self.bn3 = nn.BatchNorm3d(n)
        self.bn4 = nn.BatchNorm3d(n)
        self.bn5 = nn.BatchNorm3d(channel)
        self.flag = relu

    def forward(self, x):
        x1 = self.Relu(self.bn1(self.conv_in(x)))
        x2 = self.Relu(self.bn2(self.conv1(x1)))
        x3 = self.Relu(self.bn3(self.conv2(x2)))
        x4 = self.Relu(self.bn4(self.conv3(x2+x3)))
        x5 = self.bn5(self.conv_out(x2+x3+x4)) + x
        if self.flag:
            x5 = self.Relu(x5)
        return x5


class ASPP3D(nn.Module):
    def __init__(self, channel, conv_list):
        super(ASPP3D, self).__init__()
        self.conv_list = conv_list
        self.conv = nn.ModuleList([DDR(channel, dil, relu=False) for dil in conv_list])
        self.bn = nn.BatchNorm3d(channel*(len(conv_list)+1))
        self.bn1 = nn.BatchNorm3d(channel)
        self.conv1 = nn.Conv3d(1, channel, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = []
        for i in range(0, len(self.conv_list)):
            y.append(self.conv[i](x))
        avg = self.bn1(self.conv1(torch.mean(x, dim=1, keepdim=True)))
        y.append(avg)
        x = self.relu(self.bn(torch.cat(y, dim=1)))
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

        self.duc1 = DUC(2048, 2048*2)
        self.duc2 = DUC(1024, 1024*2)
        self.duc3 = DUC(512, 512*2)
        self.duc4 = DUC(128, 128*2)
        self.duc5 = DUC(64, 64*2)
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

        dfm1 = fm3 + self.duc1(fm4)   # in 2048  out 1024
        dfm2 = fm2 + self.duc2(dfm1)  # in 1024  out 512
        dfm3 = fm1 + self.duc3(dfm2)  # in 2048  out 1024
        dfm3_t = self.transformer(torch.cat((dfm3, pool_x), 1))
        dfm4 = conv_x + self.duc4(dfm3_t)
        dfm5 = self.duc5(dfm4)
        out = self.ASPP(dfm5)   # 输出是有bn+relu的
        return out


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

        self.duc1 = DUC(2048, 2048*2)
        self.duc2 = DUC(1024, 1024*2)
        self.duc3 = DUC(512, 512*2)
        self.duc4 = DUC(128, 128*2)
        self.duc5 = DUC(64, 64*2)
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
        return out


class FuseSegNet(nn.Module):
    def __init__(self, num_classes):
        super(FuseSegNet, self).__init__()
        self.num_classes = num_classes

        self.cs = ColorSeg(model=models.resnet101(False), num_classes=num_classes)
        self.ds = DepthSeg(model=models.resnet101(False), num_classes=num_classes)

        self.fuse = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 64, 1))
        self.bn = nn.BatchNorm2d(64)
        self.classifiar = nn.Conv2d(64, 12, 1)

    def forward(self, color, depth):
        color = self.cs(color)
        depth = self.ds(depth)
        x = torch.cat((color, depth), dim=1)   # (bs,128,w,h)
        out = nn.ReLU()(self.bn(self.fuse(x)))
        logit = self.classifiar(out)
        pred = torch.argmax(logit, dim=1).int()
        return out, pred


class RAB(nn.Module):
    def __init__(self, channal=64, spatialsize=[1,64,60,36,60]):
        super(RAB, self).__init__()
        self.DDR2 = DDR(channal, relu=False)
        self.AttentionBlock = Attention_block(spatialsize)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.DDR2(x)
        out = self.AttentionBlock(out)
        out = out + residual
        out = self.Relu(out)
        return out


class AMFNet(nn.Module):
    def __init__(self, seg2d_path, nClassesTotal, batchsize, wo_att=False, wo_seg=False):
        super(AMFNet, self).__init__()
        self.seg2d = FuseSegNet(num_classes=nClassesTotal)
        chpo = torch.load(seg2d_path)
        self.seg2d.load_state_dict(chpo['state_dict'], strict=False)
        print("=> seg2d loaded checkpoint '{}'".format(seg2d_path))

        n = 64
        self.wo_seg = wo_seg

        if not wo_att:
            self.seq1 = RAB(n, [batchsize, n, 60, 36, 60])
            self.seq2 = RAB(n, [batchsize, n, 60, 36, 60])
            self.seq3 = RAB(n, [batchsize, n, 60, 36, 60])
            self.seq4 = RAB(n, [batchsize, n, 60, 36, 60])
        else:
            self.seq1 = DDR(n)
            self.seq2 = DDR(n)
            self.seq3 = DDR(n)
            self.seq4 = DDR(n)

        self.conv = nn.Sequential(nn.Conv3d(n*4, n, 1, bias=False), nn.BatchNorm3d(n), nn.ReLU(inplace=True))
        self.ASPP3D = ASPP3D(n, [1, 3, 5, 7])
        self.ASPP3Dout = nn.Sequential(nn.Conv3d(64*5, 64, 1, bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                       nn.Conv3d(64, 64, 1, bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                       nn.Conv3d(64, 12, 3, padding=1), nn.Conv3d(12, 12, 1))

        self.seq2d = nn.Sequential(nn.Conv3d(12, 12, 3, padding=1, bias=False), nn.BatchNorm3d(12), nn.ReLU(inplace=True),
                                   nn.Conv3d(12, 12, 3, padding=1, bias=False), nn.BatchNorm3d(12), nn.ReLU(inplace=True),
                                   nn.Conv3d(12, 12, 1), nn.Sigmoid())

        self.img_required_size = (640, 480)
        self.img_size = (384, 288)
        self.coord64 = np.load('../coord.npy')

        if operator.eq(self.img_required_size, self.img_size) == 0:
            x = np.array(range(self.img_required_size[0]), dtype=np.float32)
            y = np.array(range(self.img_required_size[1]), dtype=np.float32)
            scale = 1.0 * self.img_size[0] / self.img_required_size[0]
            x = x * scale + 0.5
            y = y * scale + 0.5
            x = x.astype(np.int64)
            y = y.astype(np.int64)
            if x[self.img_required_size[0]-1] >= self.img_size[0]:
                x[self.img_required_size[0]-1] = self.img_size[0] - 1
            if y[self.img_required_size[1]-1] >= self.img_size[1]:
                y[self.img_required_size[1]-1] = self.img_size[1] - 1
            xx = np.ones((self.img_required_size[1], self.img_required_size[0]), dtype = np.int64)
            yy = np.ones((self.img_required_size[1], self.img_required_size[0]), dtype = np.int64)
            xx[:] = x
            yy[:] = y.reshape((self.img_required_size[1], 1)) * self.img_size[0]
            image_mapping1 = (xx + yy).reshape(-1)
        else:
            image_mapping1 = np.array(range(self.img_required_size[0]*self.img_required_size[1]), dtype=np.int64)
        self.register_buffer('image_mapping', torch.autograd.Variable(torch.LongTensor(image_mapping1), requires_grad=False))

    def forward(self, img, depth, depth_mapping_3d):
        bs, ch, hi, wi = img.size()
        segres, seg2d = self.seg2d(img, depth)

        segres = segres.contiguous().view(bs * 64, hi * wi)
        segres = torch.index_select(segres, 1, self.image_mapping).contiguous().view(
            bs, 64, self.img_required_size[0] * self.img_required_size[1]).permute(0, 2, 1)
        zerosVec = Variable(torch.zeros(bs, 1, 64), requires_grad=False).cuda(img.get_device())
        segVec = torch.cat((segres, zerosVec), 1)
        segres = [torch.index_select(segVec[i], 0, depth_mapping_3d[i]) for i in range(bs)]
        segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(bs, 64, 60, 36, 60)

        pred2d = seg2d.contiguous().view(bs, hi * wi)
        pred2d = torch.index_select(pred2d, 1, self.image_mapping).view(
            bs, self.img_required_size[0] * self.img_required_size[1])
        zerosVec = Variable(torch.zeros(bs, 1).int(), requires_grad=False).cuda(img.get_device())
        pred2d = torch.cat((pred2d, zerosVec), 1)
        pred2d = [torch.index_select(pred2d[i], 0, depth_mapping_3d[i]) for i in range(bs)]
        pred2d = torch.stack(pred2d).view(bs, 60, 36, 60)

        max_coord = np.zeros([bs, 12, 3]).astype(int)
        min_coord = np.zeros([bs, 12, 3]).astype(int)
        for b in range(bs):
            for n in range(1, 12):
                l = np.array((pred2d[b]).cpu()) == n
                coord = self.coord64[l]
                if len(coord) != 0:
                    for i in range(3):
                        max_coord[b, n, i] = coord[:, i].max()
                        min_coord[b, n, i] = coord[:, i].min()

        spatial_attention = np.zeros([bs, 12, 60, 36, 60]).astype(np.float)
        spatial_attention[:, 0, :, :, :] = 1
        for b in range(bs):
            for n in range(1, 12):
                if (max_coord[b, n, :] == [0, 0, 0]).all() and (min_coord[b, n, :] == [0, 0, 0]).all():
                    continue
                else:
                    spatial_attention[b, 0, min_coord[b, n, 0]:max_coord[b, n, 0] + 1, min_coord[b, n, 1]:max_coord[b, n, 1] + 1, min_coord[b, n, 2]:max_coord[b, n, 2] + 1] = 0
                    spatial_attention[b, n, min_coord[b, n, 0]:max_coord[b, n, 0] + 1, min_coord[b, n, 1]:max_coord[b, n, 1] + 1, min_coord[b, n, 2]:max_coord[b, n, 2] + 1] = 1

        spatial_attention = torch.from_numpy(spatial_attention).float().cuda(async=True)

        x1 = self.seq1(segres)
        x2 = self.seq2(x1)
        x3 = self.seq3(x2)
        x4 = self.seq4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv(x)
        x = self.ASPP3D(x)
        x = self.ASPP3Dout(x)
        if not self.wo_seg:
            spatial_attention = self.seq2d(spatial_attention)
            x = x.mul(spatial_attention)
        return x

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.seg2d.parameters(), 'lr': lr * lrp},
            {'params': self.seq1.parameters()},
            {'params': self.seq2.parameters()},
            {'params': self.seq3.parameters()},
            {'params': self.seq4.parameters()},
            {'params': self.conv.parameters()},
            {'params': self.ASPP3D.parameters()},
            {'params': self.ASPP3Dout.parameters()},
            {'params': self.seq2d.parameters()}
        ]
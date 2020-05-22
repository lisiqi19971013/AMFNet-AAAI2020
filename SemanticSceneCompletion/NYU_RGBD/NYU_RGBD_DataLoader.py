import torch
import numpy as np
import torch.utils.data as torch_data
from PIL import Image
import os
from SemanticSceneCompletion.NYU_RGBD.NYU_Path import *
from torchvision.transforms import Compose, Normalize, ToTensor
import random


def PCA_Jittering(img):
    img = np.asanyarray(img, dtype = 'float32')
    img = img / 255.0
    img_size = int(img.size / 3)
    img1 = img.reshape(img_size, 3)
    img1 = np.transpose(img1)
    img_cov = np.cov([img1[0], img1[1], img1[2]])
    lamda, p = np.linalg.eig(img_cov)

    p = np.transpose(p)
    alpha1 = random.normalvariate(0,1)
    alpha2 = random.normalvariate(0,1)
    alpha3 = random.normalvariate(0,1)
    v = np.transpose((alpha1*lamda[0], alpha2*lamda[1], alpha3*lamda[2]))
    add_num = np.dot(p,v)

    img2 = np.array([img[:,:,0]+add_num[0], img[:,:,1]+add_num[1], img[:,:,2]+add_num[2]])
    img2 = img2.reshape(3, img_size)
    img2 = np.transpose(img2)
    img2 = img2.reshape(img.shape)
    img2 = img2 * 255.0
    img2[img2<0] = 0
    img2[img2>255] = 255
    img2 = img2.astype(np.uint8)

    return Image.fromarray(img2)


class TrainDataLoaderFused(torch_data.Dataset):
    def __init__(self, path, npz_path, train_or_test, label_transform=None, num_classes=12):

        super(TrainDataLoaderFused, self).__init__()

        fid = open(path, "r")
        self.colorlist = []
        for line in fid.readlines():
            line = line.rstrip("\n")
            if os.path.exists(line):
                self.colorlist.append(line)
        fid.close()

        self.npz_path = npz_path

        self.num_classes = num_classes
        self.color_transform = Compose([ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225])])
        self.depth_transform = Compose([ToTensor(), Normalize([.5282, .3914, .4266], [.1945, .2480, .1506])])
        self.label_transform = label_transform

        self.resize_size = (384, 288)

        if train_or_test == 'train':
            self.filelist = np.arange(795)
        else:
            self.filelist = np.arange(654)

        self.train_or_test = train_or_test

    def __len__(self):
        return len(self.colorlist)

    def __getitem__(self, index):
        color = Image.open(self.colorlist[index]).convert('RGB')
        color = color.resize(self.resize_size, Image.ANTIALIAS)
        if self.train_or_test == "train":
            color = PCA_Jittering(color)
        color = self.color_transform(color)

        if self.train_or_test == 'train':
            depth = Image.open('%s/%06d.png' % (NYU_HHA_PATH_TRAIN, self.filelist[index]+1)).convert('RGB')
        else:
            depth = Image.open('%s/%06d.png' % (NYU_HHA_PATH_TEST, self.filelist[index]+1)).convert('RGB')
        depth = depth.resize(self.resize_size, Image.ANTIALIAS)
        depth = self.depth_transform(depth)

        loaddata = np.load(os.path.join(self.npz_path, '%06d.npz' % self.filelist[index]))
        label = torch.LongTensor(loaddata['arr_1'].astype(np.int64))
        label_weight = torch.FloatTensor(loaddata['arr_2'].astype(np.float32))
        mapping = loaddata['arr_3'].astype(np.int64)
        mapping1 = np.ones(8294400, dtype=np.int64)
        mapping1[:] = -1
        ind, = np.where(mapping >= 0)
        mapping1[mapping[ind]] = ind
        mapping2 = torch.autograd.Variable(torch.FloatTensor(mapping1.reshape((1, 1, 240, 144, 240)).astype(np.float32)))
        mapping2 = torch.nn.MaxPool3d(4, 4)(mapping2).data.view(-1).numpy()
        mapping2[mapping2 < 0] = 307200
        depth_mapping_3d = torch.LongTensor(mapping2.astype(np.int64))

        return color, depth, label, label_weight, depth_mapping_3d

import torch.utils.data as torch_data
from torchvision.transforms import Compose, Normalize, ToTensor
import os
from PIL import Image
import cv2
import torch
import numpy as np
import random
import imageio


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


class TrainDataLoader(torch_data.Dataset):
    def __init__(self, path, train, num_classes=12):

        super(TrainDataLoader, self).__init__()

        if train == True:
            HHA_path = '/repository/lisiqi/SATNet/nyu_selected_HHA'
        else:
            HHA_path = '/repository/lisiqi/SATNet/nyu_selected_val_HHA'

        fid = open(path, "r")
        self.colorlist = []
        self.categlist = []
        self.depthlist = []
        ind = 1
        for line in fid.readlines():
            line = line.rstrip("\n")
            line1 = line[0:len(line)-9] + 'category_suncg.png'
            if os.path.exists(line) and os.path.exists(line1):
                self.depthlist.append(HHA_path+'/%06d.png' % ind)
                self.colorlist.append(line)
                self.categlist.append(line1)
            ind = ind + 1
        fid.close()

        self.num_classes = num_classes
        self.color_transform = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = Compose([ToTensor(), Normalize([0.5282, 0.3914, 0.4266], [0.1945, 0.2480, 0.1506])])
        self.resize_size = (384, 288)

    def __len__(self):
        return len(self.colorlist)

    def __getitem__(self, index):
        depth = Image.open(self.depthlist[index]).convert('RGB')
        depth = depth.resize(self.resize_size, Image.ANTIALIAS)
        depth1 = depth.transpose(Image.FLIP_LEFT_RIGHT)
        depth = self.depth_transform(depth)
        depth1 = self.depth_transform(depth1)
        depth = torch.cat((depth, depth1), 0)

        color = Image.open(self.colorlist[index]).convert('RGB')
        color = color.resize(self.resize_size, Image.ANTIALIAS)
        color1 = color.transpose(Image.FLIP_LEFT_RIGHT)
        color = PCA_Jittering(color)
        color1 = PCA_Jittering(color1)
        color = self.color_transform(color)
        color1 = self.color_transform(color1)
        color = torch.cat((color, color1), 0)

        categ1 = cv2.imread(self.categlist[index], -1).astype(int)
        categ1 = cv2.resize(categ1, dsize=self.resize_size, interpolation=cv2.INTER_NEAREST)
        categ11 = np.fliplr(categ1)
        categ1 = torch.from_numpy(categ1.astype(np.int64))
        categ11 = torch.from_numpy(categ11.astype(np.int64))
        label1 = torch.cat((categ1, categ11), 0)

        return color, depth, label1


if __name__ == '__main__':
    train_dataset = TrainDataLoader('./nyu_train.txt', train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, 1, shuffle=True)
    color, depth, label = train_dataset.__getitem__(144)
    ch, hi, wi = color.size()
    color = color.view(2, -1, hi, wi)
    depth = depth.view(2, -1, hi, wi)
    label = label.view(2, -1, wi)
    imageio.imwrite('./color.jpg', np.array(color[1,0:3,:,:].permute(1,2,0)))
    imageio.imwrite('./dep.jpg', np.array(depth[1,0:3,:,:].permute(1,2,0)))
    imageio.imwrite('./lab.jpg', np.array(label[1,0:288,:]))
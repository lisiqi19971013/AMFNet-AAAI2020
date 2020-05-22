import sys, resource, os
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from SemanticSceneCompletion.NYU_RGBD.NYU_RGBD_DataLoader import TrainDataLoaderFused
import h5py
from SemanticSceneCompletion.NYU_RGBD.model import AMFNet
from SemanticSceneCompletion.NYU_RGBD.NYU_Path import *
import SemanticSceneCompletion.NYU_RGBD.IOU as IOU


def main():
    '''确认'''
    # resume_path = "/repository/lisiqi/save_model/wo_seg/NYU/AMFNet.pth"
    # output_path = '/repository/lisiqi/save_model/results/NYU/result_nyu_wo_seg.hdf5'

    # resume_path = "/repository/lisiqi/save_model/wo_att/NYU/AMFNet.pth"
    # output_path = '/repository/lisiqi/save_model/results/NYU/result_nyu_wo_att.hdf5'
    #
    # resume_path = "/repository/lisiqi/save_model/ours/NYU/AMFNet.pth"
    # output_path = '/repository/lisiqi/save_model/results/NYU/result_nyu.hdf5'

    resume_path = "/repository/lisiqi/save_model/ours/NYU/AMFNet.pth"
    output_path = '/repository/lisiqi/save_model/results/NYU/test.hdf5'

    bs = 1
    val_dataset = TrainDataLoaderFused(NYU_SAMPLE_TXT_TEST, NYU_NPZ_PATH_TEST, 'test')
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=1)
    data_loader.pin_memory = True
    seg2d_path = "/repository/lisiqi/save_model/seg2d/seg_fuse_nyu/checkpoint.pth.tar"

    model = AMFNet(seg2d_path, 12, bs, wo_att=False, wo_seg=False)
    if not os.path.isfile(resume_path):
        print("=> no checkpoint found at '{}'".format(resume_path))
        exit()
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dic'], strict=True)
    model = model.cuda()
    model.eval()

    cri_weights = torch.FloatTensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    criterion = torch.nn.CrossEntropyLoss(weight=cri_weights / torch.sum(cri_weights))
    criterion = criterion.cuda()
    loss_history = []

    softmax_layer = torch.nn.Softmax(dim=1).cuda(0)
    accuracy_total = np.zeros((3, 11), dtype=np.float32)
    predictions = []

    with torch.no_grad():
        for i, (color, depth, label, label_weight, depth_mapping_3d) in enumerate(data_loader):
            color = color.cuda(async=True)
            depth = depth.cuda(async=True)
            depth_mapping_3d = depth_mapping_3d.cuda(async=True)
            label = label.cuda(async=True)

            output = model(color, depth, depth_mapping_3d)
            accuracy_total = accuracy_total + IOU.computeIOU(output, label, 12)
            output1 = softmax_layer(output)
            out = output.permute(0, 2, 3, 4, 1).contiguous().view(-1, 12)
            selectindex = torch.nonzero(label_weight.view(-1)).view(-1).cuda()
            filterLabel = torch.index_select(label.view(-1), 0, selectindex)
            filterOutput = torch.index_select(out, 0, selectindex)
            loss = criterion(filterOutput, filterLabel)
            loss_history.append(loss.item())
            print('{0}/{1}'.format(i, len(data_loader)), loss.item())
            predictions.append(output1.cpu().data.numpy())

        predictions = np.vstack(predictions)
    accuracy = accuracy_total[0] / (np.sum(accuracy_total, 0) + 0.00001)
    accuracy_mean = np.mean(accuracy)
    print(accuracy_mean)
    print(accuracy)

    fp = h5py.File(output_path, 'w')
    result = fp.create_dataset('result', predictions.shape, dtype='f')
    result[...] = predictions
    fp.close()


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_STACK, (-1, -1))
    sys.setrecursionlimit(100000)
    main()
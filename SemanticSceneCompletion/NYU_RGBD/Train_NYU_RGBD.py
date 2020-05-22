from SemanticSceneCompletion.NYU_RGBD.NYU_RGBD_DataLoader import TrainDataLoaderFused
import SemanticSceneCompletion.NYU_RGBD.IOU as IOU
from SemanticSceneCompletion.NYU_RGBD.model import AMFNet
from SemanticSceneCompletion.NYU_RGBD.NYU_Path import *
import torch.nn as nn
import numpy as np
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn
import glob
import os
import datetime
import torch


def checkpoint_restore(model, path, use_cuda=True, iter=0):
    if use_cuda:
        model.cpu()
    iteration = 0
    epoch = 0
    best_acc = 0
    total_iter = 0
    loss_history = np.array([])
    val_history = np.array([])
    if iter > 0:
        f = path+'-%09d-' % iter+'.pth'
        assert os.path.isfile(f)
        print('Restore from ' + f)
        model_CKPT = torch.load(f)
        model.load_state_dict(model_CKPT['state_dic'])
        save_dict = model_CKPT['save_dict']
        iteration = save_dict['iter']
        epoch = save_dict['epoch']
        best_acc = save_dict['best_acc']
        total_iter = save_dict['total_iter']
        loss_history = save_dict['loss_history']
        val_history = save_dict['val_history']
    else:
        f = sorted(glob.glob(path+'-*-'+'.pth'))
        if len(f) > 0:
            f = f[-1]
            print('Restore from ' + f)
            model_CKPT = torch.load(f)
            model.load_state_dict(model_CKPT['state_dic'])
            save_dict = model_CKPT['save_dict']
            iteration = save_dict['iter']
            epoch = save_dict['epoch']
            best_acc = save_dict['best_acc']
            total_iter = save_dict['total_iter']
            loss_history = save_dict['loss_history']
            val_history = save_dict['val_history']
    if use_cuda:
        model.cuda()
    return iteration, total_iter, epoch, best_acc, loss_history, val_history


def checkpoint_save(model, path, total_iter, save_dict, use_cuda=True):
    f = path+'-%09d-' % total_iter+'.pth'
    model.cpu()
    torch.save({'state_dic': model.state_dict(), 'save_dict': save_dict}, f)
    if use_cuda:
        model.cuda()


def main():
    use_gpu = torch.cuda.is_available()
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 1
    max_epoch = 50

    seg2d_path = "/repository/lisiqi/save_model/seg2d/seg_fuse_nyu/checkpoint.pth.tar"
    save_path = '/repository/lisiqi/save_model/NYU_RGBD/MMF_RGBD_Attention_Model'

    if not os.path.exists(save_path[:-25]):
        os.makedirs(save_path[:-25])
    model = AMFNet(seg2d_path, 12, batch_size, wo_att=False, wo_seg=False)

    train_dataset = TrainDataLoaderFused(NYU_SAMPLE_TXT_TRAIN, NYU_NPZ_PATH_TRAIN, 'train')
    val_dataset = TrainDataLoaderFused(NYU_SAMPLE_TXT_TEST, NYU_NPZ_PATH_TEST, 'test')
    print(train_dataset.__len__(), val_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    cri_weights = torch.FloatTensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    criterion = nn.CrossEntropyLoss(weight=cri_weights / torch.sum(cri_weights))
    optimizer = torch.optim.SGD(model.get_config_optim(lr, 0.1), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        train_loader.pin_memory = True
        val_loader.pin_memory = True
        cudnn.benchmark = True
    model.train()

    iter_per_epoch = int(train_dataset.__len__() / batch_size)
    iter, total_iter, epoch, best_acc, loss_history, val_history = checkpoint_restore(model, save_path, True)
    acc_history = []
    time_last = datetime.datetime.now()
    MAX_ITER = iter_per_epoch * max_epoch
    for _ in range(epoch, max_epoch):
        for i, (color, depth, label, label_weight, depth_mapping_3d) in enumerate(train_loader):
            if use_gpu:
                label = label.cuda(async=True)
                label_weight = label_weight.cuda(async=True)
                color = color.cuda(async=True)
                depth = depth.cuda(async=True)
                depth_mapping_3d = depth_mapping_3d.cuda(async=True)

            out1 = model(color, depth, depth_mapping_3d)

            out = out1.permute(0, 2, 3, 4, 1).contiguous().view(-1, 12)
            selectindex = torch.nonzero(label_weight.view(-1)).view(-1).cuda()
            filterLabel = torch.index_select(label.view(-1), 0, selectindex)
            filterOutput = torch.index_select(out, 0, selectindex)
            loss = criterion(filterOutput, filterLabel)

            loss_history = np.append(loss_history, loss.item())
            aver_loss = loss_history.sum() / loss_history.__len__()
            time_now = datetime.datetime.now()
            print(time_now, ' Cost Time %fs, Epoch[%d], Iter %d/%d, Total Iter:%d/%d, loss %f, average loss %f.'
                  % ((time_now - time_last).total_seconds(), epoch, iter, iter_per_epoch, total_iter, MAX_ITER,
                     loss.item(), aver_loss))
            time_last = time_now
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            iter = iter + 1
            total_iter = total_iter + 1

            if total_iter % iter_per_epoch == 0:
                save_dict = {'epoch': epoch, 'iter': iter, 'total_iter': total_iter, 'loss_history': loss_history,
                             'best_acc': best_acc, 'val_history': val_history}

                checkpoint_save(model, save_path, total_iter, save_dict, True)
                print('save done!')

                model.eval()
                val_loss = 0
                accuracy_total = np.zeros((3, 11), dtype=np.float32)
                print("Start Validate")
                for j, (color, depth, label, label_weight, depth_mapping_3d) in enumerate(val_loader):
                    if use_gpu:
                        label = label.cuda(async=True)
                        label_weight = label_weight.cuda(async=True)
                        color = color.cuda(async=True)
                        depth = depth.cuda(async=True)
                        depth_mapping_3d = depth_mapping_3d.cuda(async=True)

                    with torch.no_grad():
                        out1 = model(color, depth, depth_mapping_3d)
                        accuracy_total = accuracy_total + IOU.computeIOU(out1, label, 12)
                        out = out1.permute(0, 2, 3, 4, 1).contiguous().view(-1, 12)
                        selectindex = torch.nonzero(label_weight.view(-1)).view(-1).cuda()
                        filterLabel = torch.index_select(label.view(-1), 0, selectindex)
                        filterOutput = torch.index_select(out, 0, selectindex)
                        loss = criterion(filterOutput, filterLabel)
                        val_loss = val_loss + loss.item()
                    print("Test%d/%d, Loss: %f" % (j, val_loader.__len__(), loss.item()))
                val_loss = val_loss / val_loader.__len__()
                val_history = np.append(val_history, val_loss)
                accuracy = accuracy_total[0] / (np.sum(accuracy_total, 0) + 0.00001)
                accuracy_mean = np.mean(accuracy)
                acc_history.append(accuracy_mean)
                print(save_path)
                print("Epoch %d, Iteration %d, Validation Loss %f, Mean Accuracy %f"
                      % (epoch, iter, val_loss, accuracy_mean))
                print("Class Accuracy:", accuracy)
                print("Val_loss history: ", val_history)
                print("Accuracy history: ", acc_history)

                if accuracy_mean > best_acc:
                    best_acc = accuracy_mean
                    save_dict = {'epoch': epoch, 'iter': iter, 'total_iter': total_iter, 'loss_history': loss_history,
                                 'best_acc': best_acc, 'val_history': val_history}
                    checkpoint_save(model, save_path + 'bestacc(%f)' % accuracy_mean, total_iter, save_dict, True)
                    print("save done")
                torch.cuda.empty_cache()
                model.train()

            if iter % iter_per_epoch == 0:
                print("================================Epoch %d End================================" % epoch)
                epoch = epoch + 1
                iter = 0


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()

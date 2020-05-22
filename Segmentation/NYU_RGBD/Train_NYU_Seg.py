from NYU_Dataloader import TrainDataLoader
from Segmodel import Seg2DNet
import SegIOU as IOU
import argparse
import torch.nn as nn
import numpy as np
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn
import glob
import os
import datetime
import torch

parser = argparse.ArgumentParser(description='seg2D Training')
parser.add_argument('-eps', '--epochs', default=15, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-bs', '--batch_size', default=2, type=int, metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('-wd', '--weight_decay', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')


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
    args = parser.parse_args()
    print("Lr=%f, Bs=%d, mom=%f, Wd=%f, epoch=%d"
          % (args.learning_rate, args.batch_size,args.momentum,args.weight_decay, args.epochs))
    use_gpu = torch.cuda.is_available()
    train_dataset = TrainDataLoader('./nyu_train.txt', train=True)
    val_dataset = TrainDataLoader('./nyu_test.txt', train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    save_path = '/repository/lisiqi/save_model/Seg/NYU/Seg2D'
    if not os.path.exists(save_path[:-6]):
        os.makedirs(save_path[:-6])

    model = Seg2DNet(num_classes=12)
    cri_weights = torch.FloatTensor([0.0001, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5])
    criterion = nn.CrossEntropyLoss(weight=cri_weights/torch.sum(cri_weights))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()
        train_loader.pin_memory = True
        val_loader.pin_memory = True
        cudnn.benchmark = True
    model.train()

    iter_per_epoch = int(train_dataset.__len__() / args.batch_size)
    iter, total_iter, epoch, best_acc, loss_history, val_history = checkpoint_restore(model, save_path, True)
    acc_history = []
    time_last = datetime.datetime.now()
    MAX_ITER = iter_per_epoch * args.epochs

    for _ in range(epoch, args.epochs):
        for i, (color, depth, label) in enumerate(train_loader):
            bs, ch, hi, wi = color.size()
            color = color.view(bs*2, -1, hi, wi)
            depth = depth.view(bs*2, -1, hi, wi)
            label = label.view(bs*2, -1, wi)

            if use_gpu:
                label = label.cuda(async=True)
                color = color.cuda(async=True)
                depth = depth.cuda(async=True)

            out = model(color, depth)
            loss = criterion(out, label)
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
                accuracy_total = np.zeros((2, 11), dtype=np.float32)
                print("Start Validate")
                for j, (color, depth, label) in enumerate(val_loader):
                    bs, ch, hi, wi = color.size()
                    color = color.view(bs * 2, -1, hi, wi)
                    depth = depth.view(bs * 2, -1, hi, wi)
                    label = label.view(bs * 2, -1, wi)

                    if use_gpu:
                        label = label.cuda(async=True)
                        color = color.cuda(async=True)
                        depth = depth.cuda(async=True)

                    with torch.no_grad():
                        out = model(color, depth)
                        accuracy_total = accuracy_total + IOU.computeIOU(out, label, 12)
                        loss = criterion(out, label)
                        val_loss = val_loss + loss.item()
                    print("Test%d/%d, Loss: %f" % (j, val_loader.__len__(), loss.item()))
                val_loss = val_loss / val_loader.__len__()
                val_history = np.append(val_history, val_loss)
                accuracy_mean = np.mean(accuracy_total)
                acc_history.append(accuracy_mean)
                print(save_path)
                print("Epoch %d, Iteration %d, Validation Loss %f, Mean Accuracy %f"
                      % (epoch, iter, val_loss, accuracy_mean))
                print("Class Accuracy:", np.mean(accuracy_total, axis=0))
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
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    main()
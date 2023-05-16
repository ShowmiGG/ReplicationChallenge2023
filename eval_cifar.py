import os, sys, argparse, time, random
from functools import partial

sys.path.append('./')
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from resnet import ResNet18
from models.cifar10.resnet_DuBIN import ResNet18_DuBIN
from models.cifar10.resnet_DuBN import ResNet18_DuBN
from dataloaders.cifar10 import cifar_dataloaders, cifar_c_testloader, cifar10_1_testloader, \
    cifar_random_affine_test_set
from utils.utils import *

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

mode = 'all' # data to evaluate on. ['clean', 'c', 'v2', 'sta', 'all']

# dataset:
dataset = 'cifar10'
data_root_path = 'datasets/'
ckpt_path = ''
save_root_path = 'output\\'

# model:
model_fn = ResNet18_DuBIN
num_classes = 10 # number of classes
num_workers = 2 # number of workers
test_batch_size = 1000
k = 10 #hyperparameter k in worst-of-k spatial attack
init_stride = 1
model = model_fn(num_classes=num_classes, init_stride=init_stride).cuda()

# load model:
ckpt = torch.load(os.path.join(save_root_path, 'AugMax_results', ckpt_path, 'best_SA.pth'))
model.load_state_dict(ckpt)

ckpt = torch.load('output/AugMax_results/t_jsd.ckpt')
model.load_state_dict(ckpt['model_state_dict'])

# log file:
fp = open(os.path.join(save_root_path, 'AugMax_results', ckpt_path, 'test_results.txt'), 'a+')

## Test on CIFAR:
def val_cifar():
    _, val_data = cifar_dataloaders(data_dir=data_root_path,num_classes=num_classes,train_batch_size=256,test_batch_size=test_batch_size, num_workers=num_workers, AugMax=None)
    test_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            test_loss_meter.append(loss.item())
            test_acc_meter.append(acc.item())
    print('clean test time: %.2fs' % (time.time() - ts))
    # test loss and acc of this epoch:
    test_loss = test_loss_meter.avg
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'clean: %.4f' % test_acc
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()


def val_cifar_worst_of_k_affine(K):
    '''
    Test model robustness against spatial transform attacks using worst-of-k method on CIFAR10/100.
    '''
    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        K_loss = torch.zeros((K,test_batch_size)).cuda()
        K_logits = torch.zeros((K,test_batch_size, num_classes)).cuda()
        for k in range(K):
            random.seed(k + 1)
            val_data = cifar_random_affine_test_set(data_dir=data_root_path, num_classes=num_classes)
            test_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,pin_memory=True)
            images, targets = next(iter(test_loader))
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets, reduction='none')
            # stack all losses:
            K_loss[k, :] = loss  # shape=(K,N)
            K_logits[k, ...] = logits
        # print('K_loss:', K_loss[:,0:3], K_loss.shape)
        adv_idx = torch.max(K_loss, dim=0).indices
        logits_adv = torch.zeros_like(logits).to(logits.device)
        for n in range(images.shape[0]):
            logits_adv[n] = K_logits[adv_idx[n], n, :]
        print('logits_adv:', logits_adv.shape)
        pred = logits_adv.data.max(1)[1]
        print('pred:', pred.shape)
        acc = pred.eq(targets.data).float().mean()
        # append loss:
        test_acc_meter.append(acc.item())
    print('worst of %d test time: %.2fs' % (K, time.time() - ts))
    # test loss and acc of this epoch:
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'worst of %d: %.4f' % (K, test_acc)
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()


def val_cifar_c():
    '''
    Evaluate on CIFAR10/100-C
    '''
    test_seen_c_loader_list = []
    for corruption in CORRUPTIONS:
        test_c_loader = cifar_c_testloader(corruption=corruption, data_dir=data_root_path, num_classes=num_classes,test_batch_size=test_batch_size, num_workers=num_workers)
        test_seen_c_loader_list.append(test_c_loader)

    # val corruption:
    print('evaluating corruptions...')
    test_c_losses, test_c_accs = [], []
    for corruption, test_c_loader in zip(CORRUPTIONS, test_seen_c_loader_list):
        test_c_batch_num = len(test_c_loader)
        print(test_c_batch_num)  # each corruption has 10k * 5 images, each magnitude has 10k images
        ts = time.time()
        test_c_loss_meter, test_c_acc_meter = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_c_loader):
                images, targets = images.cuda(), targets.cuda()
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                acc = pred.eq(targets.data).float().mean()
                # append loss:
                test_c_loss_meter.append(loss.item())
                test_c_acc_meter.append(acc.item())

        print('%s test time: %.2fs' % (corruption, time.time() - ts))
        # test loss and acc of each type of corruptions:
        test_c_losses.append(test_c_loss_meter.avg)
        test_c_accs.append(test_c_acc_meter.avg)

        # print
        corruption_str = '%s: %.4f' % (corruption, test_c_accs[-1])
        print(corruption_str)
        fp.write(corruption_str + '\n')
        fp.flush()
    # mean over 16 types of attacks:
    test_c_loss = np.mean(test_c_losses)
    test_c_acc = np.mean(test_c_accs)

    # print
    avg_str = 'corruption acc: (mean) %.4f' % (test_c_acc)
    print(avg_str)
    fp.write(avg_str + '\n')
    fp.flush()


def val_cifar10_1():
    '''
    Evaluate on cifar10.1
    '''
    test_v2_loader = cifar10_1_testloader(data_dir=os.path.join(data_root_path))

    model.eval()
    ts = time.time()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for images, targets in test_v2_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            test_loss_meter.append(loss.item())
            test_acc_meter.append(acc.item())
    print('cifar10.1 test time: %.2fs' % (time.time() - ts))
    # test loss and acc of this epoch:
    test_loss = test_loss_meter.avg
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'cifar10.1 test acc: %.4f' % test_acc
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()


if __name__ == '__main__':
    model.apply(lambda m: setattr(m, 'route', 'M'))

    if mode in ['clean', 'all']:
        val_cifar()
    if mode in ['c', 'all']:
        val_cifar_c()
    if mode in ['v2']:
        val_cifar10_1()
    if mode in ['sta']:
        val_cifar_worst_of_k_affine(k)

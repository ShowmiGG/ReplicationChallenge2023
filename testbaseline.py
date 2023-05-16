import sys,time
sys.path.append('./')

import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloaders.tiny_imagenet import tiny_imagenet_dataloaders, tiny_imagenet_c_testloader

from utils.utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


model = torch.load('current.pth')
file = open('eval_results.txt', 'a+')


## mCE weights:
ResNet18_c_CE_list = [
    0.8037, 0.7597, 0.7758, 0.8426, 0.8274,
    0.7907, 0.8212, 0.7497, 0.7381, 0.7433,
    0.6800, 0.8939, 0.7308, 0.6121, 0.6452
]


def find_mCE(target_model_c_CE, anchor_model_c_CE):
    assert len(target_model_c_CE) == 15
    mCE = 0
    for target_model_CE, anchor_model_CE in zip(target_model_c_CE, anchor_model_c_CE):
        mCE += target_model_CE / anchor_model_CE
    mCE /= len(target_model_c_CE)
    return mCE


def val_tin():
    _, val_data = tiny_imagenet_dataloaders('tiny-imagenet-200')
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    test_loss_meter, test_acc_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            test_loss_meter.append(loss.item())
            test_acc_meter.append(acc.item())
    test_acc = test_acc_meter.avg

    # print:
    clean_str = 'clean acc: %.4f' % test_acc
    print(clean_str)
    file.write(clean_str + '\n')


def val_tin_c():
    '''
    Evaluate on Tiny ImageNet-C
    '''
    test_seen_c_loader_list = []
    for corruption in CORRUPTIONS:
        test_seen_c_loader_list_c = []
        for severity in range(1, 6):
            test_c_loader_c_s = tiny_imagenet_c_testloader(
                data_dir=os.path.join('./data', 'TinyImageNet-C/Tiny-ImageNet-C'),
                corruption=corruption, severity=severity,
                test_batch_size=32, num_workers=4)
            test_seen_c_loader_list_c.append(test_c_loader_c_s)
        test_seen_c_loader_list.append(test_seen_c_loader_list_c)

    model.eval()
    # val corruption:
    print('evaluating corruptions...')
    test_CE_c_list = []
    for corruption, test_seen_c_loader_list_c in zip(CORRUPTIONS, test_seen_c_loader_list):
        test_c_CE_c_s_list = []
        ts = time.time()
        for severity in range(1, 6):
            test_c_loader_c_s = test_seen_c_loader_list_c[severity - 1]
            test_c_batch_num = len(test_c_loader_c_s)
            # print(test_c_batch_num) # each corruption has 10k * 5 images, each magnitude has 10k images
            test_c_loss_meter, test_c_CE_meter = AverageMeter(), AverageMeter()
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(test_c_loader_c_s):
                    images, targets = images.cuda(), targets.cuda()
                    logits = model(images)
                    loss = F.cross_entropy(logits, targets)
                    pred = logits.data.max(1)[1]
                    CE = (~pred.eq(targets.data)).float().mean()
                    # append loss:
                    test_c_loss_meter.append(loss.item())
                    test_c_CE_meter.append(CE.item())

            # test loss and acc of each type of corruptions:
            test_c_CE_c_s = test_c_CE_meter.avg
            test_c_CE_c_s_list.append(test_c_CE_c_s)
        test_CE_c = np.mean(test_c_CE_c_s_list)
        test_CE_c_list.append(test_CE_c)

        # print
        print('%s test time: %.2fs' % (corruption, time.time() - ts))
        corruption_str = '%s CE: %.4f' % (corruption, test_CE_c)
        print(corruption_str)
        file.write(corruption_str + '\n')
        file.flush()
    # mean over 16 types of corruptions:
    test_c_acc = 1 - np.mean(test_CE_c_list)
    # weighted mean over 16 types of corruptions:
    test_mCE = find_mCE(test_CE_c_list, anchor_model_c_CE=ResNet18_c_CE_list)

    # print
    avg_str = 'corruption acc: %.4f' % (test_c_acc)
    print(avg_str)
    file.write(avg_str + '\n')
    mCE_str = 'mCE: %.4f' % test_mCE
    print(mCE_str)
    file.write(mCE_str + '\n')
    file.flush()





if __name__ == '__main__':

        val_tin()
        val_tin_c()


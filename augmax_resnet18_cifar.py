'''
Training with AugMax data augmentation
'''
import os, sys, argparse, time

sys.path.append('./')

import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from augmax_modules.augmax import AugMaxDataset, AugMaxModule, AugMixModule

from models.cifar10.resnet_DuBIN import ResNet18_DuBIN

from dataloaders.cifar10 import cifar_dataloaders

from utils.utils import *
from utils.context import ctx_noparamgrad_and_eval
from utils.attacks import AugMaxAttack, FriendlyAugMaxAttack

### config
model_fn = ResNet18_DuBIN
data_root_path = 'datasets/'

train_batch_size = 256 # training batch size
test_batch_size = 4# test batch size
num_workers = 2 # number of workers
num_classes = 10 # number of classes
epochs = 50 #200 # Number of epochs to train.

mixture_width = 3 #Number of augmentation chains to mix per augmented example
mixture_depth = -1 #Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]
aug_severity = 3 #Severity of base augmentation operators

Lambda = 15 #Trade-off hyper-parameter in loss function.
attacker = 'fat' #How to solve the inner maximization problem.
tau = 1 #Early stop iteration for FAT.
steps = 5 #The maximum iteration for the attack (FAT/PGD)
targeted = True #If true, targeted attack
alpha = 0.1 #attack step size

lr = 0.1 # learning rate
momentum = 0.9 # momentum
wd = 0.0005 # weight decay

# create saved folder
save_root_path = 'output\\'
dataset_str = 'cifar10'
model_str = model_fn.__name__
loss_str = 'Lambda%s' % Lambda
opt_str = 'e%d-b%d_adam-lr%s-wd%s' % (epochs, train_batch_size, lr, wd)
decay_str = 'cos'
attack_str = ('%s-%s' % (attacker, tau) if attacker == 'fat' else attacker) + '-' + ('targeted' if targeted else 'untargeted') + '-%d-%s' % (steps, alpha)
save_folder = os.path.join(os.getcwd(), save_root_path, 'AugMax_results\\augmax_training', dataset_str, model_str, '%s_%s_%s_%s' % (attack_str, loss_str, opt_str, decay_str))
create_dir(save_folder)
print('saving to %s' % save_folder)

def train():
    # initialise device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # define dataset
    train_data, val_data = cifar_dataloaders(data_dir=data_root_path, num_classes=num_classes,AugMax=AugMaxDataset, mixture_width=mixture_width,mixture_depth=mixture_depth, aug_severity=aug_severity)

    # define dataloader
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    # define model
    model = model_fn(num_classes=num_classes, init_stride=1).to(device)

    # define optimiser
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    # define learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # define attacker
    attacker = FriendlyAugMaxAttack(steps=steps, alpha=alpha, tau=tau, targeted=targeted)
    augmix_model = AugMixModule(mixture_width, device=device)
    augmax_model = AugMaxModule(device=device)


    best_SA = 0
    training_loss, val_SA = [], []  # training curve lists:

    ###training
    print("Training")
    for epoch in range(epochs):
        fp = open(os.path.join(save_folder, 'train_log.txt'), 'a+')
        start_time = time.time()

        model.train()
        requires_grad_(model, True)
        print(model.training)
        accs, accs_augmax, accs_augmix, losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for i, (images_tuples, labels) in enumerate(train_loader):
            # get batch:
            images_tuple = images_tuples[0]
            images_tuple = [images.to(device) for images in images_tuple]
            images_tuple_2 = images_tuples[1]
            images_tuple_2 = [images.to(device) for images in images_tuple_2]
            labels = labels.to(device)

            # AUGMAX
            if 'DuBIN' in model_fn.__name__: model.apply(
                lambda m: setattr(m, 'route', 'A'))  # use auxilary BN for AugMax images
            # generate and forward augmax images:
            with ctx_noparamgrad_and_eval(model):
                # generate augmax images:
                if targeted:
                    targets = torch.fmod(labels + torch.randint(low=1, high=num_classes, size=labels.size()).to(device),num_classes)
                    imgs_augmax_1, _, _ = attacker.attack(augmax_model, model, images_tuple, labels=labels,targets=targets, device=device)
                else:
                    imgs_augmax_1, _, _ = attacker.attack(augmax_model, model, images_tuple, labels=labels,device=device)

            logits_augmax_1 = model(imgs_augmax_1.detach())  # logits for augmax imgs:

            # AUGMIX
            if 'DuBIN' in model_fn.__name__: model.apply(
                lambda m: setattr(m, 'route', 'M'))  # use main BN for normal images
            imgs_augmix_1 = augmix_model(images_tuple_2)  # generate augmix images:
            logits_augmix_1 = model(imgs_augmix_1.detach())  # logits for augmix imgs:

            # CLEAN
            logits = model(images_tuple[0])  # logits for clean imgs:

            # calculate loss
            loss_clean = F.cross_entropy(logits, labels)
            p_clean, p_aug1, p_aug2 = F.softmax(logits, dim=1), F.softmax(logits_augmax_1, dim=1), F.softmax(
                logits_augmix_1, dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss_cst = Lambda * (
                    F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')
            ) / 3.
            loss = loss_clean + loss_cst

            # gradient optimisation
            optimizer.zero_grad() # return gradient
            loss.backward() # calculate gradients
            optimizer.step() # update weights

            # metrics:
            accs.append((logits.argmax(1) == labels).float().mean().item())
            accs_augmax.append((logits_augmax_1.argmax(1) == labels).float().mean().item())
            losses.append(loss.item())

            if i % 50 == 0:
                train_str = 'Epoch %d-%d | Train | Loss: %.4f (%.4f, %.4f), SA: %.4f, RA: %.4f' % (epoch, i, losses.avg, loss_clean, loss_cst, accs.avg, accs_augmax.avg)
                print(train_str)

        # lr schedualr update at the end of each epoch:
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        ### validation
        print("Validating")
        model.eval()
        requires_grad_(model, False)
        print(model.training)

        eval_this_epoch = (epoch % 10 == 0) or (epoch>=int(0.7*epochs)) # boolean

        if eval_this_epoch:
            val_SAs = AverageMeter()
            if 'DuBN' in model_fn.__name__ or 'DuBIN' in model_fn.__name__:
                model.apply(lambda m: setattr(m, 'route', 'M'))  # use main BN

            for i, (imgs, labels) in enumerate(val_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                # logits for clean imgs:
                logits = model(imgs)
                val_SAs.append((logits.argmax(1) == labels).float().mean().item())
            val_str = 'Epoch %d | Validation | Time: %.4f | lr: %s | SA: %.4f' % (epoch, (time.time() - start_time), current_lr, val_SAs.avg)
            print(val_str)
            fp.write(val_str + '\n')

            if val_SAs.avg >= best_SA:
                best_SA = val_SAs.avg
                torch.save(model.state_dict(), os.path.join(save_folder, 'best_SA.pth'))

        # save loss curve:
        training_loss.append(losses.avg)
        plt.plot(training_loss)
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, 'training_loss.png'))
        plt.close()

        val_SA.append(val_SAs.avg)
        plt.plot(val_SA, 'r')
        plt.grid(True)
        plt.savefig(os.path.join(save_folder, 'val_SA.png'))
        plt.close()

if __name__ == '__main__':
    train()





"""
Training MoCo and Instance Discrimination

InsDis: Unsupervised feature learning via non-parametric instance discrimination
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

"""
from __future__ import print_function

import os
import sys
sys.path.append('.')
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
from util import adjust_learning_rate, AverageMeter, adjust_learning_rate_simclr

from models.resnet import InsResNet50
from models.resnet_cifar import InsResNet50_cifar

from NCE.NCEAverage import MemoryInsDis
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
from NCE.BatchAverage import BatchCriterion

from dataset import ImageFolderInstance, CIFAR10Instance, CIFAR10Instance_double
from utils_cifar import *

from torchlars import LARS

try:
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=50, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1.0, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr_warmup', type=float, default=10, help='Linear warm-up cycle')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list') # For moco/CMC/InsDis

    # cosine with warm restart
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--lr_cycle', type=float, default=10, help='Cosine decay with warm restart cycle interval')

    parser.add_argument('--filter_ED', type=str, default='bn,bias', help='Filter out these parameters when performing weight decay')
    parser.add_argument('--LARS', action='store_true', help='LARS optimizer, cannot be used with amp')
    # parser.add_argument('--warm', action='store_true', help='add linear warm-up setting')

    # amp
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])
    parser.add_argument('--amp', action='store_true', help='using mixed precision, cannot not be used with LARS')

    # crop
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar', choices=['imagenet100', 'imagenet', 'cifar'])

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # augmentation setting
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ', 'simple'])

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'resnet50x2', 'resnet50x4', 'resnet50_cifar'])

    # loss function
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)

    # memory setting
    parser.add_argument('--contrastive_model', type=str, default='simclr',  choices=['simclr', 'moco', 'insdis'], help='which contrastive_model to use')
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')



    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')




    opt = parser.parse_args()

    # # set the path according to the environment
    # if hostname.startswith('visiongpu'):
    #     opt.data_folder = '/dev/shm/yonglong/{}'.format(opt.dataset)
    #     opt.model_path = '/data/vision/phillipi/rep-learn/Pedesis/CMC/{}_models'.format(opt.dataset)
    #     opt.tb_path = '/data/vision/phillipi/rep-learn/Pedesis/CMC/{}_tensorboard'.format(opt.dataset)
    # else:
    #     raise NotImplementedError('server invalid: {}'.format(hostname))

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop = 0.08

    opt.filter_WD = opt.filter_WD.split(',')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    if opt.softmax == 'softmax':
        prefix = 'softmax{}'.format(opt.alpha)
    elif opt.contrastive_model == 'simclr':
        prefix = 'simclr'
        opt.nce_k = opt.batch_size

    else:
        prefix = 'nce{}'.format(opt.alpha)

    opt.model_name = '{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_crop_{}'.format(prefix, opt.model, opt.nce_k, opt.model,
                                                                        opt.learning_rate, opt.weight_decay,
                                                                        opt.batch_size, opt.crop)

    # if opt.warm:
    #     opt.model_name = '{}_warm'.format(opt.model_name)
    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    opt.model_name = '{}_aug_{}'.format(opt.model_name, opt.aug)
    opt.model_name += suffix

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1.detach().data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
    rnd_color_jitter,
    rnd_gray])
    return color_distort

def add_weight_decay(model, weight_decay=1e-6, skip_list=()):
    """
    reference: https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        for to_filter in skip_list:
            if to_filter in name:
                no_decay.append(param)

        if name not in no_decay:
            decay.append(param)

        # if len(param.shape) == 1 or name in skip_list:
        #     no_decay.append(param)
        # else:
        #     decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def main():

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # set the data loader
    data_folder = os.path.join(args.data_folder, 'train')
    val_folder = os.path.join(args.data_folder, 'val')

    crop_padding = 32
    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)


    if args.aug == 'NULL' and args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.aug == 'CJ':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    # elif args.aug == 'NULL' and args.dataset == 'cifar':
    #     train_transform = transforms.Compose([
    #         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    #
    #     test_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])
    elif args.aug == 'simple' and args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            get_color_distortion(1.0),
            transforms.ToTensor(),
            normalize,
        ])

        # TODO: Currently follow CMC
        test_transform = transforms.Compose([
            transforms.Resize(image_size + crop_padding),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.aug == 'simple' and args.dataset == 'cifar':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(p=0.5),
            get_color_distortion(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    else:
        raise NotImplemented('augmentation not supported: {}'.format(args.aug))


    # Get Datasets
    if args.dataset == "imagenet":
        train_dataset = ImageFolderInstance(data_folder, transform=train_transform, two_crop=args.moco)
        print(len(train_dataset))
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        test_dataset = datasets.ImageFolder(
            val_folder,
            transforms=test_transform
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)


    elif args.dataset == 'cifar':
        # cifar-10 dataset
        if args.contrastive_model == 'simclr':
            train_dataset = CIFAR10Instance_double(root='./data', train=True, download=True, transform=train_transform, double=True)
        else:
            train_dataset = CIFAR10Instance(root='./data', train=True, download=True, transform=train_transform)
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

        test_dataset = CIFAR10Instance(root='./data', train=False, download=True, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=100, shuffle=False, num_workers=args.num_workers)


    # create model and optimizer
    n_data = len(train_dataset)

    if args.model == 'resnet50':
        model = InsResNet50()
        if args.contrastive_model == 'moco':
            model_ema = InsResNet50()
    elif args.model == 'resnet50x2':
        model = InsResNet50(width=2)
        if args.contrastive_model == 'moco':
            model_ema = InsResNet50(width=2)
    elif args.model == 'resnet50x4':
        model = InsResNet50(width=4)
        if args.contrastive_model == 'moco':
            model_ema = InsResNet50(width=4)
    elif args.model == 'resnet50_cifar':
        model = InsResNet50_cifar()
        if args.contrastive_model == 'moco':
            model_ema = InsResNet50_cifar()
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))



    # copy weights from `model' to `model_ema'
    if args.contrastive_model == 'moco':
        moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    if args.contrastive_model == 'moco':
        contrast = MemoryMoCo(128, n_data, args.nce_k, args.nce_t, args.softmax).cuda(args.gpu)
    elif args.contrastive_model == 'simclr':
        contrast = None
    else:
        contrast = MemoryInsDis(128, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax).cuda(args.gpu)

    if args.softmax:
        criterion = NCESoftmaxLoss()
    elif args.contrastive_model == 'simclr':
        criterion = BatchCriterion(1, args.nce_t, args.batch_size)
    else:
        criterion = NCECriterion(n_data)
    criterion = criterion.cuda(args.gpu)

    model = model.cuda()
    if args.contrastive_model == 'moco':
        model_ema = model_ema.cuda()



    # Exclude BN and bias if needed
    weight_decay = args.weight_decay
    if weight_decay and args.filter_WD:
        parameters = add_weight_decay(model, weight_decay, [''])
        weight_decay = 0.
    else:
        parameters = model.parameters()

    optimizer = torch.optim.SGD(parameters,
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=weight_decay)
    cudnn.benchmark = True

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        if args.contrastive_model == 'moco':
            optimizer_ema = torch.optim.SGD(model_ema.parameters(),
                                            lr=0,
                                            momentum=0,
                                            weight_decay=0)
            model_ema, optimizer_ema = amp.initialize(model_ema, optimizer_ema, opt_level=args.opt_level)

    if args.LARS:
        optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)


    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if contrast:
                contrast.load_state_dict(checkpoint['contrast'])
            if args.contrastive_model == 'moco':
                model_ema.load_state_dict(checkpoint['model_ema'])

            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        print("==> training...")

        time1 = time.time()
        if args.contrastive_model == 'moco':
            loss, prob = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args)
        elif args.contrastive_model == 'simclr':
            print("Train using simclr")
            loss, prob = train_simclr(epoch, train_loader, model, criterion, optimizer, args)
        else:
            print("Train using InsDis")
            loss, prob = train_ins(epoch, train_loader, model, contrast, criterion, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('ins_loss', loss, epoch)
        logger.log_value('ins_prob', prob, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)



        test_epoch = 2
        if epoch % test_epoch == 0:
            model.eval()

            if args.contrastive_model == 'moco':
                model_ema.eval()

            print('----------Evaluation---------')
            start = time.time()

            if args.dataset == 'cifar':
                acc = kNN(epoch, model, train_loader, test_loader, 200, args.nce_t, n_data, low_dim=128, memory_bank=None)

            print("Evaluation Time: '{}'s".format(time.time() - start))
            # writer.add_scalar('nn_acc', acc, epoch)
            logger.log_value('Test accuracy', acc, epoch)

            # print('accuracy: {}% \t (best acc: {}%)'.format(acc, best_acc))
            print('[Epoch]: {}'.format(epoch))
            print('accuracy: {}%)'.format(acc))
            # test_log_file.flush()


        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                # 'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.contrastive_model == 'moco':
                state['model_ema'] = model_ema.state_dict()
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        # saving the model
        print('==> Saving...')
        state = {
            'opt': args,
            'model': model.state_dict(),
            # 'contrast': contrast.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if args.contrastive_model == 'moco':
            state['model_ema'] = model_ema.state_dict()
        if args.amp:
            state['amp'] = amp.state_dict()
        save_file = os.path.join(args.model_folder, 'current.pth')
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()



def train_simclr(epoch, train_loader, model, criterion, optimizer, opt):
    """
    one epoch training for instance discrimination
    """

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    c = 0
    for idx, (inputs, inputs2, _, _) in enumerate(train_loader):
        # c += 1
        # if c == 3:
        #     break

        adjust_learning_rate_simclr(epoch, opt, optimizer=optimizer, warmup_epoch=opt.lr_warmup,
                                    cycle_interval=opt.lr_cycle, step_in_epoch=idx,
                                    total_steps_in_epoch=len(train_loader))

        data_time.update(time.time() - end)
        # inputs = torch.cat((inputs,inputs2), 0)
        # bsz = inputs.size(0)

        inputs = inputs.float()
        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu, non_blocking=True)
            inputs2 = inputs2.cuda(opt.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
            inputs2 = inputs2.cuda()
        # index = index.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        # print("Training input size:", inputs.shape)
        feat = model(inputs)
        bsz = inputs.size(0)

        # Shuffle BN
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)
        bsz *= 2

        inputs2 = inputs2[shuffle_ids]
        feat_2 = model(inputs2)
        feat_2 = feat_2[reverse_ids]
        feat = torch.cat([feat, feat_2], 0)

        loss, prob = criterion(feat)
        # prob = out[:, 0].mean()
        prob = prob.mean()

        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, prob=prob_meter))
            # print(out.shape)
            sys.stdout.flush()

    return loss_meter.avg, prob_meter.avg


def train_ins(epoch, train_loader, model, contrast, criterion, optimizer, opt):
    """
    one epoch training for instance discrimination
    """
    adjust_learning_rate(epoch, opt, optimizer)

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        inputs = inputs.float()
        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
        index = index.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        # print("Training input size:", inputs.shape)
        feat = model(inputs)
        out = contrast(feat, index)

        loss = criterion(out)
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, prob=prob_meter))
            print(out.shape)
            sys.stdout.flush()

    return loss_meter.avg, prob_meter.avg


def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, opt):
    """
    one epoch training for instance discrimination
    """
    adjust_learning_rate(epoch, opt, optimizer)

    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        inputs = inputs.float()
        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
        index = index.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        x1, x2 = torch.split(inputs, [3, 3], dim=1)

        # ids for ShuffleBN
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)

        feat_q = model(x1)
        with torch.no_grad():
            x2 = x2[shuffle_ids]
            feat_k = model_ema(x2)
            feat_k = feat_k[reverse_ids]

        out = contrast(feat_q, feat_k)

        loss = criterion(out)
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, prob=prob_meter))
            print(out.shape)
            sys.stdout.flush()

    return loss_meter.avg, prob_meter.avg


if __name__ == '__main__':
    main()

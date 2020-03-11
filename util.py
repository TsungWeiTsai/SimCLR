from __future__ import print_function

import torch
import numpy as np
import argparse


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def adjust_learning_rate_simclr(epoch, args, optimizer, warmup_epoch, cycle_interval, step_in_epoch, total_steps_in_epoch):
    lr = args.learning_rate * args.batch_size / 256.0   # LARS adjustment

    epoch = epoch + step_in_epoch / total_steps_in_epoch
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    # lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    if epoch <= warmup_epoch:
        lr = linear_rampup(epoch, warmup_epoch) * lr
    else:
        lr *= cosine_rampdown(epoch-warmup_epoch, args.epochs)

    eps = 1e-9
    lr += eps

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    # print('rd', current, rampdown_length)
    assert 0 <= current <= rampdown_length
    return max(0., float(.5 * (np.cos(np.pi * current / rampdown_length) + 1)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    # meter = AverageMeter()
    parser = argparse.ArgumentParser('argument for training')
    args = parser.parse_args()
    args.epochs = 300
    args.batch_size = 256
    args.learning_rate = 1.0


    for epoch in range(0, args.epochs):
        tmp = adjust_learning_rate_simclr(epoch, args, optimizer=None, warmup_epoch=10, cycle_interval=10, step_in_epoch=1,
                                    total_steps_in_epoch=1)
        print(epoch, tmp)
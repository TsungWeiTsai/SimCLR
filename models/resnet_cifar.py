'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.normalize import Normalize
import math
from torch.autograd import Variable

import torch
from torch.autograd import Variable
from torch import nn


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, pool_len=4, low_dim=128, width=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.base = int(64 * width)
        self.layer1 = self._make_layer(block, self.base, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.base * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, num_blocks[3], stride=2)

        self.mlp = nn.Sequential(
                nn.Linear(self.base*8*block.expansion, self.base*8*block.expansion, bias=False),
                nn.BatchNorm1d(self.base*8*block.expansion),
                nn.ReLU(inplace=True),
                nn.Linear(self.base*8*block.expansion, low_dim, bias=False),
                nn.BatchNorm1d(low_dim),
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=.01)
        self.mlp.apply(init_weights)


        self.softmax = nn.Softmax(dim=1)

        self.l2norm = Normalize(2)
        self.pool_len = pool_len
        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=7):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.pool_len)
        out = out.view(out.size(0), -1)

        if layer == 6:
            return out
            # return self.l2norm(out)

        # out = self.linear(out)
        out = self.mlp(out)

        # return self.softmax(out)

        out = self.l2norm(out)
        return out


def ResNet18(pool_len = 4, low_dim=128):
    return ResNet(BasicBlock, [2,2,2,2], pool_len, low_dim)

def ResNet34(pool_len = 4, low_dim=128):
    return ResNet(BasicBlock, [3,4,6,3], pool_len, low_dim)

def ResNet50(pool_len = 4, low_dim=128, **kwargs):
    return ResNet(Bottleneck, [3,4,6,3], pool_len, low_dim, **kwargs)

def ResNet101(pool_len = 4, low_dim=128):
    return ResNet(Bottleneck, [3,4,23,3], pool_len, low_dim)

def ResNet152(pool_len = 4, low_dim=128):
    return ResNet(Bottleneck, [3,8,36,3], pool_len, low_dim)


class InsResNet50_cifar(nn.Module):
    """Encoder for instance discrimination and MoCo"""
    def __init__(self, width=1, low_dim=128):
        super(InsResNet50_cifar, self).__init__()
        self.encoder = ResNet50(width=width, low_dim=low_dim)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=7):
        return self.encoder(x, layer)




def test():
    net = ResNet18()
    # y = net(Variable(torch.randn(1,3,32,32)))
    # pdb.set_trace()
    y = net(Variable(torch.randn(1,3,96,96)))
    # pdb.set_trace()
    print(y.size())

# test()

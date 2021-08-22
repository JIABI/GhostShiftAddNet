import argparse
import os, time
import torch
torch.cuda.empty_cache()
import shutil
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
from ptflops import get_model_complexity_info
from torch.autograd import Variable
import models
import optim
import torch.backends.cudnn as cudnn
from torch_poly_lr_decay import PolynomialLRDecay
import torchsummary
from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type
import distutils
import matplotlib.pyplot as plt
from adder import adder as adder_fast
import deepshift
from pathlib import Path
import collections
from collections import OrderedDict

# Training settings
parser = argparse.ArgumentParser(description='PyTorch AdderNet Trainning')
parser.add_argument('--dataset', type=str, default='imagenet', help='training dataset')
parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=256, metavar='N', help='batch size for testing')
parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='restart point')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120], help='learning rate schedule')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--lr-sign', default=None, type=float, help='separate initial learning rate for sign params')
parser.add_argument('--lr_decay', default='stepwise', type=str, choices=['stepwise', 'poly'])
parser.add_argument('--optimizer', type=str, default='sgd', help='used optimizer')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH', help='path to save prune model')
parser.add_argument('--save_suffix', default='retrain_', type=str, help='identify which retrain')
parser.add_argument('--arch', default='resnet20', type=str, help='architecture to use')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# shift hyper-parameters
parser.add_argument('--shift_depth', type=int, default=0, help='how many layers to convert to shift')
parser.add_argument('--shift_type', type=str, choices=['Q', 'PS'], help='shift type for representing weights')
parser.add_argument('--rounding', default='deterministic', choices=['deterministic', 'stochastic'])
parser.add_argument('--weight_bits', type=int, default=5, help='number of bits to represent the shift weights')
parser.add_argument('--use-kernel', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='whether using custom shift kernel')
# add hyper-parameters
parser.add_argument('--add_quant', type=bool, default=False, help='whether to quantize adder layer')
parser.add_argument('--add_bits', type=int, default=8, help='number of bits to represent the adder filters')
# visualization
parser.add_argument('--visualize', action="store_true", default=False, help='if use visualization')
# reinit
parser.add_argument('--reinit', type=str, default=None, help='whether reinit or finetune')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

cudnn.benchmark = True

gpu = args.gpu_ids
gpu_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for gpu_id in gpu_ids:
    id = int(gpu_id)
    args.gpu_ids.append(id)
print(args.gpu_ids)
# if len(args.gpu_ids) > 0:
#    torch.cuda.set_device(args.gpu_ids[0])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'mnist':
    trainset = datasets.MNIST('../MNIST', download=True, train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )
    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = datasets.MNIST('../MNIST', download=True, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )
    )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, num_workers=4)
else:
    DATA = Path("D:/datasets/imagenet-mini")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(DATA / 'train', transform_train)
    test_dataset = datasets.ImageFolder(DATA / 'val', transform_test)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        pin_memory=True, num_workers=6)
    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    #train_dataset = datasets.ImageFolder(
    #    traindir,
    #    transforms.Compose([
    #        transforms.RandomResizedCrop(224),
    #       transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        normalize,
    #    ]))

    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset, batch_size=args.batch_size, shuffle=True,
    #    num_workers=16, pin_memory=True)

    #test_loader = torch.utils.data.DataLoader(
    #    datasets.ImageFolder(valdir, transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        normalize,
    #    ])),
    #    batch_size=args.test_batch_size, shuffle=False,
    #    num_workers=16, pin_memory=True)

if args.dataset == 'imagenet':
    num_cls = 1000
elif args.dataset == 'cifar10':
    num_cls = 10
elif args.dataset == 'cifar100':
    num_cls = 100
elif args.dataset == 'mnist':
    num_cls = 10
else:
    raise NotImplementedError('No such dataset!')

model = models.__dict__['resnet50_add'](num_classes=num_cls, quantize=args.add_quant, weight_bits=args.add_bits)
if args.reinit is not None:
    model_ref = models.__dict__[args.arch](num_classes=num_cls, quantize=args.add_quant, weight_bits=args.add_bits)

if 'shift' in args.arch: # no pretrain
    model, _ = convert_to_shift(model, args.shift_depth, args.shift_type, convert_weights=False, use_kernel=args.use_kernel, rounding=args.rounding, weight_bits=args.weight_bits)
    if args.reinit is not None:
        model_ref, _ = convert_to_shift(model_ref, args.shift_depth, args.shift_type, convert_weights=False, use_kernel=args.use_kernel, rounding=args.rounding, weight_bits=args.weight_bits)


# create optimizer
model_other_params = []
model_sign_params = []
model_shift_params = []

for name, param in model.named_parameters():
    if(name.endswith(".sign")):
        model_sign_params.append(param)
    elif(name.endswith(".shift")):
        model_shift_params.append(param)
    else:
        model_other_params.append(param)

params_dict = [
    {"params": model_other_params},
    {"params": model_sign_params, 'lr': args.lr_sign if args.lr_sign is not None else args.lr, 'weight_decay': 0},
    {"params": model_shift_params, 'lr': args.lr, 'weight_decay': 0}
    ]

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = None
if (args.optimizer.lower() == "sgd"):
    optimizer = torch.optim.SGD(params_dict, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adadelta"):
    optimizer = torch.optim.Adadelta(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adagrad"):
    optimizer = torch.optim.Adagrad(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adam"):
    optimizer = torch.optim.Adam(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "rmsprop"):
    optimizer = torch.optim.RMSprop(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "radam"):
    optimizer = optim.RAdam(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "ranger"):
    optimizer = optim.Ranger(params_dict, args.lr, weight_decay=args.weight_decay)
else:
    raise ValueError("Optimizer type: ", args.optimizer, " is not supported or known")

scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=args.epochs, end_learning_rate=0.0001, power=0.9)

def save_checkpoint(state, is_best, epoch, filepath):
    if epoch == 'init':
        filepath = os.path.join(filepath, 'init.pth.tar')
        torch.save(state, filepath)
    else:
        # filename = os.path.join(filepath, 'ckpt'+str(epoch)+'.pth.tar')
        # torch.save(state, filename)
        filename = os.path.join(filepath, 'ckpt.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        # args.start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    save_checkpoint({'state_dict': model.state_dict()}, False, epoch='init', filepath=args.save)

def change_name(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if 'conv' in key and '.1.weight' in key:
            new_key = key.replace('weight', 'adder')
        elif 'downsample' in key and '.1.weight' in key:
            new_key = key.replace('weight', 'adder')
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


if args.cuda:
    model.cuda()
    if args.reinit is not None:
        model_ref.cuda()
if len(args.gpu_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

if args.reinit is not None:
    print('use reinit!')
    # checkpoint = torch.load(args.reinit)
    # checkpoint['state_dict'] = change_name(checkpoint['state_dict'])
    # model_ref.load_state_dict(checkpoint['state_dict'])

    for m, m_ref in zip(model.modules(), model_ref.modules()):
        if isinstance(m, deepshift.modules_q.Conv2dShiftQ): # Q shift
            weight_copy = m.weight.data.abs().clone()
            # mask = weight_copy.gt(0).float().cuda()
            mask = torch.ne(weight_copy, 1.).float().cuda()
            m_ref.weight.data = m_ref.weight.data.mul_(mask) + 1 - mask # no shift
        elif isinstance(m, deepshift.modules.Conv2dShift): # PS shift
            weight_copy = m.shift.data.abs().clone()
            mask = weight_copy.gt(0).float().cuda()
            m_ref.shift.data.mul_(mask)
            m_ref.sign.data.mul_(mask)
        elif isinstance(m, adder_fast.Adder2D):
            adder_copy = m.adder.data.abs().clone()
            mask = adder_copy.gt(0).float().cuda()
            m_ref.adder.data.mul_(mask)
    model = model_ref


# print model summary
#model_summary = None
#try:
#model_summary, model_params_info = torchsummary.summary(model, (3,32,32))
#print(model_summary)
#print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
#except:
#    print("WARNING: Unable to obtain summary of model")

# save name
# name model sub-directory "shift_all" if all layers are converted to shift layers
conv2d_layers_count = count_layer_type(model, nn.Conv2d) #+ count_layer_type(model, unoptimized.UnoptimizedConv2d)
linear_layers_count = count_layer_type(model, nn.Linear) #+ count_layer_type(model, unoptimized.UnoptimizedLinear)
print('{:<30}  {:<8}'.format('conv2d counts: ', conv2d_layers_count))
print('{:<30}  {:<8}'.format('linear counts: ', linear_layers_count))

macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

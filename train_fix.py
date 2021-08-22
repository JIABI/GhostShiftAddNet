import argparse
import os, time
import torch
import shutil
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import models
import optim
import torch.backends.cudnn as cudnn
from cyclicLR import CyclicCosAnnealingLR

import matplotlib.pyplot as plt

from se_shift import SEConv2d, SELinear
from se_shift.utils_quantize import sparsify_and_nearestpow2
from se_shift.utils_swa import bn_update, moving_average
from se_shift.utils_optim import SGD
from adder.adder import Adder2D
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch AdderNet Trainning')
parser.add_argument('--data', type=str, default='/data3/imagenet-data/raw-data/', help='path to imagenet')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset')
parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=64, metavar='N', help='batch size for testing')
parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='restart point')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120], help='learning rate schedule')
parser.add_argument('--lr', type=float, default=0.25, metavar='LR', help='learning rate')
parser.add_argument('--lr-sign', default=None, type=float, help='separate initial learning rate for sign params')
parser.add_argument('--lr_decay', default='stepwise', type=str, choices=['stepwise', 'cosine', 'cyclic_cosine'])
parser.add_argument('--optimizer', type=str, default='sgd', help='used optimizer')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--resume', default='./ShiftAddNet_ckpt/shiftaddnet_fix/resnet20-cifar10-FIX8-Modify.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--save', default='./temp', type=str, metavar='PATH', help='path to save prune model')
parser.add_argument('--arch', default='resnet20_shiftadd_se', type=str, help='architecture to use')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
""" SE model arguments """
parser.add_argument('--threshold', type=float, default=7 * 1e-3, # (>= 2^-7)
                    help='Threshold in prune weight.')
parser.add_argument('--quant_each_step', action='store_true',
                    help='Sparsify and quantize coefficient matrcies after each training step.')
""" Bucket Switch arguments """
parser.add_argument('--switch', action='store_true', default=True,
                    help='Use Bucket Switch update scheme.')
parser.add_argument('--switch_bar', type=int, default=7, # 5 / 7
                    help='Minimal times of accumulated gradient directions before an update.')
parser.add_argument('--dweight_threshold', type=float, default=0.0008, # 1e-2 for mobilenet; need finetune
                    help='Threshold that filter small changes in Ce')
parser.add_argument('--max_weight', type=float, default=1, metavar='MC',
                    help='maximal magnitude in Ce matrices. not set by default')
# swa arguments
parser.add_argument('--swa', action='store_true', default=False, help='whether to use swa')
parser.add_argument('--swa_start', type=float, default=1, metavar='N',
                    help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR',
                    help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
# sparse
parser.add_argument('--sign_threshold', type=float, default=0.5, help='Threshold for pruning.')
parser.add_argument('--dist', type=str, default='normal', choices=['kaiming_normal', 'normal', 'uniform'])
parser.add_argument('--l1', action='store_true', default=True, help='whether sparse shift l1 norm')
# add hyper-parameters
parser.add_argument('--add_quant', type=bool, default=False, help='whether to quantize adder layer')
parser.add_argument('--add_bits', type=int, default=32, help='number of bits to represent the adder filters')
parser.add_argument('--add_sparsity', type=float, default=0.2, help='sparsity in adder filters')
# distributed parallel
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--port", type=str, default="10000")
parser.add_argument('--distributed', action='store_true', help='whether to use distributed training')
# eval only
parser.add_argument('--eval_only', action='store_true', help='whether only evaluation')

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
#(args.gpu_ids)
# if len(args.gpu_ids) > 0:
#    torch.cuda.set_device(args.gpu_ids[0])

if args.distributed:
    os.environ['MASTER_PORT'] = args.port
    torch.distributed.init_process_group(backend="nccl")


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
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=16, pin_memory=True)

if args.dataset == 'imagenet':
    num_classes = 1000
    model = models.__dict__[args.arch](threshold=args.threshold, sign_threshold=args.sign_threshold, distribution=args.dist, num_classes=1000,
                                       quantize=args.add_quant, weight_bits=args.add_bits, sparsity=args.add_sparsity)
    if args.swa:
        swa_model = models.__dict__[args.arch](threshold=args.threshold, sign_threshold=args.sign_threshold, distribution=args.dist, num_classes=1000,
                                       quantize=args.add_quant, weight_bits=args.add_bits, sparsity=args.add_sparsity)
elif args.dataset == 'cifar10':
    num_classes = 10
    model = models.__dict__['resnet20_shiftadd_se'](threshold=args.threshold, sign_threshold=args.sign_threshold, distribution=args.dist, num_classes=10,
                                       quantize=args.add_quant, weight_bits=args.add_bits, sparsity=args.add_sparsity)
    if args.swa:
        swa_model = models.__dict__[args.arch](threshold=args.threshold, sign_threshold=args.sign_threshold, distribution=args.dist, num_classes=10,
                                       quantize=args.add_quant, weight_bits=args.add_bits, sparsity=args.add_sparsity)
elif args.dataset == 'cifar100':
    num_classes = 100
    model = models.__dict__[args.arch](threshold=args.threshold, sign_threshold=args.sign_threshold, distribution=args.dist, num_classes=100,
                                       quantize=args.add_quant, weight_bits=args.add_bits, sparsity=args.add_sparsity)
    if args.swa:
        swa_model = models.__dict__[args.arch](threshold=args.threshold, sign_threshold=args.sign_threshold, distribution=args.dist, num_classes=100,
                                       quantize=args.add_quant, weight_bits=args.add_bits, sparsity=args.add_sparsity)
elif args.dataset == 'mnist':
    num_classes = 10
    model = models.__dict__[args.arch](threshold=args.threshold, sign_threshold=args.sign_threshold, distribution=args.dist, num_classes=10,
                                       quantize=args.add_quant, weight_bits=args.add_bits, sparsity=args.add_sparsity)
    if args.swa:
        swa_model = models.__dict__[args.arch](threshold=args.threshold, sign_threshold=args.sign_threshold, distribution=args.dist, num_classes=10,
                                       quantize=args.add_quant, weight_bits=args.add_bits, sparsity=args.add_sparsity)
else:
    raise NotImplementedError('No such dataset!')

if args.cuda:
    model.cuda()
    if args.swa:
        swa_model.cuda()
if len(args.gpu_ids) > 1:
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu_ids, find_unused_parameters=True)
        if args.swa:
            swa_model = torch.nn.parallel.DistributedDataParallel(swa_model, device_ids=args.gpu_ids)
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        if args.swa:
            swa_model = torch.nn.DataParallel(swa_model, device_ids=args.gpu_ids)

# create optimizer
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = None
if (args.optimizer.lower() == "sgd"):
    optimizer = SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adadelta"):
    optimizer = torch.optim.Adadelta(model.parameters(), args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adagrad"):
    optimizer = torch.optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adam"):
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "rmsprop"):
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "radam"):
    optimizer = optim.RAdam(model.parameters(), args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "ranger"):
    optimizer = optim.Ranger(model.parameters(), args.lr, weight_decay=args.weight_decay)
else:
    raise ValueError("Optimizer type: ", args.optimizer, " is not supported or known")

schedule_cosine_lr_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
scheduler_cyclic_cosine_lr_decay = CyclicCosAnnealingLR(optimizer, milestones=[40,60,80,100,140,180,200,240,280,300,340,400], decay_milestones=[100, 200, 300, 400], eta_min=0)

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

def save_checkpoint_swa(state, is_best, is_swa_best, epoch, filepath):
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
        if is_swa_best:
            shutil.copyfile(filename, os.path.join(filepath, 'swa_model_best.pth.tar'))

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # if 'epoch' in checkpoint.keys():
            # args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if ('swa_state_dict' in checkpoint.keys() and checkpoint['swa_state_dict'] is not None):
            swa_model.load_state_dict(checkpoint['swa_state_dict'])
        if 'swa_n' in checkpoint.keys() and checkpoint['swa_n'] is not None:
            swa_n = checkpoint['swa_n']
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    save_checkpoint({'state_dict': model.state_dict()}, False, epoch='init', filepath=args.save)

# set mask
if args.swa:
    for m, swa_m in zip(model.modules(), swa_model.modules()):
        if isinstance(m, (SEConv2d, SELinear,)):
            m.set_mask()
            swa_m.mask.data = m.mask.data.clone()
else:
    for m in model.modules():
        if isinstance(m, SEConv2d):
            m.set_mask()
print('All masks are set....')

""" Switch arguments setting. """
if args.switch:
    for i in range(-10, 1):
        if 2**i >= args.threshold:
            args.min_weight = 2**i
            break
shift_label = "shift-se"
if args.swa:
    shift_label += '-swa-lr-{}'.format(args.swa_lr)
if args.switch:
    shift_label += '-switch-bar-{}-max_weight-{}-min_weight-{}'.format(args.switch_bar, args.max_weight, args.min_weight)
shift_label += '-dweight_thre-{}'.format(args.dweight_threshold)
if args.add_quant:
    shift_label += '-add-{}'.format(args.add_bits)
if args.l1:
    shift_label += 'l1'

args.save = os.path.join(args.save, shift_label)
if not os.path.exists(args.save):
    os.makedirs(args.save)

# print(args.add_quant)
# exit()

history_score = np.zeros((args.epochs, 7))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

global total
total = 0
for m in model.modules():
    if isinstance(m, SEConv2d):
        total += m.weight.data.numel()

global total_add
total_add = 0
for m in model.modules():
    if isinstance(m, Adder2D):
        total_add += m.adder.data.numel()

def get_pruning_ratio(model):
    if 'shift' in args.arch:
        shift_masks = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, SEConv2d):
                size = m.weight.data.numel()
                shift_masks[index:(index+size)] = m.mask.data.view(-1).abs().clone()
                index += size
    print('Pruning ratio:'.format(np.round((1 - torch.sum(shift_masks)), 2)))
    return np.round((1 - torch.sum(shift_masks) / float(total)) * 100, 2)


if 'shift' in args.arch:
    shift_module = SEConv2d
if 'add' in args.arch:
    add_module = Adder2D

def get_shift_range(model):
    if 'shift' in args.arch:
        print('shift candidates:')
        total = 0
        for m in model.modules():
            if isinstance(m, shift_module):
                total += m.weight.data.numel()
        shift_weights = torch.zeros(total)
        shift_masks = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, shift_module):
                size = m.weight.data.numel()
                shift_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
                shift_masks[index:(index+size)] = m.mask.data.view(-1).abs().clone()
                index += size

        # y, i = torch.sort(shift_weights)
        weight_unique = torch.unique(shift_weights)
        #print(weight_unique)
        print('shift range:', weight_unique.size()[0]-1)
        weight_dist = []
        #for value in weight_unique:
        #    weight_dist.append(np.round((torch.sum(shift_weights == value) / float(total)) * 100, 2))
        #print(weight_dist)
        left_shift_weight = int(torch.sum(shift_weights == 0))
        left_shift_mask = int(torch.sum(shift_masks))
        print('pruning ratio:', (1 - left_shift_mask / float(total)) * 100, '%')
        print('left mask:', left_shift_mask)
        print('left weights:', left_shift_weight)
        print('total shift:', index)
        history_score[epoch][5] = left_shift_mask

def get_adder_sparsity(model):
    #if args.add_sparsity == 0:
    #    print('no sparisty in adder layer.')
    if 'add' in args.arch:
        adder_masks = torch.zeros(total_add)
        index = 0
        for m in model.modules():
            if isinstance(m, Adder2D):
                size = m.adder.data.numel()
                adder_masks[index:(index+size)] = m.adder_mask.data.view(-1).abs().clone()
                index += size
        left_adder_mask = int(torch.sum(adder_masks))
        print('left adder mask', left_adder_mask)
        print('Add sparsity ratio:', (1 - left_adder_mask / float(total_add)) * 100, '%')
        print('total adders:', total_add)
        history_score[epoch][6] = left_adder_mask

def train(epoch):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    end_time = time.time()
    feat_loader = []
    idx_loader = []
  #  l1 = None
 #   l1_regularization = torch.tensor(0).to(torch.device('cuda'))
    start_time = time.time()
    # reset ``dweight_counter`` in SE layers
    if args.switch:
        for m in model.modules():
            if hasattr(m, 'mask'):
                m.reset_dweight_counter()

    batch_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):

        # print('total time for one batch: {}'.format(time.time()-batch_time))
        batch_time = time.time()
        # l1_regularization = torch.tensor(0).to(torch.device('cuda'))
        # if args.quant_each_step:
        #     for m in model.modules():
        #         if hasattr(m, 'mask'):
        #             with torch.no_grad():
        #                 m.weight_prev = m.sparsify_and_quantize_weight()
        # start_time = time.time()
        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        data, target = Variable(data), Variable(target)
        # print('!!!!!!!!prepare data: ', time.time()-start_time)
        # start_time = time.time()
        optimizer.zero_grad()
        output = model(data)
        # print('!!!!!!!!forward time: ', time.time()-start_time)
        #if args.l1:
        loss = F.cross_entropy(output, target)
        #for param in model.parameters():
        #      l1 = torch.norm(param, 1)
        #      l1 = l1.type(torch.LongTensor)
        #      l1_regularization = l1

  #          if torch.sum(abs(cross_entropy_loss + l1_regularization) != 0):
           # mask = torch.sign(cross_entropy_loss)
        #loss = cross_entropy_loss - 0.0001 * l1_regularization
            #loss = cross_entropy_loss * mask
            #newmask = torch.sign(loss)
            #elif torch.sum(abs(cross_entropy_loss + l1_regularization) < 0):
            #      loss = cross_entropy_loss - l1_regularization

        #else:
        #       loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()
        # start_time = time.time()
        loss.backward()
        # torch.cuda.synchronize()



        # fix conv (used for normaladd variant)
        # for name, m in model.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.grad = None


        # update weight
        if args.switch:
            # Bucket Switching
          for name, m in model.named_modules():
              if not hasattr(m, 'mask'):
                   continue
              with torch.no_grad():
                    #pre = torch.sum(abs(m.weight.data) < args.threshold)
                    pre = m.weight.data
                    pre[pre.abs() <= args.threshold] = 0.0
                    pre_mask = pre.sign().float()
                    #print(pre_mask)
                    qweight = m.sparsify_and_quantize_weight()
                    dweight = optimizer.get_d(m.weight)
                    all_linear1_weight = sum(p.abs().sum() for p in pre)
                    linear1_weight = 0.000001*all_linear1_weight
                    if dweight is None:
                       continue
                    if args.dweight_threshold > 0.0:
                         # adative LR
                        #dweight = 0.1 * np.sqrt(dweight.numel()) / torch.norm(dweight) * dweight
    #                    # print(dweight.mean())
                        dweight[dweight.abs() <= args.dweight_threshold] = 0.0
                    # m.weight.grad = None
                    dweight_sign = dweight.sign().float()
     #                # update dweight_counter
                    m.dweight_counter.add_(dweight_sign)
                    activated = m.dweight_counter.abs() == args.switch_bar
                    dweight_sign = m.dweight_counter.sign() * activated.float()
     #                # weight nonzero and gradient nonzero
                    dweight_pow = dweight_sign * qweight.sign().float()
                    dweight_mul = 2 ** dweight_pow
                    #print(dweight_mul)
     #                # weight zero and gradient nonzero
                    dweight_add = (qweight == 0.0).float() * m.mask * dweight_sign * args.min_weight
     #                # print(torch.sum(dweight_add))
     # # update weight
                    new_weight = (qweight.data * dweight_mul + dweight_add + linear1_weight) * pre_mask
                    #new_weight[new_weight.abs() <= args.min_weight] = 0.0
                    #m.new_pre_mask = new_weight.sign().float()
                    #new_weight = new_weight * m.new_pre_mask
                    #print(new_weight)
                    if args.max_weight is not None:
                       new_weight.clamp_(-args.max_weight, args.max_weight)
                    m.weight.data = new_weight
     #                # check whether new_weight contains weights that less than given threshold
     #                # now = torch.sum(abs(new_weight) < args.threshold)
     #                # print(now-pre)
                    #print('pruning weights sum'.format(torch.sum(abs(now - pre))))
     #                # reset the activated counters to 0
                    m.dweight_counter[activated] = 0.0

        optimizer.step()

        # print('!!!!!!!!backward time: ', time.time()-start_time)
  #       pruning_ratio = get_pruning_ratio(model)
        #if args.quant_each_step:
        for name, m in model.named_modules():
            if hasattr(m, 'mask'):
                 with torch.no_grad():
  #                      #pass # do not need if fixed shift parameters
                    m.weight.data = m.sparsify_and_quantize_weight()
  #                  m.weight.data = m.weight.data * m.new_pre_mask
                         # reset mask??
  #                      if pruning_ratio < 50:
                    m.set_mask()
        for name, m in model.named_modules():
                 if not hasattr(m, 'mask'):
                    continue
                 if isinstance(m, Adder2D):
                     all_linear1_params =sum(p.abs().sum() for p in m.adder.data)
                     linear1_params = 0.001 * all_linear1_params
                     adder_sum = m.adder.data
                     adder_sum_mask = adder_sum.sign().float()
                     #mask1 = torch.sign(abs(adder_sum))
                     #if l1 is None:
                     #else:
                     #   l1 = l1+adder_sum.norm(1)
                     #print(linear1_params)
                     adder_data_pre = adder_sum - linear1_params
                     #mask = torch.sign(abs(adder_data_pre))
                     #adder_data_new = m.round_weight_each_step(adder_data_pre, m.weight_bits)
                     #m.adder.data = adder_data_new * mask * mask1
                     m.adder.data = (m.round_weight_each_step(adder_data_pre, m.weight_bits)) * adder_sum_mask
                     print('adder data:'.format(m.adder.data))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    Total_train_time = time.time() - start_time
    print('total training time for one epoch: {}'.format(Total_train_time))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = np.round(train_acc / len(train_loader), 2)
    history_score[epoch][3] = Total_train_time

def test(model):
    start_time = time.time()
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
        #data, target = Variable(data, volatile=True), Variable(target)
             output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    Total_test_time = time.time() - start_time
    print('total test time for one epoch: {}'.format(Total_test_time))
    history_score[epoch][4] = Total_test_time
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)


best_prec1 = 0.
swa_best_prec1 = 0.
swa_n = 0
swa_acc1 = 0.

if __name__ == '__main__':
  for epoch in range(args.start_epoch, args.epochs):
    if args.eval_only:
        with torch.no_grad():
            prec1 = test(model)
            print('Prec1: {}: '.format(prec1))
        exit()

    if args.lr_decay == 'stepwise':
        # step-wise LR schedule
        if epoch in args.schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
    elif args.lr_decay == 'cosine':
        schedule_cosine_lr_decay.step(epoch)
    elif args.lr_decay == 'cyclic_cosine':
        scheduler_cyclic_cosine_lr_decay.step(epoch)
    else:
        raise NotImplementedError
    train(epoch)
    prec1 = test(model)
    history_score[epoch][2] = prec1
    #history_score[epoch][3] = prec5
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    #best_prec5 = max(prec5, best_prec5)

    get_shift_range(model)
    get_adder_sparsity(model)

    # swa part
    if args.swa:
        if ((epoch + 1) >= args.swa_start and
                (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0):
            moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            bn_update(train_loader, swa_model)
            swa_acc1 = test(swa_model)
        #history_score[epoch][3] = swa_acc1

        is_swa_best = swa_acc1 > swa_best_prec1
        swa_best_prec1 = max(swa_acc1, swa_best_prec1)

 #       save_checkpoint_swa({
 #           'epoch': epoch + 1,
 #           'arch': args.arch,
 #           'state_dict': model.state_dict(),
 #           'swa_state_dict': swa_model.state_dict(),
 #           'swa_n': swa_n,
 #           'best_prec1': best_prec1,
 #           'optimizer': optimizer.state_dict(),
 #       }, is_best, is_swa_best, epoch, filepath=args.save)
 #   else:
 #       save_checkpoint({
 #           'epoch': epoch + 1,
 #           'arch': args.arch,
 #           'state_dict': model.state_dict(),
 #           'best_prec1': best_prec1,
 #           'optimizer': optimizer.state_dict(),
 #       }, is_best, epoch, filepath=args.save)

    np.savetxt(os.path.join(args.save, 'record_sparse0noswitch.txt'), history_score, fmt='%10.5f', delimiter=',')

print("Best accuracy: " + str(best_prec1))
#history_score[-1][2] = best_prec1
#history_score[-1][3] = best_prec5
np.savetxt(os.path.join(args.save, 'record_sparse0noswitch.txt'), history_score, fmt='%10.5f', delimiter=',')
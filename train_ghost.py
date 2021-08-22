import argparse
import os, time
import torch
import shutil
import numpy as np
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from pthflops import count_ops
import torch.optim as optim
from thop import profile
from thop import clever_format
from torch.autograd import Variable
import models
import optim
import se_shift.utils_optim
from optim import svrg
import torch.backends.cudnn as cudnn
from cyclicLR import CyclicCosAnnealingLR
from torchsummary import summary
from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type
import distutils.util
import matplotlib.pyplot as plt
import time
from PIL import Image
from pathlib import Path
from adder import Adder2D
from models import adder as adder_slow
from adder import adder as adder_fast
import deepshift
from models import resnet20_shiftadd_ghost
from models import mobilenet_shiftadd_ghost
from models import shufflenet_shiftadd
from models import shufflenet_shiftadd_ghost
from models import shufflenet_shiftadd_ghost_cpu
from models import mobilenetv2
from models import GhostNet_backbone
from models import ghostnet
from models import mobilenetv3_ghost
from models import resnet_backbone
from se_shift.utils_optim import SGD
from se_shift import SEConv2d, SELinear
from se_shift.utils_quantize import sparsify_and_nearestpow2
from se_shift.utils_swa import bn_update, moving_average
from collections import OrderedDict
import pytorch_model_summary as pms
import torchvision.models as tm
from torchvision.models import shufflenet_v2_x0_5
from torchvision.models import shufflenet_v2_x1_5
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import mobilenet
from models import mobilenet_v3
from models import mobilenet_v2_backbone
from models import vgg_shiftadd
from models import wideres
CUDA_VISIBLE_DEVICES = 0
import summary_model

# Training settings
parser = argparse.ArgumentParser(description='PyTorch AdderNet Trainning')
parser.add_argument('--data', type=str, default='/home/bella/Desktop/datasets/imagenet-mini', help='path to imagenet')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset')
parser.add_argument('--data_path', type=str, default='/home/bella/Desktop/ShiftAddNet', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=128, metavar='N', help='batch size for testing')
parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='restart point')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 120], help='learning rate schedule')
parser.add_argument('--lr', type=float, default=0.008, metavar='LR', help='learning rate')
parser.add_argument('--lr-sign', default=None, type=float, help='separate initial learning rate for sign params')
parser.add_argument('--lr_decay', default='cosine', type=str, choices=['stepwise', 'cosine', 'cyclic_cosine'])
parser.add_argument('--optimizer', type=str, default='svrg', help='used optimizer')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--save', default='./temp', type=str, metavar='PATH', help='path to save prune model')
parser.add_argument('--arch', default='mobilenet_shiftadd_ghost', type=str, help='architecture to use')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# shift hyper-parameters
parser.add_argument('--shift_depth', type=int, default=40, help='how many layers to convert to shift')
parser.add_argument('--shift_type', type=str, choices=['Q', 'PS'], default='PS', help='shift type for representing weights')
parser.add_argument('--rounding', default='deterministic', choices=['deterministic', 'stochastic'])
parser.add_argument('--weight_bits', type=int, default=5, help='number of bits to represent the shift weights')
parser.add_argument('--sign_threshold_ps', type=float, default=None, help='can be controled')
parser.add_argument('--use_kernel', type=lambda x: bool(distutils.util.strtobool(x)), default=False, help='whether using custom shift kernel')
# add hyper-parameters
parser.add_argument('--add_quant', type=bool, default=True, help='whether to quantize adder layer')
parser.add_argument('--add_bits', type=int, default=8, help='number of bits to represent the adder filters')
parser.add_argument('--add_sparsity', type=float, default=None, help='sparsity in adder filters')
parser.add_argument('--quantize_v', type=str, default='sbm', help='quantize version')
# shift hyper-parameters
parser.add_argument('--shift_quant_bits', type=int, default=16, help='quantization training for shift layer')
#parser.add_argument('--sign_threshold', type=float, default=0, help='Threshold for pruning.')
#parser.add_argument('--distributed', action='store_true', help='whether to use distributed training')
# distributed parallel
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--port", type=str, default="15000")
parser.add_argument('--distributed', action='store_true', help='whether to use distributed training')
# eval only
parser.add_argument('--eval_only', action='store_true', default=False, help='whether only evaluation')
parser.add_argument('--l1', action='store_true', default=True, help='whether sparse shift l1 norm')
# sparse
parser.add_argument('--threshold', type=float, default=1 * 1e-4, # (>= 2^-7)
                    help='Threshold in prune weight.')
parser.add_argument('--sign_threshold', type=float, default=0.1, help='Threshold for pruning.')
parser.add_argument('--dist', type=str, default='uniform', choices=['kaiming_normal', 'normal', 'uniform'])
parser.add_argument('--percent', default=0.0001, type=float, help='percentage of weight to prune')
parser.add_argument('--prune_method', default='magnitude', choices=['random', 'magnitude'])
parser.add_argument('--prune_layer', default='all', choices=['shift', 'add', 'all'])

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
#print(args.gpu_ids)
#if len(args.gpu_ids) > 0:
#    torch.cuda.set_device(args.gpu_ids[0])

if args.distributed:
    os.environ['MASTER_PORT'] = args.port
    torch.distributed.init_process_group(backend="nccl")

kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(size=32, padding=4),
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
    DATA = Path("/home/bella/Desktop/datasets/imagenet-mini")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=64),
        transforms.CenterCrop(size=(64, 64)),
#        transforms.RandomResizedCrop(64),
#        transforms.RandomHorizontalFlip(),
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
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
        pin_memory=True, num_workers=6)
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')

    #train_dataset = datasets.ImageFolder(
    #    traindir,
    #    transforms.Compose([
    #        transforms.RandomResizedCrop(64),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        normalize,
    #    ]))

    #test_loader = torch.utils.data.DataLoader(
    #    datasets.ImageFolder(valdir, transforms.Compose([
    #        transforms.Resize(64),
    #        transforms.CenterCrop(64),
    #        transforms.ToTensor(),
    #        normalize,
    #    ])),
    #   batch_size=args.test_batch_size, shuffle=False,
    #   num_workers=16, pin_memory=True)

model1 = models.mobilenet_v2_backbone.mobilenet_v2(num_classes=10)
#model1 = shufflenet_v2_x0_5(num_classes=10)
#model1 = models.GhostNet_backbone.ghostnet()
#model1 = mobilenet_v3.mobilenet_v3_small(num_classes=10)
#model1 = models.resnet_backbone.resnet18(num_classes=10)
#model1 =  models.wideres.Wide_ResNet(28, 5, 0.3, 10)
input = torch.randn(1, 3, 256, 256)
macs, params = profile(model1, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print('macs', macs)
print('param', params)
pms.summary(model1, torch.zeros((1, 3, 32, 32)), batch_size=128, show_hierarchical=False, print_summary=True)

if args.dataset == 'imagenet':
    num_classes = 1000
    #model = model1
    #model = models.mobilenetv3_ghost.mobilenet_v3_small(num_classes=num_classes)
    model = models.mobilenetv2.mobilenet_v2(num_classes=num_classes)
    #model = models.__dict__['resnet50'](num_classes=1000, quantize=args.add_quant, weight_bits=args.add_bits)
elif args.dataset == 'cifar10':
    num_classes = 10
#    model = models.shufflenet_shiftadd.ghostnet(num_classes=num_classes, pretrained=False)
    #model = shufflenet_v2_x1_0()
    #model = mobilenet_v2()
    #model = models.ghostnet.ghostnet()
    #model = model1
    #model = models.wideres.Wide_ResNet(28, 5, 0.3, 10)
    #model = models.vgg_shiftadd.vgg16_nd_ss()
    model = models.mobilenetv3_ghost.mobilenet_v3_small(num_classes=num_classes)
    #model = models.mobilenetv2.mobilenet_v2(num_classes=10)
    #model = models.shufflenet_shiftadd_ghost.ghostnet(num_classes=num_classes, pretrained=False)
    #model = models.mobilenet_shiftadd_ghost.ghostnet(num_classes=num_classes, kernel_size=3, quantize=args.add_quant, weight_bits=args.add_bits, quantize_v=args.quantize_v)
    #model = models.resnet20_shiftadd_ghost.resnet20_shift(num_classes=num_classes, quantize=args.add_quant, weight_bits=args.add_bits, quantize_v=args.quantize_v)
    #model = models.__dict__[args.arch](num_classes=num_classes, quantize=args.add_quant, weight_bits=args.add_bits, quantize_v=args.quantize_v)
#   model = models.__dict__[args.arch](threshold = args.threshold, sign_threshold = args.sign_threshold, distribution = args.dist, num_classes=10, quantize=args.add_quant, weight_bits=args.add_bits)
elif args.dataset == 'cifar100':
    num_classes = 100
    model = models.__dict__[args.arch](num_classes=100, quantize=args.add_quant, weight_bits=args.add_bits, quantize_v=args.quantize_v)
elif args.dataset == 'mnist':
    model = models.__dict__[args.arch](num_classes=10, quantize=args.add_quant, weight_bits=args.add_bits)
else:
    raise NotImplementedError('No such dataset!')

#print(model)
#M = []
#N = []
#K = []
#S = []
#C = []
#size = 32
#for m in model.modules():
#    if isinstance(m, nn.Conv2d):
#        M.append(m.weight.shape[0])
#        N.append(m.weight.shape[1])
#        K.append(m.weight.shape[2])
#        S.append(m.stride[0])
#        C.append(int(size))
#        if S[-1] == 2:
#            size /= 2
#print('M', M)
#print('N', N)
#print('K', K)
#print('S', S)
#print('C', C)
#print(len(M))
#for i in range(len(M)):
#     print('const int M{} = {}, N{} = {}, K{} = {}, S{} = {}, C{} = {};'.format(
#            i, M[i], i, N[i], i, K[i], i, S[i], i, C[i]))
#     print('const int H{} = C{} - S{} + K{};'.format(i, i, i, i))
#exit()
#best_prec1 = None
#shift_depth = []
#if best_prec1 is None: # no pretrain
#   if 'shift' in args.arch:
#       model, conversion_count = convert_to_shift(model, args.shift_depth, args.shift_type, convert_weights=False, use_kernel=args.use_kernel, rounding=args.rounding,
#        weight_bits=args.weight_bits, sign_threshold_ps=args.sign_threshold_ps, quant_bits=args.shift_quant_bits)
#else:
#   if 'shift' in args.arch:
#        model, conversion_count = convert_to_shift(model, shift_depth, args.shift_type, convert_weights=False,
#                                                   use_kernel=args.use_kernel, rounding=args.rounding,
#                                                   weight_bits=args.weight_bits,
#                                                   sign_threshold_ps=args.sign_threshold_ps,
#                                                   quant_bits=args.shift_quant_bits)


#if args.cuda:
model.cuda()
if len(args.gpu_ids) > 1:
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu_ids)
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = se_shift.utils_optim.SGD(params_dict, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adadelta"):
    optimizer = torch.optim.Adadelta(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "svrg"):
    optimizer = svrg.SVRG(model.parameters(), lr = 0.1, freq = 3, momentum=args.momentum, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adagrad"):
    optimizer = torch.optim.Adagrad(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "adam"):
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "rmsprop"):
    optimizer = torch.optim.RMSprop(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "radam"):
    optimizer = optim.RAdam(params_dict, args.lr, weight_decay=args.weight_decay)
elif (args.optimizer.lower() == "ranger"):
    optimizer = optim.Ranger(params_dict, args.lr, weight_decay=args.weight_decay)
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
            shutil.copyfile(filename, os.path.join(filepath, 'CIFAR10_ghost_Back0_5_svrg.pth.tar'))


def load_add_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'weight' in k and not 'bn' in k and not 'fc' in k:
            if k == 'conv1.weight' or 'downsample.1' in k:
                new_state_dict[k] = v
                continue
            k = k[:-6] + 'adder'
        # print(k)
        new_state_dict[k] = v
    return new_state_dict

def load_shiftadd_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'weight' in k and not 'bn' in k and not 'fc' in k:
            if k == 'conv1.weight' or 'downsample.2' in k:
                new_state_dict[k] = v
                continue
            k = k[:-6] + 'adder'
        # print(k)
        new_state_dict[k] = v
    return new_state_dict


if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        try:
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model.load_state_dict(load_add_state_dict(checkpoint['state_dict']))
        except:
            model.load_state_dict(load_shiftadd_state_dict(checkpoint['state_dict']))
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    save_checkpoint({'state_dict': model.state_dict()}, False, epoch='init', filepath=args.save)

#inp = torch.rand(1,3,224,224).to(device)
#count_ops(model, inp)
#print(count_ops())
#exit()
#    print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
#except:
#    print("WARNING: Unable to obtain summary of model")

# save name
# name model sub-directory "shift_all" if all layers are converted to shift layers
conv2d_layers_count = count_layer_type(model, nn.Conv2d) #+ count_layer_type(model, unoptimized.UnoptimizedConv2d)
linear_layers_count = count_layer_type(model, nn.Linear) #+ count_layer_type(model, unoptimized.UnoptimizedLinear)
#print(conv2d_layers_count)

if (args.shift_depth > 0):
    if (args.shift_type == 'Q'):
        shift_label = "shift_q"
    else:
        shift_label = "shift_ps"
else:
    shift_label = "shift"

# if (conv2d_layers_count==0 and linear_layers_count==0):
if conv2d_layers_count == 0:
    shift_label += "_all"
else:
    shift_label += "_%s" % (args.shift_depth)

if (args.shift_depth > 0):
    shift_label += "_wb_%s" % (args.weight_bits)

if args.add_quant:
    shift_label += '_add-{}'.format(args.add_bits)

if args.sign_threshold_ps:
    shift_label += '_ps_thre-{}'.format(args.sign_threshold_ps)

args.save = os.path.join(args.save, shift_label)
if not os.path.exists(args.save):
    os.makedirs(args.save)

history_score = np.zeros((args.epochs, 7))
#history_score1 = np.zeros((args.epochs, 1))
def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    # plt.xlim(xmin=-8,xmax=8)
    # plt.ylim(ymin=-8,ymax=8)
    # plt.text(-7.8,7.3,"epoch=%d" % epoch)
    plt.title("epoch=%d" % epoch)
    vis_dir = os.path.join(args.save, 'visualization')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    plt.savefig(vis_dir+'/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)


def accuracy(output, target, topk=(1, 5)):
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

#if 'shift' in args.arch:
if args.shift_type == 'Q':
        shift_module = deepshift.modules_q.Conv2dShiftQ
elif args.shift_type == 'PS':
        shift_module = deepshift.modules.Conv2dShift
else:
        raise NotImplementedError


global total_add
total_add = 0
for m in model.modules():
    if isinstance(m, Adder2D):
        total_add += m.adder.data.numel()

global total
total = 0
def get_shift_range(model):
    if 'shift' in args.arch:
        # pruning
        if args.shift_type == 'Q':
            total = 0
            for m in model.modules():
                if isinstance(m, shift_module):
                    total += m.weight.data.numel()
            shift_weights = torch.zeros(total)
            index = 0
            for m in model.modules():
                if isinstance(m, shift_module):
                    size = m.weight.data.numel()
                    shift_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
                    #print(shift_weights)
                    index += size

            y, i = torch.sort(shift_weights)
            thre_index = int(total * percent)
            thre = y[thre_index] - 1e-7
            weight_unique = torch.unique(shift_weights)
            #print(weight_unique)
            print('shift_range:', weight_unique.size()[0]-1)
        elif args.shift_type == 'PS':
            total = 0
            for m in model.modules():
                if isinstance(m, shift_module):
                    total += m.sign.data.numel()
            sign_weights = torch.zeros(total)
            shift_weights = torch.zeros(total)
            index = 0
            for m in model.modules():
                if isinstance(m, shift_module):
                    size = m.sign.data.numel()
                    sign_weights[index:(index+size)] = m.sign.data.view(-1).abs().clone()
                    shift_weights[index:(index+size)] = m.shift.data.view(-1).abs().clone()
                    index += size

            y, i = torch.sort(shift_weights)
            print('y is:', len(y))
            print('i is:', len(i))
            shift_unique = torch.unique(shift_weights)
            print('shift range:', shift_unique.size()[0]-1)
            left_shift_weight = int(torch.sum(shift_weights != 0))
            left_shift_mask = int(torch.sum(sign_weights != 0))
#            print('pruning ratio:', (1 - left_shift_mask / float(total)) * 100, '%')
            print('left mask:', left_shift_mask)
            print('left weights:', left_shift_weight)
            print('total shift:', total)
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
                adder_masks[index:(index+size)] = m.adder.data.view(-1).abs().clone()
                index += size
        left_adder_mask = int(torch.sum(adder_masks != 0))
        print('left adder mask', left_adder_mask)
#        print('Add sparsity ratio:', (1 - left_adder_mask / float(total_add)) * 100, '%')
        print('total adders:', total_add)
        history_score[epoch][6] = left_adder_mask

from deepshift import utils
def build(self):
     for name, self in model.named_modules():
          if isinstance(self, deepshift.modules.Conv2dShift):
            self.shift_grad = torch.zeros_like(self.shift.data)
            self.shift_mean_grad = torch.zeros_like(self.shift.data)
            self.sign_grad = torch.zeros_like(self.shift.data)
            self.sign_mean_grad = torch.zeros_like(self.shift.data)
            self.shift_sum = torch.zeros_like(self.shift.data)
            self.shift_mask = torch.zeros_like(self.shift.data)
          if isinstance(self, Adder2D):
            self.adder_grad = torch.zeros_like(self.adder.data)
            self.adder_mean_grad = torch.zeros_like(self.adder.data)
            self.adder_mask = torch.zeros_like(self.adder.data)
            self.adder_sum = torch.zeros_like(self.adder.data)

def create_mask(shape, rate):
    mask = torch.cuda.FloatTensor(shape).uniform_() > rate
    return mask + 0

def train(epoch):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    pruned = 0.
    end_time = time.time()
    feat_loader = []
    idx_loader = []
#    batch_time = time.time()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
       # print('total time for one batch: {}'.format(time.time()-batch_time))
       # batch_time = time.time()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #if args.cuda:
        data, target = data.cuda(), target.cuda()
       # with torch.no_grad():
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()
        loss.backward()
        #for name, m in model.named_modules():
        #  if isinstance(m, deepshift.modules.Conv2dShift):
        #      print(m.shift.data)
        #    sign = m.sign.data
        #    sign[sign < -0.2] = -1
        #    sign[sign > 0.2] = 1
        #    sign[(-0.2 <= sign) & (sign <= 0.2)] = 0
        optimizer.step()
        #for name, m in model.named_modules():
        #  if isinstance(m, deepshift.modules.Conv2dShift):
            # m.shift.data = m.shift.data - 0.0008**(epoch+1)*torch.norm(m.shift.data).float().cuda()
            # m.shift.data[m.shift.data.abs() <= m.shift_mask.abs()] = 0.0
            #m.shift_mask = m.shift.data.min().cuda()
        #    weight_copy = m.shift.data.abs().clone()
        #    mask = weight_copy.gt(0).float().cuda()
        #    m.shift.grad.data.mul_(mask)
        #    m.sign.grad.data.mul_(mask)
        #    m.shift.data.mul_(mask)
        #    m.sign.data.mul_(mask)
            #if epoch == [0, args.epochs, 3]:
            #    m.shift_grad = m.shift.grad.data
            #    m.shift_mean_grad[m.shift_grad != m.shift_grad.mean().cuda()] = m.shift_grad.mean().cuda()
        #  if isinstance(m, Adder2D):
            #m.mask = m.adder.data.min().cuda()
        #    adder_copy = m.adder.data.abs().clone()
        #    mask = adder_copy.gt(0).float().cuda()
        #    m.adder.grad.data.mul_(mask)
        #    m.adder.data.mul_(mask)
            #if epoch == [0, args.epochs, 3]:
            #    m.adder_grad = m.adder.grad.data
            #    m.adder_mean_grad[m.adder_grad != m.adder_grad.mean().cuda()] = m.adder_grad.mean().cuda()
        #torch.cuda.synchronize()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    Total_train_time = time.time() - start_time
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = np.round(train_acc / len(train_loader), 2)
    history_score[epoch][3] = Total_train_time
    print('total training time for one epoch: {}'.format(Total_train_time))


def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    test_acc_5 = 0
    Total_time = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = data.cpu(), target.cpu()
        with torch.no_grad():
        #data, target = Variable(data, volatile=True), Variable(target)
            t0 = time.time()
            output = model(data)
            if args.cuda:
               torch.cuda.current_stream().synchronize()
            t1 = time.time()
            Time_Per = t1 - t0
            Total_time += Time_Per
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()
        test_acc_5 += prec5.item()
        #print('total test time for one epoch: {}'.format(Time_Per))

    test_loss /= len(test_loader.dataset)
    history_score[epoch][4] = test_loss
    print('\nTest set: Average loss: {:.4f}, Prec1: {}/{} ({:.2f}%), Prec5: ({:.2f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader), test_acc_5 / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2), np.round(test_acc_5 / len(test_loader), 2)


best_prec1 = 0.
best_prec5 = 0.
best_prec1_p = 0.
best_prec1_p1 = 0.
percent = 0.1
percent_add = 0.05
if __name__ == '__main__':
  for epoch in range(args.start_epoch, args.epochs):
  #for epoch in range(100):
    if args.eval_only:
        with torch.no_grad():
            prec1, prec5 = test()
            print('Prec1: {}; Prec5: {}'.format(prec1, prec5))
    if args.lr_decay == 'stepwise':
        # step-wise LR schedule
        if epoch in args.schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.08
    elif args.lr_decay == 'cosine':
        schedule_cosine_lr_decay.step(epoch)
    elif args.lr_decay == 'cyclic_cosine':
        scheduler_cyclic_cosine_lr_decay.step(epoch)
    else:
        raise NotImplementedError
    shift_depth = 60 #44 is expect conv head
    #if 'shift' in args.arch:  # no pretrain
    #if best_prec1 <= best_prec1_p <= 65 and epoch in range(0, args.epochs, 2):
    model.cuda()
    if epoch == 0:
  #      for m in model.modules():
  #          if isinstance(m, nn.Conv2d):
 #               if m.kernel_size!=(1,1):
         model, conversion_count = convert_to_shift(model, shift_depth, args.shift_type, convert_weights=True,
                                             use_kernel=False, use_cuda=True, rounding=args.rounding,
                                             weight_bits=args.weight_bits, sign_threshold=args.sign_threshold_ps,
                                             quant_bits=args.shift_quant_bits)

         input = torch.zeros(1, 3, 32, 32).cuda()
         macs, params = profile(model, inputs=(input,))
         macs, params = clever_format([macs, params], "%.3f")
         print('macs', macs)
         print('param', params)
         print(summary_model.summary(model, input_size=(3, 32, 32), batch_size=1, device=torch.device('cuda'), dtypes=None))



    print('shift_depth', shift_depth)
    best_prec1_p = best_prec1
    #build(model)
    ## pruning
    if epoch !=10000:
     if 'shift' in args.arch and args.prune_layer != 'add':
        print('prune for shift layer:')
        if args.shift_type == 'Q':
            shift_module = deepshift.modules_q.Conv2dShiftQ
        elif args.shift_type == 'PS':
            shift_module = deepshift.modules.Conv2dShift
        else:
            raise NotImplementedError
        # pruning
        if args.shift_type == 'Q':
            total = 0
            for m in model.modules():
                if isinstance(m, shift_module):
                    total += m.weight.data.numel()
            shift_weights = torch.zeros(total)
            index = 0
            for m in model.modules():
                if isinstance(m, shift_module):
                    size = m.weight.data.numel()
                    shift_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
                    index += size

            y, i = torch.sort(shift_weights)
            thre_index = int(total * percent)
            thre = y[thre_index] - 1e-7
            pruned = 0
            print('Pruning threshold: {}'.format(thre))
            zero_flag = False
            # ----------------------------------------------------------------
            if args.prune_method == 'magnitude':
                for k, m in enumerate(model.modules()):
                    if isinstance(m, shift_module):
                        shift_copy = m.weight.data.abs().clone()
                        # prune at boundary (weight == thre)
                        _mask = torch.eq(shift_copy, thre + 1e-7).float().cuda()
                        _mask = _mask * torch.cuda.FloatTensor(shift_copy.shape).uniform_(-percent,
                                                                                          1 - percent)
                        shift_copy += _mask
                        # ---------------------------------
                        mask = shift_copy.gt(thre).float().cuda()
                        pruned = pruned + mask.numel() - torch.sum(mask)
                        m.weight.data = m.weight.data.mul_(mask) + 1 - mask  # no shift
                        if int(torch.sum(mask)) == 0:
                            zero_flag = True
                        #print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                        #      format(k, mask.numel(), int(torch.sum(mask))))
            elif args.prune_method == 'random':
                for k, m in enumerate(model.modules()):
                    if isinstance(m, shift_module):
                        shift_copy = m.weight.data.abs().clone()
                        mask = create_mask(shift_copy.shape, percent)
                        pruned = pruned + mask.numel() - torch.sum(mask)
                        m.weight.data = m.weight.data.mul_(mask) + 1 - mask  # no shift
                        if int(torch.sum(mask)) == 0:
                            zero_flag = True
                        #print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                        #      format(k, mask.numel(), int(torch.sum(mask))))
            else:
                raise NotImplementedError
            # ----------------------------------------------------------------
        elif args.shift_type == 'PS':
            total = 0
            for m in model.modules():
                if isinstance(m, shift_module):
                    total += m.shift.data.numel()
            shift_weights = torch.zeros(total)
            index = 0
            if epoch >= 1:
               for m in model.modules():
                   if isinstance(m, shift_module):
                      size = m.shift.data.numel()
                      shift_weights[index:(index + size)] = m.shift.data.view(-1).abs().clone()
                      index += size

               y, i = torch.sort(shift_weights)
               thre_index = int(total * percent)
               thre = y[thre_index]
               pruned = 0
               print('Pruning threshold: {}'.format(thre))
               zero_flag = False
            # ----------------------------------------------------------------
               if args.prune_method == 'magnitude':
                 for k, m in enumerate(model.modules()):
                    if isinstance(m, shift_module):
                        shift_copy = m.shift.data.abs().clone()
                        mask = shift_copy.gt(thre).float().cuda()
                        pruned = pruned + mask.numel() - torch.sum(mask)
                        m.shift.data = m.shift.data.mul_(mask)
                        m.sign.data = m.sign.data.mul_(mask)
                        if int(torch.sum(mask)) == 0:
                            zero_flag = True
                        #print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                        #      format(k, mask.numel(), int(torch.sum(mask))))
               elif args.prune_method == 'random':
                 for k, m in enumerate(model.modules()):
                    if isinstance(m, shift_module):
                        shift_copy = m.shift.data.abs().clone()
                        mask = create_mask(shift_copy.shape, percent)
                        pruned = pruned + mask.numel() - torch.sum(mask)
                        m.shift.data.mul_(mask)
                        m.sign.data.mul_(mask)
                        if int(torch.sum(mask)) == 0:
                            zero_flag = True
                        #print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                        #      format(k, mask.numel(), int(torch.sum(mask))))
               else:
                 raise NotImplementedError
            # ----------------------------------------------------------------
        #print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned,
        #                                                                               float(pruned) / total))
     if 'add' in args.arch and args.prune_layer != 'shift':
        print('prune for adder layer:')
        # adder_module = adder_slow.adder2d
        adder_module = adder_fast.Adder2D
        total = 0
        for m in model.modules():
            if isinstance(m, adder_module):
                total += m.adder.data.numel()
        adder_weights = torch.zeros(total)
        index = 0
        if epoch >= 1:
          for m in model.modules():
            if isinstance(m, adder_module):
                size = m.adder.data.numel()
                adder_weights[index:(index + size)] = m.adder.data.view(-1).abs().clone()
                index += size

          y, i = torch.sort(adder_weights)
          thre_index = int(total * percent_add)
          thre = y[thre_index]
          pruned = 0
          print('Pruning threshold: {}'.format(thre))
          zero_flag = False
        # ----------------------------------------------------------------
          if args.prune_method == 'magnitude':
            for k, m in enumerate(model.modules()):
                if isinstance(m, adder_module):
                    adder_copy = m.adder.data.abs().clone()
                    mask = adder_copy.gt(thre).float().cuda()
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    m.adder.data = m.adder.data.mul_(mask)
                    if int(torch.sum(mask)) == 0:
                        zero_flag = True
                    #print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                    #      format(k, mask.numel(), int(torch.sum(mask))))
          elif args.prune_method == 'random':
            for k, m in enumerate(model.modules()):
                if isinstance(m, adder_module):
                    shift_copy = m.adder.data.abs().clone()
                    mask = create_mask(shift_copy.shape, percent_add)
                    pruned = pruned + mask.numel() - torch.sum(mask)
                    m.adder.data.mul_(mask)
                    if int(torch.sum(mask)) == 0:
                        zero_flag = True
                    #print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                    #      format(k, mask.numel(), int(torch.sum(mask))))
          else:
            raise NotImplementedError
   # percent = percent + args.percent
    train(epoch)
    if best_prec1 >= best_prec1_p1 and epoch in range(4, args.epochs, 3):
        percent_add = percent_add + args.percent
        percent = percent + args.percent
    else:
        percent_add = percent_add
        percent = percent
  #  else:
  #      percent_add = percent_add
  #      percent = percent
    #device = 'cuda:0'
    #inp = torch.rand(1, 3, 64, 64).to(device)
    #ops = count_ops(model, inp)[0]
    #print(ops)
    best_prec1_p1 = best_prec1
    with torch.no_grad():
 #       model.cpu()
        prec1, prec5 = test()
    history_score[epoch][2] = prec1
    # history_score[epoch][3] = prec5
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    best_prec5 = max(prec5, best_prec5)
    model_rounded = round_shift_weights(model, clone=False)
    get_shift_range(model_rounded)
    get_adder_sparsity(model_rounded)
#    print('how many conv2d conver to shift', conversion_count)
    np.savetxt(os.path.join(args.save, 'CIFAR10_ghost_Back0_5_svrg.txt'), history_score, fmt='%10.5f', delimiter=',')

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),}, is_best, epoch, filepath=args.save)

    # break
print("Best accuracy: " + str(best_prec1))
history_score[-1][2] = best_prec1
#history_score[-1][3] = best_prec5
np.savetxt(os.path.join(args.save, 'CIFAR10_ghost_Back0_5_svrg.txt'), history_score, fmt='%10.5f', delimiter=',')

# print model summary
#model_summary = None
#try:
#device = 'cuda:0'
#model = model.to(device)
#model_summary, model_params_info = torchsummary.summary_string(model, input_size=(3,32,32))
#input=(3,224,224)
#summary(model,input_size=input,batch_size=64,device='cuda',dtypes=float)
#print(model_summary)
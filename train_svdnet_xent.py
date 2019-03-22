from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.losses import CrossEntropyLoss, DeepSupervision
from torchreid.utils.iotools import check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.utils.generaltools import set_random_seed
from torchreid.eval_metrics import evaluate
from torchreid.optimizers import init_optimizer
from torchreid.lr_schedulers import init_lr_scheduler


os.environ['TORCH_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.torch'))

testloader_dict = trainloader = criterion = None
use_gpu = False

# global variables
parser = argument_parser()
args = parser.parse_args()


def corr_metric(W: 'K x N'):

    G = W.permute(1, 0) @ W
    return torch.trace(G) / abs(G).sum()


def replace_weight(layer):

    with torch.no_grad():
        # NECESSARY! The weight of Linear layer has been transposed!
        A = layer.weight.t()
        M, N = A.size()
        M: 2048
        N: 1024
        U, S, V = torch.svd(A, some=False)
        W = A @ V
        W: '2048 x 1024 = M x N'

        NW = torch.zeros_like(A)

        for i in range(N):

            curr_N = W.size(1)

            W_norm = torch.norm(W, p=2, dim=0)
            W_norm: 'curr_N'

            index = i
            vec_i = A[:, i]
            vec_i_norm = torch.norm(vec_i)

            co = (A[:, i].view(M, 1).t() @ W).view(curr_N)
            co: 'curr_N'
            co = co / vec_i_norm
            absco = abs(co / W_norm)
            maxco_index = torch.max(absco, 0)[1].item()

            NW[:, index] = W[:, maxco_index] * torch.sign(co[maxco_index])

            # Remove selected column vector from W
            W = W[:, sorted({x for x in range(curr_N) if x != maxco_index})]

        layer.weight.copy_(NW.t())
        print(layer.weight)

    return layer


def main():
    global args, criterion, testloader_dict, trainloader, use_gpu

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'test.log' if args.evaluate else 'train.log'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent'}, pretrained=not args.no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    criterion = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=None)
        resumed = True
    else:
        resumed = False

    if args.evaluate:
        print('Evaluate only')

        for name in args.target_names:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

    time_start = time.time()
    # ranklogger = RankLogger(args.source_names, args.target_names)
    print('=> Start training')

    if not resumed:
        train_base(model)
    train_RRI(model, 7)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    # ranklogger.show_summary()


def train(epoch, model, criterion, optimizer, trainloader, use_gpu, fixbase=False):
    losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    # if fixbase or args.always_fixbase:
    #     open_specified_layers(model, args.open_layers)
    # else:
    #     open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs = model(imgs)
        loss = sum(criterion(x, pids) for x in outputs) / len(outputs)
        # if isinstance(outputs, (tuple, list)):
        #     loss = DeepSupervision(criterion, outputs, pids)
        # else:
        #     loss = criterion(outputs, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                      epoch + 1, batch_idx + 1, len(trainloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accs
                  ))

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print('Computing CMC and mAP')
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    if return_distmat:
        return distmat
    return cmc[0]


def get_base_optimizer(model):

    kwargs = {
        'weight_decay': 5e-4,
        'lr': 0.0003,
        'betas': (0.9, 0.999),
    }
    param_groups = model.parameters()

    optimizer = torch.optim.Adam(param_groups, **kwargs)
    scheduler = init_lr_scheduler(optimizer, stepsize=[20, 40], gamma=0.1)

    return optimizer, scheduler


def get_RRI_optimizer(
    model,
    lr
):

    kwargs = {
        'weight_decay': 5e-4,
        'lr': lr,
        'momentum': 0.9,
    }
    param_groups = model.parameters()

    optimizer = torch.optim.SGD(param_groups, **kwargs)
    scheduler = init_lr_scheduler(optimizer, stepsize=[12], gamma=0.1)

    return optimizer, scheduler


def train_R(model, lr, T, fix_eigen_layer: bool=False):

    eigen_layers = model.module.get_fcs()

    if fix_eigen_layer:
        for eigen_layer in eigen_layers:
            eigen_layer.eval()
            for p in eigen_layer.parameters():
                p.requires_grad = False

        stage_name = 'restraint'
    else:
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        stage_name = 'relaxation'

    prefix = '{}_{}_'.format(T, stage_name)

    optimizer, scheduler = get_RRI_optimizer(model, lr)

    for epoch in range(20):
        train(epoch, model, criterion, optimizer, trainloader, use_gpu=use_gpu)

        scheduler.step()

        print('=> Test')

        if (epoch + 1) % args.eval_freq == 0:
            for name in args.target_names:
                print('Evaluating {} ...'.format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1 = test(model, queryloader, galleryloader, use_gpu)

    save_checkpoint({
        'state_dict': model.state_dict(),
        'rank1': rank1,
        'epoch': 0,
        'arch': args.arch,
        'optimizer': (),
    }, args.save_dir, prefix=prefix)


def train_base(model):

    optimizer, scheduler = get_base_optimizer(model)

    model.train()
    print('=== train base ===')

    open_layers = ['fc', 'classifier1', 'classifier2_1', 'classifier2_2', 'fc2_1', 'fc2_2', 'reduction', 'classifier']

    print('Train {} for {} epochs while keeping other layers frozen'.format(open_layers, 10))

    for epoch in range(10):

        open_specified_layers(model, open_layers)
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)

    print('Done. All layers are open to train for {} epochs'.format(60))
    open_all_layers(model)

    optimizer, scheduler = get_base_optimizer(model)

    for epoch in range(60):
        train(epoch, model, criterion, optimizer, trainloader, use_gpu=use_gpu)

        print('=> Test')

        if (epoch + 1) % args.eval_freq == 0:

            for name in args.target_names:
                print('Evaluating {} ...'.format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1 = test(model, queryloader, galleryloader, use_gpu)

    save_checkpoint({
        'state_dict': model.state_dict(),
        'rank1': rank1,
        'epoch': 0,
        'arch': args.arch,
        'optimizer': optimizer.state_dict(),
    }, args.save_dir, prefix='base_')


def train_RRI(model, Ts: int=7):

    base_lrs = [0.001] * 3 + [0.0001] * 10

    for T in range(Ts):
        print('=== T = {} ==='.format(T))
        print('Replacing eigen layer weight...')
        for eigen_layer in model.module.get_fcs():
            replace_weight(eigen_layer)
        print('Replaced.')
        print('--- Restraint ({}) ---'.format(T))
        train_R(model, base_lrs[T], T, fix_eigen_layer=True)
        print('--- Relaxation ({}) ---'.format(T))
        train_R(model, base_lrs[T], T, fix_eigen_layer=False)

    for name in args.target_names:
        print('Evaluating {} ...'.format(name))
        queryloader = testloader_dict[name]['query']
        galleryloader = testloader_dict[name]['gallery']
        rank1 = test(model, queryloader, galleryloader, use_gpu)

    save_checkpoint({
        'state_dict': model.state_dict(),
        'rank1': rank1,
        'epoch': 0,
        'arch': args.arch,
        'optimizer': (),
    }, args.save_dir, prefix='final_')


if __name__ == '__main__':
    main()

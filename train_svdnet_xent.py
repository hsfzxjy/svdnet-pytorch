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


testloader_dict = trainloader = criterion = None
use_gpu = False

# global variables
parser = argument_parser()
args = parser.parse_args()


def corr_metric(W: 'K x N'):

    G = W.permute(1, 0) @ W
    return torch.trace(G) / abs(G).sum()


def replace_weight(layer):

    _, _, V = torch.svd(layer.weight, some=False)
    with torch.no_grad():
        layer.weight.copy_(layer.weight.clone() @ V.clone())

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
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)
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
        'lr': 0.001,
        'momentum': 0.9,
    }
    param_groups = model.parameters()

    optimizer = torch.optim.SGD(param_groups, **kwargs)
    scheduler = init_lr_scheduler(optimizer, stepsize=[25, 50], gamma=0.1)

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

    if fix_eigen_layer:
        eigen_layer = model.module.fc
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
    for epoch in range(61):
        train(epoch, model, criterion, optimizer, trainloader, use_gpu=use_gpu)

        print('=> Test')

        for name in args.target_names:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            rank1 = test(model, queryloader, galleryloader, use_gpu)  # noqa

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
        replace_weight(model.module.fc)
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

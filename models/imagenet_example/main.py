import os
import shutil
import argparse
import random
import re
import time
import yaml
import json
import socket
import logging
from addict import Dict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.backends import cudnn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import models
from utils.dataloader import build_dataloader
from utils.misc import accuracy, check_keys, AverageMeter, ProgressMeter
from utils.loss import LabelSmoothLoss

parser = argparse.ArgumentParser(description='ImageNet Training Example')
parser.add_argument('--config',
                    default='configs/resnet50.yaml',
                    type=str,
                    help='path to config file')
parser.add_argument('--test',
                    dest='test',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--port',
                    default=12345,
                    type=int,
                    metavar='P',
                    help='master port')
parser.add_argument('--output',
                    dest='output',
                    default='inception_result.json',
                    help='output json file to hold perf results')

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()
logger_all = logging.getLogger('all')


def main():
    args = parser.parse_args()
    args.config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(args.config)

    args.rank = int(os.environ['SLURM_PROCID'])
    args.world_size = int(os.environ['SLURM_NTASKS'])
    args.local_rank = int(os.environ['SLURM_LOCALID'])

    node_list = str(os.environ['SLURM_NODELIST'])
    node_parts = re.findall('[0-9]+', node_list)
    os.environ[
        'MASTER_ADDR'] = f'{node_parts[1]}.{node_parts[2]}.{node_parts[3]}.{node_parts[4]}'
    os.environ['MASTER_PORT'] = str(args.port)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)

    if args.rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)

    logger_all.info("rank {} of {} jobs, in {}".format(args.rank,
                                                       args.world_size,
                                                       socket.gethostname()))

    dist.barrier()

    logger.info("config\n{}".format(
        json.dumps(cfgs, indent=2, ensure_ascii=False)))

    if cfgs.get('seed', None):
        random.seed(cfgs.seed)
        torch.manual_seed(cfgs.seed)
        torch.cuda.manual_seed(cfgs.seed)
        cudnn.deterministic = True

    model = models.__dict__[cfgs.net.arch](**cfgs.net.kwargs)
    model.cuda()

    logger.info("creating model '{}'".format(cfgs.net.arch))

    model = DDP(model, device_ids=[args.local_rank])
    logger.info("model\n{}".format(model))

    if cfgs.get('label_smooth', None):
        criterion = LabelSmoothLoss(cfgs.trainer.label_smooth,
                                    cfgs.net.kwargs.num_classes).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    logger.info("loss\n{}".format(criterion))

    optimizer = torch.optim.SGD(model.parameters(),
                                **cfgs.trainer.optimizer.kwargs)
    logger.info("optimizer\n{}".format(optimizer))

    cudnn.benchmark = True

    args.start_epoch = -cfgs.trainer.lr_scheduler.get('warmup_epochs', 0)
    args.max_epoch = cfgs.trainer.max_epoch
    args.test_freq = cfgs.trainer.test_freq
    args.log_freq = cfgs.trainer.log_freq

    best_acc1 = 0.0
    if cfgs.saver.resume_model:
        assert os.path.isfile(
            cfgs.saver.resume_model), 'Not found resume model: {}'.format(
                cfgs.saver.resume_model)
        checkpoint = torch.load(cfgs.saver.resume_model)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("resume training from '{}' at epoch {}".format(
            cfgs.saver.resume_model, checkpoint['epoch']))
    elif cfgs.saver.pretrain_model:
        assert os.path.isfile(
            cfgs.saver.pretrain_model), 'Not found pretrain model: {}'.format(
                cfgs.saver.pretrain_model)
        checkpoint = torch.load(cfgs.saver.pretrain_model)
        check_keys(model=model, checkpoint=checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("pretrain training from '{}'".format(
            cfgs.saver.pretrain_model))

    if args.rank == 0 and cfgs.saver.get('save_dir', None):
        if not os.path.exists(cfgs.saver.save_dir):
            os.makedirs(cfgs.saver.save_dir)
            logger.info("create checkpoint folder {}".format(
                cfgs.saver.save_dir))

    # Data loading code
    train_loader, train_sampler, test_loader, _ = build_dataloader(
        cfgs.dataset, args.world_size)

    # test mode
    if args.test:
        test(test_loader, model, criterion, args)
        return

    # choose scheduler
    lr_scheduler = torch.optim.lr_scheduler.__dict__[
        cfgs.trainer.lr_scheduler.type](optimizer if isinstance(
            optimizer, torch.optim.Optimizer) else optimizer.optimizer,
                                        **cfgs.trainer.lr_scheduler.kwargs,
                                        last_epoch=args.start_epoch - 1)

    monitor_writer = None
    if args.rank == 0 and cfgs.get('monitor', None):
        if cfgs.monitor.get('type', None) == 'pavi':
            from pavi import SummaryWriter
            if cfgs.monitor.get("_taskid", None):
                monitor_writer = SummaryWriter(session_text=yaml.dump(
                    args.config),
                                               **cfgs.monitor.kwargs,
                                               taskid=cfgs.monitor._taskid)
            else:
                monitor_writer = SummaryWriter(session_text=yaml.dump(
                    args.config),
                                               **cfgs.monitor.kwargs)

    # training
    # args.max_epoch=1
    for epoch in range(args.start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args,
              monitor_writer)

        if (epoch + 1) % args.test_freq == 0 or epoch + 1 == args.max_epoch:
            # evaluate on validation set
            loss, acc1, acc5 = test(test_loader, model, criterion, args)

            if args.rank == 0:
                if monitor_writer:
                    monitor_writer.add_scalar('Accuracy_Test_top1', acc1,
                                              len(train_loader) * epoch)
                    monitor_writer.add_scalar('Accuracy_Test_top5', acc5,
                                              len(train_loader) * epoch)
                    monitor_writer.add_scalar('Test_loss', loss,
                                              len(train_loader) * epoch)

                checkpoint = {
                    'epoch': epoch + 1,
                    'arch': cfgs.net.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict()
                }

                ckpt_path = os.path.join(
                    cfgs.saver.save_dir,
                    cfgs.net.arch + '_ckpt_epoch_{}.pth'.format(epoch))
                best_ckpt_path = os.path.join(cfgs.saver.save_dir,
                                              cfgs.net.arch + '_best.pth')
                torch.save(checkpoint, ckpt_path)
                if acc1 > best_acc1:
                    best_acc1 = acc1
                    shutil.copyfile(ckpt_path, best_ckpt_path)

        lr_scheduler.step()

    # finished
    results = {}
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            try:
                results = json.load(f)
            except:
                pass

    if results.get('inceptionv3', None) is None:
        results['inceptionv3'] = {}
    results['inceptionv3']['accu'] = {'accuracy_top-1': best_acc1}

    with open(args.output, 'w') as f:
        json.dump(results, f)


def train(train_loader, model, criterion, optimizer, epoch, args,
          monitor_writer):
    batch_time = AverageMeter('Time', ':.3f', 200)
    data_time = AverageMeter('Data', ':.3f', 200)

    losses = AverageMeter('Loss', ':.4f', 50)
    top1 = AverageMeter('Acc@1', ':.2f', 50)
    top5 = AverageMeter('Acc@5', ':.2f', 50)

    memory = AverageMeter('Memory(MB)', ':.0f')
    progress = ProgressMeter(len(train_loader),
                             batch_time,
                             data_time,
                             losses,
                             top1,
                             top5,
                             memory,
                             prefix="Epoch: [{}/{}]".format(
                                 epoch + 1, args.max_epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        stats_all = torch.tensor([loss.item(), acc1[0].item(),
                                  acc5[0].item()]).float().cuda()
        dist.all_reduce(stats_all)
        stats_all /= args.world_size

        losses.update(stats_all[0].item())
        top1.update(stats_all[1].item())
        top5.update(stats_all[2].item())
        memory.update(torch.cuda.max_memory_allocated() / 1024 / 1024)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_freq == 0:
            progress.display(i)
            if args.rank == 0 and monitor_writer:
                cur_iter = epoch * len(train_loader) + i
                monitor_writer.add_scalar('Train_Loss', losses.avg, cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top1', top1.avg,
                                          cur_iter)
                monitor_writer.add_scalar('Accuracy_train_top5', top5.avg,
                                          cur_iter)


def test(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':.3f', 10)
    losses = AverageMeter('Loss', ':.4f', -1)
    top1 = AverageMeter('Acc@1', ':.2f', -1)
    top5 = AverageMeter('Acc@5', ':.2f', -1)
    stats_all = torch.Tensor([0, 0, 0]).long()
    progress = ProgressMeter(len(test_loader),
                             batch_time,
                             losses,
                             top1,
                             top5,
                             prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5), raw=True)

            losses.update(loss.item())
            top1.update(acc1[0].item() * 100.0 / target.size(0))
            top5.update(acc5[0].item() * 100.0 / target.size(0))

            stats_all.add_(
                torch.tensor([acc1[0].item(), acc5[0].item(),
                              target.size(0)]).long())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                progress.display(i)

        logger_all.info(
            ' Rank {} Loss {:.4f} Acc@1 {} Acc@5 {} total_size {}'.format(
                args.rank, losses.avg, stats_all[0].item(),
                stats_all[1].item(), stats_all[2].item()))

        loss = torch.tensor([losses.avg])
        dist.all_reduce(loss.cuda())
        loss_avg = loss.item() / args.world_size
        dist.all_reduce(stats_all.cuda())
        acc1 = stats_all[0].item() * 100.0 / stats_all[2].item()
        acc5 = stats_all[1].item() * 100.0 / stats_all[2].item()

        logger.info(
            ' * All Loss {:.4f} Acc@1 {:.3f} ({}/{}) Acc@5 {:.3f} ({}/{})'.
            format(loss_avg, acc1, stats_all[0].item(), stats_all[2].item(),
                   acc5, stats_all[1].item(), stats_all[2].item()))

    return loss_avg, acc1, acc5


if __name__ == '__main__':
    main()

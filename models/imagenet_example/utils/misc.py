import numpy as np
import torch
import logging

logger = logging.getLogger()


def check_keys(model, checkpoint):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(checkpoint['state_dict'].keys())
    missing_keys = model_keys - ckpt_keys
    for key in missing_keys:
        logger.warning('missing key in model:{}'.format(key))
    unexpected_keys = ckpt_keys - model_keys
    for key in unexpected_keys:
        logger.warning('unexpected key in checkpoint:{}'.format(key))
    shared_keys = model_keys & ckpt_keys
    for key in shared_keys:
        logger.info('shared key:{}'.format(key))
    return shared_keys


def accuracy(output, target, topk=(1, ), raw=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(
                0, keepdim=True)
            if raw:
                res.append(correct_k)
            else:
                res.append(correct_k.mul(100.0 / target.size(0)))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value
       When length < 0 , save all history data """

    def __init__(self, name, fmt=':f', length=1):
        self.name = name
        self.fmt = fmt
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 1:
            self.history = []
        elif self.length < 0:
            self.count = 0
            self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val):
        self.val = val
        if self.length > 1:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            self.avg = np.mean(self.history)
        elif self.length < 0:
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count

    def __str__(self):
        if self.length == 0 or self.length == 1:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, *meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

import numpy as np
import tensorflow as tf
from long_tail_bench.core.executer import Executer


def bbox2delta(proposals,
               gt,
               means=(0.0, 0.0, 0.0, 0.0),
               stds=(1.0, 1.0, 1.0, 1.0)):
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = tf.math.log(gw / pw)
    dh = tf.math.log(gh / ph)
    deltas = tf.stack([dx, dy, dw, dh], axis=-1)

    means = np.array(means)
    means = tf.expand_dims(means, 0)
    stds = np.array(stds)
    stds = tf.expand_dims(stds, 0)

    means = tf.cast(means, dtype=tf.float32)
    stds = tf.cast(stds, dtype=tf.float32)
    deltas = deltas - means
    deltas = deltas / stds

    return deltas


def args_adaptor(np_args):
    proposals = tf.convert_to_tensor(np_args[0]).cuda()
    gt = tf.convert_to_tensor(np_args[1]).cuda()
    means = np_args[2].tolist()
    stds = np_args[3].tolist()
    return [proposals, gt, means, stds]


def executer_creator():
    return Executer(bbox2delta, args_adaptor)

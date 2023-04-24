# Copyright (c) OpenMMLab. All rights reserved.
import time
import sys

from .hook import HOOKS, Hook


@HOOKS.register_module()
class IterTimerHook(Hook):
    def before_epoch(self, runner):
        self.iter_cnt = 0
        self.total_iter_time = 0
        self.total_iter_data_time = 0
        self.total_iter_op_time = 0
        self.begin_iter = 200
        self.end_iter=500
        self.t = time.time()
        self.t_epoch_begin = self.t
        self.t_iter_begin = self.t

    def before_iter(self, runner):
        self.t_iter_begin = time.time()
        runner.log_buffer.update({'data_time': self.t_iter_begin - self.t})

    def after_iter(self, runner):
        iter_t = time.time()-self.t 
        iter_data_time = self.t_iter_begin - self.t
        iter_op_time = iter_t - iter_data_time
        runner.log_buffer.update({'time': iter_t})
        
        self.t = time.time()
        # import pdb; pdb.set_trace()
        if self.iter_cnt > self.end_iter:
            runner.logger.info(f"===average iter time: {self.total_iter_time/(self.end_iter-self.begin_iter)} ===average iter data time: {self.total_iter_data_time/(self.end_iter-self.begin_iter)} ===average iter op time: {self.total_iter_op_time/(self.end_iter-self.begin_iter)} ")
            sys.exit()
        if self.iter_cnt >= self.begin_iter:
            self.total_iter_time += iter_t
            self.total_iter_data_time += iter_data_time
            self.total_iter_op_time += iter_op_time
        self.iter_cnt += 1


import time
import sys
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class CustomIterTimerHook(Hook):
    priority = 'NORMAL'
    
    def __init__(self, begin_iter=200, end_iter=500):
        self.begin_iter = begin_iter
        self.end_iter = end_iter
    
    def before_train_epoch(self, runner):
        self.iter_cnt = 0
        self.total_iter_time = 0
        self.total_iter_data_time = 0
        self.total_iter_op_time = 0
        self.t = time.time()
        self.t_iter_begin = self.t

    def before_train_iter(self, runner, batch_idx=None, data_batch=None):
        self.t_iter_begin = time.time()
        runner.message_hub.update_info('data_time', self.t_iter_begin - self.t)

    def after_train_iter(self, runner, batch_idx=None, data_batch=None, outputs=None):
        iter_t = time.time() - self.t 
        iter_data_time = self.t_iter_begin - self.t
        iter_op_time = iter_t - iter_data_time
        runner.message_hub.update_info('time', iter_t)
        self.t = time.time()
        
        if self.iter_cnt > self.end_iter:
            avg_iter = self.total_iter_time / (self.end_iter - self.begin_iter)
            avg_data = self.total_iter_data_time / (self.end_iter - self.begin_iter)
            avg_op = self.total_iter_op_time / (self.end_iter - self.begin_iter)
            runner.logger.info(f"=== AVG_ITER_TIME: {avg_iter:.4f}s | DATA: {avg_data:.4f}s | OP: {avg_op:.4f}s ===")
            sys.exit(0)
            
        if self.iter_cnt >= self.begin_iter:
            self.total_iter_time += iter_t
            self.total_iter_data_time += iter_data_time
            self.total_iter_op_time += iter_op_time
        self.iter_cnt += 1

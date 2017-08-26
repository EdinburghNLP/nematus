'''
Training monitoring (experimental)
'''

import sys
import numpy
from collections import OrderedDict

import theano
import theano.tensor as tensor

class TrainingMonitor(object):
    '''
    Log tensors to a file
    '''

    def __init__(self):
        self.log_file = 'training_monitor.log'
        self.monitors = []
        self.log_fs = None
        self.counter = 0
        self.n_flush = 10
        self.enable_add = False
    
    def add_monitor(self, monitor):
        if self.enable_add:
            self.monitors.append(monitor)

    def init_logfile(self):
        if not self.monitors:
            return
        self.log_fs = open(self.log_file, "w")
        fields = ['step'] + [monitor.name for monitor in self.monitors]
        print >> self.log_fs, ", ".join(fields)

    def monitor_step(self, step_id, *monitored_values):
        if not self.monitors:
            return
        if self.log_fs is None:
            self.init_logfile()

        vals = [str(step_id)] + [str(val) for val in monitored_values]
        print >> self.log_fs, ", ".join(vals)

        self.counter += 1
        if (self.counter % self.n_flush) == 0:
            self.log_fs.flush()

    def flush_logfile(self):
        if self.log_fs is not None:
            self.log_fs.flush()

    def close_logfile(self):
        if self.log_fs is not None:
            self.log_fs.close()

the_training_monitor = TrainingMonitor()


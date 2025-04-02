# Created by Steven Walton
# Contact: code@walton.mozmail.com
# Modified Cosine scheduler to handle a bit more flexibility
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineScheduler(_LRScheduler):
    '''
    Flexible Cosine Scheduler that allows for a more flexible learning rate
    scheduler. Allows for features such as linear or exponential warmup, a
    plateaued top LR before beginning decay, as well as a cooldown period
    
    [ARGS]
        optimizer: (torch.optim) your optimizer
        total_epochs: (int) number of total epochs in the run
        warmup: (int) number of warmup epochs. Epochs at a lower learning rate
            increasing to the base learning rate
        cooldown: (int) number of epochs to train at the minimum learning rate
        last_epoch: (int) ignore this
        pause_epochs: (int) number of epochs to train at full learning rate
        min_lr: (float) smallest learning rate. Constrains minimal step size.
            Make this larger than precision level.
        linear_wu: (bool) True if you want to use a linear warmup ele non-linear
    '''
    def __init__(self,
                 optimizer,       # Input optimizer you want to use
                 total_epochs,    # Number of total epochs to run
                 warmup=0,        # Number of warmup epochs
                 cooldown=0,      # NUmber of cooldown epochs
                 last_epoch=-1,
                 pause_epochs=0,  # Number of epochs to train at full LR, after warmup, before cosine starts
                 min_lr=1e-8,     # Minimum learning rate
                 linear_wu=False,
                 ):
        self.total_epochs = total_epochs
        self.warmup = warmup
        self.cooldown = cooldown
        self.pause_epochs = pause_epochs
        self.min_lr = min_lr
        self.linear_wu = linear_wu

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            if self.linear_wu:
                return [max(self.min_lr, base_lr * (self.last_epoch / self.warmup)) for base_lr in self.base_lrs]
            else:
                return [max(self.min_lr,
                            base_lr / (self.warmup - self.last_epoch))
                        for base_lr in self.base_lrs]
        elif self.last_epoch < (self.warmup + self.pause_epochs):
            return [base_lr for base_lr in self.base_lrs]
        elif self.last_epoch < (self.total_epochs - self.cooldown):
            return [max(self.min_lr,
                        base_lr * 0.5 * (1. + math.cos(math.pi * (self.last_epoch - self.pause_epochs - self.warmup) / (self.total_epochs - self.warmup - self.pause_epochs - self.cooldown))))
                        for base_lr in self.base_lrs]
        else: # Cooldown
            #return [self.min_lr for _ in self.base_lrs]
            return [max(self.min_lr, base_lr) for base_lr in self.base_lrs]

    def step(self, value=None, epoch=None):
        return super().step(epoch)

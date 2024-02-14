import logging
from torch import nn
import torch


logger = logging.getLogger(__name__)


class ScheduledDropout(nn.Module):
    def __init__(self, start, end, p, use_inst=False, use_spatial=False):
        super().__init__()
        self.training_step = 0
        self.start = start
        self.end = end
        self.p = p
        self.use_inst=use_inst
        self.use_spatial = use_spatial
        self.droout = nn.Dropout(p)

    def forward(self, x):
        if self.p==0:
            return x
        if self.training:
            self.training_step +=1
        if self.training_step<self.start:
            return x
        elif self.start<=self.training_step<self.end:
            #if self.training_step== (self.start+1):
            #    logger.info('start dropout %s', self.start)
            if self.use_inst:
                mask = torch.ones_like(x[:, :1, :1])
                mask = self.droout(mask)
                return x*mask
            elif self.use_spatial:
                mask = torch.ones_like(x[:, :1, :])
                mask = self.droout(mask)
                return x * mask
            elif self.use_spatial:
                mask = torch.ones_like(x[:, :1, :])
                mask = self.droout(mask)
                return x*mask
            else:
                return self.droout(x)
        else:
            return x

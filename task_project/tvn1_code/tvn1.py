import torch
from torch import nn
from blocks_pkg.blocks import *

class TVN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(TinyBlock1())
        
    def forward(self, input):
        x, vid_lens = input
        x, vid_lens = self.body((x, vid_lens))
        return x
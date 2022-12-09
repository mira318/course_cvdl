import torch
from torch import nn
from blocks import *

class TVN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(TinyBlock1())
        
    def forward(self, input):
        # batches_frames, channels, leng, h, w = input.shape
        # input = input.permute(0, 2, 1, 3, 4)
        # input = input.reshape(batches_frames * leng, channels, h, w, [l for _ in range(batches_frames)])
        
        print('self.body[0] = ', self.body[0])
        x, vid_lens = input
        x, vid_lens = self.body((x, vid_lens))
        return x
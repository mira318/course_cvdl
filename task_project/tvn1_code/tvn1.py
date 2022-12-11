import torch
from torch import nn
from blocks_pkg.blocks import *

class TVN1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.body = nn.Sequential(
            TinyBlock1(),
            TinyBlock2(),
            TinyBlock3(),
            TinyBlock4()
        )
        self.head_linear1 = nn.Linear(512, 512)
        self.head_lrelu = nn.LeakyReLU(0.2)
        self.head_linear2 = nn.Linear(512, num_classes)
        
    def forward(self, input):
        x, vid_lens = input
        x, vid_lens = self.body((x, vid_lens))
        
        _, channels, h, w = x.shape
        out = []
            
        for i in range(len(vid_lens)):
            idx = sum(vid_lens[:i])
            curr_len = vid_lens[i]
            part_x = x[idx:(idx + curr_len)]
                
            y = torch.mean(part_x, dim = [0, 2, 3], keepdim = True)
            y = y.squeeze(-1).squeeze(-1)    
            out.append(y)
                
        x = torch.cat(out, dim = 0)
        print('before head x.shape = ', x.shape)
        x = self.head_linear1(x)
        x = self.head_lrelu(x)
        x = self.head_linear2(x)
        return x
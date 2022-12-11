import torch
from torch import nn
from .sublocks import *


class TinyBlock1(nn.Module):
    def __init__(self, strides = 0, in_channels = 3, num_frames = 2):
        super().__init__()
        self.spatial_conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32,
                                      kernel_size = 3, stride = 1, padding = 1)  
        self.temporal_pool = MultiMaxPool(kernel_size = 2, stride = 2, padding = 0)
        self.context_gate1 = MultiCG(in_channels = 32, out_channels = 32, 
                                     kernel_size = 3, stride = 2, padding = 1)
        
        self.spatial_conv2 = nn.Conv2d(in_channels = 32, out_channels = 64,
                                      kernel_size = 3, stride = 1, padding = 1)  
        self.temporal_avg = MultiAvg(kernel_size = 3, stride = 1, padding = 1)
        
        self.context_gate2 = MultiCG(in_channels = 64, out_channels = 64, 
                                     kernel_size = 3, stride = 2, padding = 1)        
        
    def forward(self, input):
        x, vid_lens = input
        x = self.spatial_conv1(x)
        x, vid_lens = self.temporal_pool(x, vid_lens)
            
        x = self.spatial_conv2(self.context_gate1(x))
        x, vid_lens = self.temporal_avg(x, vid_lens)
        x = self.context_gate2(x)
        return (x, vid_lens)
        
        
class TinyBlock2(nn.Module):
    def __init__(self, in_channels = 64, out_channels = 128):
        super().__init__()
        self.conv1_1 = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, 
                               kernel_size = 3, stride = 1, padding = 1, dilation = 1, 
                               groups = 1)
        
        self.se = MultiSE(out_channels, out_channels)
        
    def forward(self, input):
        x, vid_lens = input
        x = nn.AdaptiveAvgPool2d((x.shape[2] // 2, x.shape[3] // 2))(x)
            
        _, channels, h, w = x.shape
        new_vid_lens = []
        out = []
            
        for i in range(len(vid_lens)):
            idx = sum(vid_lens[:i])
            curr_len = vid_lens[i]
            part_x = x[idx:(idx + curr_len)]
                
            part_x = part_x.reshape(curr_len, channels, h * w)
            y = torch.transpose(part_x, 2, 0)
            y = self.conv1_1(y)
            y = torch.transpose(y, 2, 0)
            y = y.reshape(y.shape[0], y.shape[1], h, w)
                
            new_vid_lens.append(y.shape[0])
            out.append(y)
                
        x = torch.cat(out, dim = 0)
        vid_lens = new_vid_lens
            
        x = self.se(x)
        return x, vid_lens

class TinyBlock3(nn.Module):
    def __init__(self, in_channels = 128):
        super().__init__()
        self.sublock1 = Tiny_3_sublock(in_channels = 128)
        self.sublock2 = Tiny_3_sublock(in_channels = 128)
        self.sublock3 = Tiny_3_sublock(in_channels = 128)
        
    def forward(self, input):
        x = self.sublock3(self.sublock2(self.sublock1(input)))
        return x
    
    
class TinyBlock4(nn.Module):
    def __init__(self, in_channels = 128, out_channels = 512):
        super().__init__()
        self.conv1_1 = nn.Conv1d(in_channels = 128, out_channels = 256, 
                               kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1)
        self.context_gate1 = MultiCG(in_channels = 256, out_channels = 256, 
                                     kernel_size = 3, stride = 1, padding = 1)
        self.se1 = MultiSE(256, 256)
        
        self.conv2_1 = nn.Conv1d(in_channels = 256, out_channels = 512, 
                               kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1)
        self.context_gate2 = MultiCG(in_channels = 512, out_channels = 512, 
                                     kernel_size = 3, stride = 1, padding = 1)
        self.se2 = MultiSE(512, 512)
        
        
    def forward(self, input):
        x, vid_lens = input
        x = nn.AdaptiveAvgPool2d((x.shape[2] // 2, x.shape[3] // 2))(x)
        
        _, channels, h, w = x.shape
        new_vid_lens = []
        out = []
            
        for i in range(len(vid_lens)):
            idx = sum(vid_lens[:i])
            curr_len = vid_lens[i]
            
            part_x = x[idx:(idx + curr_len)]  
            part_x = part_x.reshape(curr_len, channels, h * w)
            y = torch.transpose(part_x, 2, 0)
            
            y = self.conv1_1(y)
            y = torch.transpose(y, 2, 0)
            y = y.reshape(y.shape[0], y.shape[1], h, w)
                
            new_vid_lens.append(y.shape[0])
            out.append(y)
                
        x = torch.cat(out, dim = 0)
        vid_lens = new_vid_lens
        x = self.se1(self.context_gate1(x))
        
        x = nn.AdaptiveAvgPool2d((x.shape[2] // 2, x.shape[3] // 2))(x)
        _, channels, h, w = x.shape
        new_vid_lens = []
        out = []
            
        for i in range(len(vid_lens)):
            idx = sum(vid_lens[:i])
            curr_len = vid_lens[i]
            part_x = x[idx:(idx + curr_len)]
                
            part_x = part_x.reshape(curr_len, channels, h * w)
            y = torch.transpose(part_x, 2, 0)
            y = self.conv2_1(y)
            y = torch.transpose(y, 2, 0)
            y = y.reshape(y.shape[0], y.shape[1], h, w)
                
            new_vid_lens.append(y.shape[0])
            out.append(y)
                
        x = torch.cat(out, dim = 0)
        vid_lens = new_vid_lens
        x = self.se2(self.context_gate2(x))
        
        return x, vid_lens
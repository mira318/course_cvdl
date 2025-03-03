import torch
import math
from torch import nn

class MultiMaxPool(nn.Module):
    def __init__(self, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward_single(self, x):
        batch_frames, c, h, w = x.shape
        x = torch.transpose(x.reshape(batch_frames, c, h * w), 2, 0)
        y = nn.MaxPool1d(kernel_size = self.kernel_size, stride = self.stride, padding = self.padding)(x)
        y = torch.transpose(y, 2, 0)
        y = y.reshape(y.shape[0], y.shape[1], h, w)
        return y
    
    def forward(self, x, vid_lens):
        new_vid_lens = []
        out = []
        for i in range(len(vid_lens)):
            idx = sum(vid_lens[:i])
            curr_len = vid_lens[i]
            
            part_x = x[idx:(idx + curr_len)]
            y = self.forward_single(part_x)
            
            new_vid_lens.append(y.shape[0])
            out.append(y)
            
        return torch.cat(out, dim = 0), new_vid_lens

class MultiCG(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1 , padding = 0,
                 dilation = 1, groups = 1, bias = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        if kernel_size == 1:
            self.simple = True
        else:
            self.simple = False
            self.avg_pool = nn.AdaptiveAvgPool2d((kernel_size, kernel_size))
            self.linear1 = nn.Linear(in_features = kernel_size * kernel_size, 
                                     out_features = int((kernel_size * kernel_size) / 2 + 1),
                                     bias = False)
            
            self.bn1 = nn.BatchNorm1d(in_channels)
            self.bn2 = nn.BatchNorm1d(in_channels)
            self.relu = nn.ReLU()
            
            if in_channels // 16:
                self.reduced_features = 16
            else:
                self.reduced_features = in_channels
            
            self.linear2 = nn.Linear(in_features = self.reduced_features, 
                                     out_features = out_channels // (in_channels // self.reduced_features),
                                     bias = False)
            self.bn3 = nn.BatchNorm1d(out_channels)
            
            self.linear3 = nn.Linear(in_features = int((kernel_size * kernel_size) / 2 + 1),
                                     out_features = kernel_size * kernel_size, bias = False)
            self.linear4 = nn.Linear(in_features = int((kernel_size * kernel_size) / 2 + 1),
                                     out_features = kernel_size * kernel_size, bias = False)
            
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
            self.sigmoid = nn.Sigmoid()            
    
    def forward(self, x):
        if self.simple:
            return nn.Conv2d(self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)(x)
        else:
            batch, channels, h, w = x.shape
            out = self.avg_pool(x)
            out = self.linear1(out.reshape(batch, channels, -1))
            features = out
            out = self.linear3(self.relu(self.bn1(out)))
            
            if self.reduced_features > 3:
                features = self.bn2(features)
                features = features.reshape(batch, channels // self.reduced_features, self.reduced_features, -1)
                features = features.transpose(2, 3)
                features = self.linear2(self.relu(features))
                features = features.transpose(2, 3).contiguous() 
            else:
                features = self.bn2(features)
                features = features.transpose(2, 1)
                features = self.linear2(self.relu(features))
                features = features.transpose(2, 1).contiguous()
           
            features = features.reshape(batch, self.out_channels, -1)
            features = self.linear4(self.relu(self.bn3(features)))
            
            out = out.reshape(batch, 1, channels, self.kernel_size, self.kernel_size)
            features = features.reshape(batch, self.out_channels, 1, self.kernel_size, self.kernel_size)
            out = self.sigmoid(out + features)
            
            x_unfolded = self.unfold(x)
            batch, c_w, last = x_unfolded.shape
            out = out * self.weight.unsqueeze(0)
            out = out.reshape(batch, self.out_channels, -1)
            out = torch.matmul(out, x_unfolded)
            out = out.reshape(batch, self.out_channels, int(math.sqrt(last)), int(math.sqrt(last)))
            return out
        
class MultiAvg(nn.Module):
    def __init__(self, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x, vid_lens):
        lens_sum, channels, h, w = x.shape
        new_vid_lens = []
        out = []
        for i in range(len(vid_lens)):
            idx = sum(vid_lens[:i])
            curr_len = vid_lens[i]
            part_x = x[idx:(idx + curr_len)]
            
            part_x = part_x.reshape(curr_len, channels, h * w)
            part_x = torch.transpose(part_x, 2, 0)
            y = nn.AvgPool1d(kernel_size = self.kernel_size, stride = self.stride, 
                             padding = self.padding)(part_x)
            y = torch.transpose(y, 2, 0)
            y = y.reshape(y.shape[0], y.shape[1], h, w)
            
            new_vid_lens.append(y.shape[0])
            out.append(y)
            
        return torch.cat(out, dim = 0), new_vid_lens

class MultiSE(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1, downsample = None, reduction = 16):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = out_planes, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels = out_planes, out_channels = out_planes, kernel_size = 1,
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.linear1 = nn.Linear(out_planes, out_planes // reduction, bias = False)
        self.relu2 = nn.ReLU()
        
        self.linear2 = nn.Linear(out_planes // reduction, out_planes, bias = False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        projection = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        batch, channel, h, w = out.shape
        out = self.avg_pool(out)
        out = out.reshape(batch, channel)
        out = self.linear2(self.relu2(self.linear1(out)))
        out = self.sigmoid(out).reshape(batch, channel, 1, 1)
        out = x * out.expand_as(x)
        
        if self.downsample is not None:
            projection = self.downsample(x)
        
        out = out + projection
        out = self.relu1(out)
        
        return out
    
class MultiNL(nn. Module):
    def __init__(self, in_channels):
        super().__init__()
        self.subsample = True
        self.need_bn = True
        self.in_channels = in_channels
        self.inter_channels = max(1, self.in_channels // 2) 
        
        self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels = self.inter_channels,
                               kernel_size = 1, stride = 1, padding = 0)
        self.max_pool = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(in_channels = self.inter_channels, out_channels = self.in_channels, 
                               kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)
        
        self.conv3_t = nn.Conv2d(in_channels = self.in_channels, out_channels = self.inter_channels, 
                                 kernel_size = 1, stride = 1, padding = 0)
        
        self.conv4_p = nn.Conv2d(in_channels = self.in_channels, out_channels = self.inter_channels, 
                                 kernel_size = 1, stride = 1, padding = 0)
        
        self.conv5_pr = nn.Conv2d(in_channels = 2 * self.inter_channels, out_channels = 1, 
                                  kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.relu = nn.ReLU()
        
        
    def forward(self, x, return_map = False):
        batches = x.shape[0]
        xg = self.max_pool(self.conv1(x))
        xg = xg.reshape(batches, self.inter_channels, -1)
        xg = xg.permute(0, 2, 1)
        
        xt = self.conv3_t(x)
        xt = xt.reshape(batches, self.inter_channels, -1, 1)
        
        xp = self.max_pool(self.conv4_p(x))
        xp = xp.reshape(batches, self.inter_channels, 1, -1)
        
        h, w = xt.shape[2], xp.shape[3]
        xt = xt.repeat(1, 1, 1, w)
        xp = xp.repeat(1, 1, h, 1)
        
        features = torch.cat([xt, xp], dim = 1)
        features = self.relu(self.conv5_pr(features))
        f_batches, _, h, w = features.shape
        features = features.reshape(f_batches, h, w)
        
        c_new = features.shape[-1]
        y = torch.matmul(features / c_new, xg)
        y = y.permute(0, 2, 1)
        y = y.reshape(batches, self.inter_channels, *x.shape[2:]).contiguous()
        
        y = self.bn1(self.conv2(y))
        
        if not return_map:
            return y + x
        else:
            return y + x, features / c_new
        
        
class Tiny_3_sublock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, 
                                      kernel_size = 3, stride = 1, padding = 1)
        self.conv2_1 = nn.Conv1d(in_channels = in_channels, out_channels = in_channels, 
                                 kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1)
        
        self.nl = MultiNL(in_channels)
        
    def forward(self, input):
        x, vid_lens = input
        projection = x    
        x = self.spatial_conv(x)
        
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
        
        x = self.nl(x)
        x = x + projection
        return x, vid_lens
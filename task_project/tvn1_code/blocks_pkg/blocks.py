import torch
from torch import nn

class MultiMaxPool(nn.Module):
    def __init__(self, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.kerne_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward_single(self, x):
        batch_frames, c, h, w = x.shape()
        x = torch.transpose(x.reshape(batch_frames, c, h * w), 2, 0)
        y = nn.MaxPool1d(x, kernel_size = self.kernel_size, stride = self.stride, 
                         padding = self.padding)
        y = torch.transpose(y, 2, 0)
        y = y.reshape(y.shape[0], y.shape[1], 2, 2)
        return y
    
    def forward(self, x, batch_frames):
        new_batch_frames = []
        out = []
        for i in range(len(batch_frames)):
            idx = torch.sum(batch_frames[:i])
            curr_len = batch_frames[i]
            
            part_x = x[idx:(idx + curr_len)]
            y = self.forward_single(part_x)
            
            new_btches_frames.append(y.shape[0])
            out.append(y)
            
        return torch.cat(out, dim = 0), new_vid_len

class MultiCG(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1 , padding = 0,
                 dilation = 1, groups = 1, bias = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
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
    
    def forward():
        if self.simple:
            return nn.Conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            batch, channels, h, w = x.shape
            out = self.linear1(self.avg_pool(x.reshape(batch, channels, h * w)))
            features = out
            out = self.linear3(self.relu(self.bn1(out)))
            
            if self.reduced_features > 3:
                features = self.bn2(features)
                features = features.reshape(batch, channels // self.reduced_features, reduced_features, -1)
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
            out = out * weight.unsqueeae(0)
            out = out.reshape(batch, self.out_channels, -1)
            out = torch.matmul(out, x_unfolded)
            out = out.reshape(batch, self.out_channels, int(torch.sqrt(last)), int(torch.sqrt(last)))
            return out
        
class MultiAvg(nn.Module):
    def __init__(self, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x, vid_lens):
        new_vid_lens = []
        out = []
        for i in range(len(vid_lens)):
            idx = torch.sum(vid_lens[:i])
            curr_len = vid_lens[i]
            part_x = x[idx:(idx + curr_len)]
            
            part_x = part_x.reshape(curr_len, channels, h * w)
            part_x = part_x.transpose(part_x, 2, 0)
            y = nn.AvgPool1d(kernel_size = self.kernel_size, 
                             stride = self.stride, 
                             padding = self.padding
                            )(part_x)
            y = y.transpose(y, 2, 0)
            y = y.reshape(y.shape[0], y.shape[1], h, w)
            
            new_vid_lens.append(y.shape[0])
            out.append(y)
            
        return torch.cat(out, dim = 0), new_vid_lens

        
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
            print('was in tinyblock forward')
            x, vid_lens = input
            x = self.spatial_conv1(x)
            x, vid_lens = self.temporal_pool1d(x, vid_lens)
            
            x = self.spatial_conv2(self.context_gate1(x))
            x, vid_lens = self.temporal_avg(x, vid_lens)
            x = self.context_gate2(x)
            return (x, vid_lens)
        
      
        
        
        
        """if strides == 0:
            self.sp_strides = 2
        else:
            self.sp_strides = strides

        if strides == 0:
            self.t_strides = 2
        else:
            self.t_strides = strides
            
        self.num_frames = num_frames
        # believe in channels_last
        # if data_format == 'channels_first':
        #    self.axis = 1
        # else:
        #    self.axis = -1
            
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, 
                               kernel_size = 1, padding = 1, stride = 1)
        self.bn1 = nn.BatchNorm2d(num_features = 64, momentum = 0.9)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, 
                                   padding = int(self.sp_strides / 2), stride = self.sp_strides)
        self.bn2 = nn.BatchNorm2d(num_features = 64, momentum = 0.9)
        self.relu2 = nn.ReLU()
        
        nn.init.normal_(self.conv1.weight, std = 2.0)
        nn.init.normal_(self.conv2.weight, std = 2.0)
        
        self.temporal_maxpool = nn.MaxPool1d(kernel_size = 4, 
                                     stride = self.t_strides, padding = int(self.t_strides / 2))
        
        self.expand_conv = nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size = 1,
                                   padding = 1, stride = 1)
        self.gate_conv = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 1, 
                                   padding = 1, stride = 1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu3 = nn.ReLU()
        nn.init.normal_(self.gate_conv.weight, std = 2.0)
        
    def forward(self, input):
        res = input
        inputs = self.relu1(self.bn1(self.conv1(input)))
        print('after conv1:', inputs.shape)
        inputs = self.relu2(self.bn2(self.conv2(inputs)))
        print('after conv2:', inputs.shape)
        
        features_shape = inputs.shape
        batch_size = features_shape[0] // self.num_frames
        inputs = inputs.reshape(features_shape[0], features_shape[1], features_shape[2] * features_shape[3])
        print('after first reshape:', inputs.shape)
        inputs = self.temporal_maxpool(inputs)
        print('after maxpool:', inputs.shape)
        new_frames = self.num_frames
        print('new_frames = ', new_frames)
        inputs = self.exapnd_conv(inputs)
        inputs = inputs.reshape(features_shape[0], features_shape[1], features_shape[2], features_shape[3])
        print('after second reshape:', inputs.shape)
        features_shape = (inputs.shape[0], 1, 1, inputs.shape[3])
        context_inputs = inputs.reshape(int(inputs.shape[0] // new_frames), 
                                        new_frames * inputs.shape[1], inputs.shape[2], -1)
                                
        feature = torch.mean(context_inputs, dim = (1, 2), keepdim = True)
        feature = self.sigmoid(self.gate_conv(feature))
        feature = feature.expand(feature.shape[0], 1, feature.shape[1], 
                                 feature.shape[2], feature.shape[3])
        pattern = (1, new_frames, 1, 1, 1)
        tiled = torch.tile(feature, pattern)
        inputs = torch.reshape(tailed, features_shape) * inputs
        return nn.relu3(inputs), new_frames"""
                                
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
                                
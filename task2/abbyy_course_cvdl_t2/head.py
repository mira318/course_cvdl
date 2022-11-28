""" Здесь находится 'Голова' CenterNet, описана в разделе 4 статьи https://arxiv.org/pdf/1904.07850.pdf"""
from torch import nn
import torch


class CenterNetHead(nn.Module):
    """
    Принимает на вход тензор из Backbone input[B, K, W/R, H/R], где
    - B = batch_size
    - K = количество каналов (в ResnetBackbone K = 64)
    - H, W = размеры изображения на вход Backbone
    - R = output stride, т.е. во сколько раз featuremap меньше, чем исходное изображение
      (в ResnetBackbone R = 4)

    Возвращает тензора [B, C+4, W/R, H/R]:
    - первые C каналов: probs[B, С, W/R, H/R] - вероятности от 0 до 1
    - еще 2 канала: offset[B, 2, W/R, H/R] - поправки координат в пикселях от 0 до 1
    - еще 2 канала: sizes[B, 2, W/R, H/R] - размеры объекта в пикселях
    """
    def __init__(self, k_in_channels=64, c_classes: int = 2):
        super().__init__()
        self.c_classes = c_classes
        out_channels = c_classes + 4
        self.conv1 = nn.Conv2d(in_channels = k_in_channels, out_channels = out_channels, kernel_size = 3, 
                               stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1,
                               stride = 1, padding = 0)


    def forward(self, input_t: torch.Tensor):
        self.w = input_t.shape[2]
        self.h = input_t.shape[3]
        
        out_t = self.conv2(self.relu(self.conv1(input_t)))
        classes_map = out_t[:, 0:self.c_classes, :, :]
        offset_map = out_t[:, self.c_classes:(self.c_classes + 2), :, :]
        size_map = out_t[:, (self.c_classes + 2):(self.c_classes + 4), :, :]

        class_heatmap = nn.Sigmoid()(classes_map.clone())
        offset_map = nn.Sigmoid()(offset_map.clone())
        size_map = nn.Sigmoid()(size_map.clone())
        
        return torch.cat([class_heatmap, offset_map, size_map], dim=1)

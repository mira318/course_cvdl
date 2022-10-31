"""
Здесь находится backbone на основе resnet-18, в статье "Objects as Points" он описан в
5.Implementation details/Resnet и в Figure 6-b.
"""
from torch import nn
from torchvision.models import resnet18


class HeadlessPretrainedResnet18Encoder(nn.Module):
    """
    Предобученная на imagenet версия resnet, у которой
    нет avg-pool и fc слоев.
    Принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    def __init__(self):
        super().__init__()
        md = resnet18(pretrained=True)
        # все, кроме avgpool и fc
        self.md = nn.Sequential(
            md.conv1,
            md.bn1,
            md.relu,
            md.maxpool,
            md.layer1,
            md.layer2,
            md.layer3,
            md.layer4
        )

    def forward(self, x):
        return self.md(x)


class HeadlessResnet18Encoder(nn.Module):
    """
    Версия resnet, которую надо написать с нуля.
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, 512, H/32, W/32].
    """
    
    class SimpleBlock(nn.Module):
        
        def __init__(self, in_channels, out_channels, kernel_size, padding, stride, is_projection = False):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(num_features = in_channels)
            self.relu1 = nn.ReLU()
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                                   kernel_size = kernel_size, padding = padding, stride = stride)
            
            self.bn2 = nn.BatchNorm2d(num_features = out_channels)
            self.relu2 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                                   kernel_size = kernel_size, padding = padding, stride = 1)
            if is_projection:
                self.project = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                         kernel_size = 1, padding = 0, stride = stride)
            else:
                self.project = lambda x: x
            
        def forward(self, x):
            project = x
            out = self.conv1(self.relu1(self.bn1(x)))
            out = self.conv2(self.relu2(self.bn2(out)))
            return out + self.project(x)
            
    
    def __init__(self):
        # полносверточная сеть, архитектуру можно найти в
        # https://arxiv.org/pdf/1512.03385.pdf, Table1
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, padding = 3, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            
            self.SimpleBlock(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
            self.SimpleBlock(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
            
            self.SimpleBlock(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, stride = 2, 
                        is_projection = True),
            self.SimpleBlock(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, stride = 1),
            
            self.SimpleBlock(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, stride = 2,
                        is_projection = True),
            self.SimpleBlock(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1, stride = 1),
            
            self.SimpleBlock(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1, stride = 2, 
                        is_projection = True),
            self.SimpleBlock(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, stride = 1)
        )

    def forward(self, x):
        y = self.net(x)
        return y


class UpscaleTwiceLayer(nn.Module):
    """
    Слой, повышающий height и width в 2 раза.
    В реализации из "Objects as Points" используются Transposed Convolutions с
    отсылкой по деталям к https://arxiv.org/pdf/1804.06208.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, output_padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size, padding = padding, output_padding = output_padding, stride = 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class ResnetBackbone(nn.Module):
    """
    Сеть-экстрактор признаков, принимает на вход тензор изображений
    [B, 3, H, W], возвращает [B, C, H/R, W/R], где R = 4.
    C может быть выбрано разным, в конструкторе ниже C = 64.
    """
    def __init__(self, pretrained: bool = True, out_channels=64):
        super().__init__()
        # downscale - fully-convolutional сеть, снижающая размерность в 32 раза
        if pretrained:
            self.downscale = HeadlessPretrainedResnet18Encoder()
        else:
            self.downscale = HeadlessResnet18Encoder()

        # upscale - fully-convolutional сеть из UpscaleTwiceLayer слоев, повышающая размерность в 2^3 раз
        downscale_channels = 512 # выход resnet
        channels = [downscale_channels, 256, 128, out_channels]
        layers_up = [
            UpscaleTwiceLayer(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ]
        self.upscale = nn.Sequential(*layers_up)

    def forward(self, x):
        x = self.downscale(x)
        x = self.upscale(x)
        return x


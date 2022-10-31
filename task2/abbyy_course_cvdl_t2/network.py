from torch import nn
import torch
from abbyy_course_cvdl_t2.head import CenterNetHead
from abbyy_course_cvdl_t2.backbone import ResnetBackbone
from abbyy_course_cvdl_t2.convert import PointsToObjects


class PointsNonMaxSuppression(nn.Module):
    """
    Описан в From points to bounding boxes, фильтрует находящиеся
    рядом объекты.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, points):
        batch_size, classes, h, w = points.shape
        padded_points = nn.functional.pad(points, (1, 1, 1, 1), "constant", 0)
        cleared_points = torch.zeros_like(points)
        for b in range(batch_size):
            for c in range(classes):
                for i in range(h):
                    for j in range(w):
                        if padded_points[b][c][i + 1][j + 1] == torch.max(padded_points[b][c][i:(i + 3), j:(j + 3)]):
                            cleared_points[b][c][i][j] = padded_points[b][c][i + 1][j + 1]
                            
        return cleared_points


class ScaleObjects(nn.Module):
    """
    Объекты имеют размеры в пикселях, и размер входа в сеть
    в несколько раз больше (R, output_stride), чем размер выхода.
    Из-за этого все объекты, полученные с помощью PointsToObjects
    имеют меньший размер.
    Чтобы это компенисровать, надо увеличить размеры объектов.
    """
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, objects):
        b, n, d6 = objects.shape
        objects[:, :, :4] *= self.scale
        return objects


class CenterNet(nn.Module):
    """
    Детектор объектов из статьи 'Objects as Points': https://arxiv.org/pdf/1904.07850.pdf
    """
    def __init__(self, pretrained=True, head_kwargs={}, nms_kwargs={}, points_to_objects_kwargs={}):
        super().__init__()
        self.backbone = ResnetBackbone(pretrained)
        self.head = CenterNetHead(**head_kwargs)
        self.return_objects = torch.nn.Sequential(
            PointsNonMaxSuppression(**nms_kwargs),
            PointsToObjects(**points_to_objects_kwargs),
            ScaleObjects()
        )

    def forward(self, input_t, return_objects=False):
        x = input_t
        x = self.backbone(x)
        x = self.head(x)
        if return_objects:
            x = self.return_objects(x)
        return x

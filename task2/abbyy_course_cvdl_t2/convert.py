"""
Здесь находятся слои для конвертации между представлениями объектов.

В статье "Objects as Points" используется два представления объектов:
- Objects(bounding boxes): тензор [B, N, 6] объектов, где каждый объект имеет характеристики [y, x, h, w, cls, confidence]
- Points(heatmaps): тензор [B, C + 4, H, W] с вероятностями C классов и характеристиками
  [dy, dx, h, w] для каждой точки (y, x)
Все характеристики вычисляются в пикселях.
"""
import torch
from torch import nn
from typing import Tuple


class ObjectsToPoints(nn.Module):
    """
    В статье используются следующие обозначения:
    - H, W: 2D размеры входного изображения
    - H/R, W/R: 2D размеры выходного featuremap
    - R: output_stride, соотношение размеров входа-выхода(равен 4 для resnet backbone)

    Отличия реализации:
    - для упрощения ожидаются квадратные входы и выходы и используется HW == H == W
    - не используется R - ожидается что размеры объектов уже отмасштабированы.
    - для упрощения размер гауссианы не зависит от размера объекта и определяется smooth_kernel_size

    Пример:
        - размер входа сети: [256, 256]
        - размер выхода сети: [64, 64]
        - исходная gt деткция: [y=128, x=128, h=64, x=32, cls=0, confidence=1]
        Чтобы корректно отобразить объект на heatmap размера [64, 64] нужно:
        - передать в конструктор модуля hw=64
        - отмасштабировать исходную детекцию в 4 раза: [y=32, x=32, h=16, x=8, cls=0, confidence=1]
    """
    def __init__(self, hw: int = 64, num_classes: int = 2, smooth_kernel_size: int = 3):
        super().__init__()
        self.hw = hw
        self.c = num_classes
        if smooth_kernel_size == 0:
            self.smooth_kernel = None
        else:
            self.smooth_kernel = self._gaussian_2d(smooth_kernel_size)

    def forward(self, objects):
        """
        Реализован для упрощения - можно не писать с нуля, а только реализовать правильно методы.
        При желании, можно и переписать самостоятельно с нуля.
        """
        b, n, d6 = objects.shape
        assert d6 == 6 # y, x, h, w, cls, confidence
        points_heatmap = self.create_default_heatmaps(b, self.hw, self.c)
        points_heatmap = points_heatmap.to(objects.device)
        assert points_heatmap.shape == (b, self.c + 4, self.hw, self.hw)

        object_confidence = objects[:, :, -1]
        # На разных изображениях разное количество объектов, но тензор имеет
        # фиксированный размер [B, N, 6] - значит какие-то объекты "не настоящиее"
        # Такие объекты должны быть заполнены нулями
        assert object_confidence.shape == (b, n)
        is_real_object = (object_confidence == 1).float().flatten()

        # Проставляем 1 в локации объекта в соответствующем классе, если объект - реальный
        batch_idx, y_idx, x_idx = self.compute_objects_locations(objects)
        cls_idx = objects[:, :, -2].flatten().long()
        assert batch_idx.shape == (b * n,), (b, n, batch_idx.shape)
        assert y_idx.shape == (b * n,), y_idx.shape
        assert x_idx.shape == (b * n,), x_idx.shape
        assert batch_idx.dtype == torch.int64, batch_idx.dtype
        assert y_idx.dtype == torch.int64, y_idx.dtype
        assert x_idx.dtype == torch.int64, x_idx.dtype
        self._put(points_heatmap, (batch_idx, cls_idx, y_idx, x_idx), is_real_object)
        # При необходимости - "размываем" точки гауссианой
        if self.smooth_kernel is not None:
            if self.smooth_kernel.device != points_heatmap.device:
                self.smooth_kernel = self.smooth_kernel.to(points_heatmap.device)
            points_heatmap = self.smooth_points_heatmap(points_heatmap, self.smooth_kernel)

        # Проставляем поправки центра dy-dx в локации объекта, если объект - реальный
        dy, dx = self.compute_objects_offsets(objects)
        channel_idx = torch.zeros_like(batch_idx) + self.c
        assert dy.shape == is_real_object.shape, dy.shape
        self._put(points_heatmap, (batch_idx, channel_idx, y_idx, x_idx), is_real_object * dy)
        channel_idx += 1
        assert dx.shape == is_real_object.shape, dx.shape
        self._put(points_heatmap, (batch_idx, channel_idx, y_idx, x_idx), is_real_object * dx)

        # Проставляем размеры в локации объекта, если объект - реальный
        hy, wx = self.compute_objects_sizes(objects)
        channel_idx += 1
        assert hy.shape == is_real_object.shape, hy.shape
        self._put(points_heatmap, (batch_idx, channel_idx, y_idx, x_idx), is_real_object * hy)
        channel_idx += 1
        assert wx.shape == is_real_object.shape, wx.shape
        self._put(points_heatmap, (batch_idx, channel_idx, y_idx, x_idx), is_real_object * wx)
        return points_heatmap

    @classmethod
    def create_default_heatmaps(self, b:int, hw:int, c:int) -> torch.Tensor:
        """
        По умолчанию heatmaps - пустые и заполнены нулями.
        """
        return torch.zeros([b, c + 4, hw, hw])

    @classmethod
    def compute_objects_locations(self, objects: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Принимает тензор объектов [B, N, 6].
        Возвращает для всех объектов их локацию (индекс точки в heatmap), в виде трех плоских индексов:
            batch_idx[B * N], y_idx[B * N], x_idx[B * N]
        """
        batch_idx = []
        y_idx = []
        x_idx = []
        batch_size, objects_per_image, d6 = objects.shape
        for b in range(batch_size):
            for n in range(objects_per_image):
                batch_idx.append(b)
                y_idx.append(torch.floor(objects[b][n][0]))
                x_idx.append(torch.floor(objects[b][n][1]))
                
        batch_idx = torch.tensor(batch_idx).to(objects.device)
        y_idx = torch.tensor(y_idx).to(objects.device)
        x_idx = torch.tensor(x_idx).to(objects.device)
        return batch_idx.long(), y_idx.long(), x_idx.long()

    @classmethod
    def compute_objects_offsets(self, objects: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Принимает тензор объектов [B, N, 6].
        Возвращает для всех объектов их поправки положения их центров по отношению к локациям
         в виде двух плоских тензоров: dy[B * N], dx[B * N]
        """
        dy = []
        dx = []
        batch_size, objects_per_image, d6 = objects.shape
        for b in range(batch_size):
            for n in range(objects_per_image):
                dy.append(objects[b][n][0] - torch.floor(objects[b][n][0]))
                dx.append(objects[b][n][1] - torch.floor(objects[b][n][1]))
                
        dy = torch.tensor(dy).to(objects.device)
        dx = torch.tensor(dx).to(objects.device)
        return dy, dx

    @classmethod
    def compute_objects_sizes(cls, objects: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Принимает тензор объектов [B, N, 6].
        Возвращает для всех объектов их размеры в пикселях в виде двух плоских тензоров:
            h[B * N], w[B * N]
        """
        
        w = []
        h = []
        batch_size, objects_per_image, d6 = objects.shape
        for b in range(batch_size):
            for n in range(objects_per_image):
                h.append(objects[b][n][2])
                w.append(objects[b][n][3])
        
        h = torch.tensor(h).to(objects.device)
        w = torch.tensor(w).to(objects.device)
        return h.flatten(), w.flatten()

    @classmethod
    def smooth_points_heatmap(cls, points_heatmap: torch.Tensor, smooth_kernel:torch.Tensor):
        """
        Сглаживает one-hot points_heatmap ядром с гауссианой.
        Скорее всего, через свёртку с ядром.
        """
        kernel_h = smooth_kernel.shape[0]
        kernel_w = smooth_kernel.shape[1]
        pad = (kernel_h // 2, kernel_h // 2, kernel_w // 2, kernel_w // 2)
        heatmap_padded = nn.functional.pad(points_heatmap, pad, "constant", 0)
        
        new_heatmap = torch.zeros_like(points_heatmap)
        for b in range(points_heatmap.shape[0]):
            for c in range(points_heatmap.shape[1]):
                for i in range(points_heatmap.shape[2]):
                    for j in range(points_heatmap.shape[3]):
                        new_heatmap[b][c][i][j] = torch.sum(
                            heatmap_padded[b][c][i:(i + kernel_h), j:(j + kernel_w)] * smooth_kernel
                        )
        return new_heatmap

    @staticmethod
    def _gaussian_2d(kernel_size: int) -> torch.Tensor:
        """
        Центрированная 2d-гауссиана с ядром размера kernel_size=2K+1
        Сигма устанавливается так, чтобы в ядро влезло 3 сигмы.
        """
        assert kernel_size % 2 == 1, kernel_size
        k = kernel_size // 2
        sigma = (k + 1) / 3
        x = torch.arange(kernel_size) - k
        gauss_1d = torch.exp(
            -x ** 2 / (2 * sigma**2)
        )
        gauss_2d = gauss_1d[None, :] * gauss_1d[:, None]
        gauss_2d /= gauss_2d.max()
        return gauss_2d

    @staticmethod
    def _put(tensor, indices, values):
        # Если индексы содержат повтор, то результат a[idx] = val неопределен (зависит от порядка)
        # Если использовать index_put_ с accumulate=False, то значения суммируются корректно.
        # "Фальшивые" детекции неизбежно дают повторы, поэтому вместо присваивания -
        # прибавляем значения
        # Значения для "фальшивых" детекций должны быть 0, иначе результат будет ошибочным
        tensor.index_put_(indices, values, accumulate=True)


class PointsToObjects(nn.Module):
    """
    Обратная конвертация из hetmap(points) в bounding boxes (objects).
    Необходимо извлечь из heatmap top-k объектов и "декодировать" их характеристики, т.е.
    кординаты (локация + поправки) и размеры.
    В тестах проверяется, что:
    x == PointsToObjects()(ObjectsToPoints()(x)), где x - объекты из датасета.
    """
    def __init__(self, objects_per_image: int = 42, min_confidence=0.1):
        super().__init__()
        self.top_k = objects_per_image
        self.min_conf = min_confidence
        self.objects_per_image = objects_per_image

    def forward(self, points_heatmap):
        hw = points_heatmap.shape[2]
        classes = points_heatmap.shape[1] - 4
        objects = torch.zeros(points_heatmap.shape[0], self.top_k, 6)
        for b in range(points_heatmap.shape[0]):
            topk_scores = []
            topk_inds = []
            topk_cls = []
            scores, inds = torch.topk(points_heatmap[b, 0:(classes)].flatten(), self.top_k)
            for i, s in enumerate(scores):
                if s > self.min_conf:
                    topk_scores.append(s)
                    index = inds[i]
                    topk_cls.append(index // (hw * hw))
                    index %= (hw * hw)
                    topk_inds.append((index // hw, index % hw))
                    
            topk_scores = torch.tensor(topk_scores)
            topk_inds = torch.tensor(topk_inds)
            topk_cls = torch.tensor(topk_cls)
            
            for i, s in enumerate(topk_scores):
                objects[b][i][0] = topk_inds[i][0] + points_heatmap[b][classes][topk_inds[i][0]][topk_inds[i][1]]
                objects[b][i][1] = topk_inds[i][1] + points_heatmap[b][classes + 1][topk_inds[i][0]][topk_inds[i][1]]
                objects[b][i][2] = points_heatmap[b][classes + 2][topk_inds[i][0]][topk_inds[i][1]]
                objects[b][i][3] = points_heatmap[b][classes + 3][topk_inds[i][0]][topk_inds[i][1]]
                objects[b][i][4] = topk_cls[i]
                objects[b][i][5] = s
                
        return objects

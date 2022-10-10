import numpy as np
from .base import BaseLayer


class CrossEntropyLoss(BaseLayer):
    """
    Слой-функция потерь, категориальная кросс-энтропия для задачи класификации на
    N классов.
    Применяет softmax к входным данных.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Принимает два тензора - предсказанные логиты классов и правильные классы.
        Prediction и target имеют одинаковый размер вида
         [B, C, 1, ... 1] т.е. имеют 2 оси (batch size и channels) длины больше 1
          и любое количество осей единичной длины.
        В predictions находятся логиты, т.е. к ним должен применяться softmax перед вычислением
         энтропии.
        В target[B, C, (1, ..., 1)] находится 1, если объект B принадлежит классу C, иначе 0 (one-hot представление).
        Возвращает np.array[B] c лоссом по каждому объекту в мини-батче.

        P[B, c] = exp(pred[B, c]) / Sum[c](exp(pred[B, c])
        Loss[B] = - Sum[c]log( prob[B, C] * target[B, C]) ) = -log(prob[B, C_correct])
        """
        
        loss = np.zeros(pred.shape[0])
        self.class_index = np.zeros(pred.shape[0]).astype(int)
        self.soft_max = np.zeros((pred.shape[0], pred.shape[1]))
        
        for b in range(pred.shape[0]):
            self.soft_max[b] = np.exp(pred[b]) / np.sum(np.exp(pred[b]))
            self.class_index[b] = np.squeeze(np.where(target[b] > 0))
            loss[b] = -np.log(self.soft_max[b][self.class_index[b]])
            
        return loss
        
    def backward(self) -> np.ndarray:
        """
        Возвращает градиент лосса по pred, т.е. первому аргументу .forward
        Не принимает никакого градиента по определению.
        """
        output_grads = np.zeros(self.soft_max.shape)
        for b in range(output_grads.shape[0]):
            output_grads[b] = self.soft_max[b]
            output_grads[b][self.class_index[b]] = self.soft_max[b][self.class_index[b]] - 1
            
        return output_grads
        

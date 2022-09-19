import numpy as np
from .base import BaseLayer


class ReluLayer(BaseLayer):
    """
    Слой, выполняющий Relu активацию y = max(x, 0).
    Не имеет параметров.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.mask = 1 * (input > 0) 
        return self.mask * input

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self.mask


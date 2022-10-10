import numpy as np
from .base import BaseLayer


class LinearLayer(BaseLayer):
    """
    Слой, выполняющий линейное преобразование y = x @ W.T + b.
    Параметры:
        parameters[0]: W;
        parameters[1]: b;
    Линейное преобразование выполняется для последней оси тензоров, т.е.
     y[B, ..., out_features] = LinearLayer(in_features, out_feautres)(x[B, ..., in_features].)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = 1 / np.sqrt(in_features)
        
        self.W = np.random.uniform(
            -std, std, size = [self.out_features, self.in_features]
            )
        self.b = np.random.uniform(-std, std, size = self.out_features)
        self.parameters = [self.W, self.b]
        
        self.W_grad = np.zeros([self.out_features, self.in_features])
        self.b_grad = np.zeros(self.out_features)
        self.parameters_grads = [self.W_grad, self.b_grad]

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        while len(self.input.shape) < 4:
            self.input = self.input[np.newaxis, ...] 
        return input @ self.parameters[0].T + self.parameters[1]

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.parameters_grads[0] = np.zeros((self.out_features, self.in_features))
        self.parameters_grads[1] = np.zeros(self.out_features)
        output_grad_dims = len(output_grad.shape)
        while len(output_grad.shape) < 4:
            output_grad = output_grad[np.newaxis, ...]
            
        for b in range(self.input.shape[0]):
            for c in range(self.input.shape[1]):
                self.parameters_grads[0] += output_grad[b][c].T @ self.input[b][c]
                self.parameters_grads[1] += np.sum(output_grad[b][c].T, axis = 1) 
        res_grad = output_grad @ self.parameters[0]
        
        for t in range(4 - output_grad_dims):
            res_grad = res_grad[0, ...]
        return res_grad

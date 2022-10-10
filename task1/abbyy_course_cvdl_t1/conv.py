import numpy as np
from .base import BaseLayer


class ConvLayer(BaseLayer):
    """
    Слой, выполняющий 2D кросс-корреляцию (с указанными ниже ограничениями).
    y[B, k, h, w] = Sum[i, j, c] (x[B, c, h+i, w+j] * w[k, c, i, j]) + b[k]

    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.
    В тестах input также всегда квадратный, и H==W всегда нечетные.
    К свертке входа с ядром всегда надо прибавлять тренируемый параметр-вектор (bias).
    Ядро свертки не разреженное (~ dilation=1).
    Значение stride всегда 1.
    Всегда используется padding='same', т.е. входной тензор необходимо дополнять нулями, и
     результат .forward(input) должен всегда иметь [H, W] размерность, равную
     размерности input.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        assert(in_channels > 0)
        assert(out_channels > 0)
        assert(kernel_size % 2 == 1)
        super().__init__()
        
        std = 1 / np.sqrt(in_channels)
        W = np.random.uniform(-std, std, size = [out_channels, in_channels, kernel_size, kernel_size])
        b = np.random.uniform(-std, std, size = out_channels)
        W_grad = np.zeros([out_channels, in_channels, kernel_size, kernel_size])
        b_grad = np.zeros(out_channels)
        self.parameters = [W, b]
        self.parameters_grads = [W_grad, b_grad]

    @property
    def kernel_size(self):
        return self.parameters[0].shape[-1]

    @property
    def out_channels(self):
        return self.parameters[0].shape[0]

    @property
    def in_channels(self):
        return self.parameters[0].shape[1]

    @staticmethod
    def _pad_zeros(self, tensor, one_side_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        offsets = np.zeros(tensor.ndim).astype(int)
        offsets[axis] = one_side_pad    
        insertion_mask = ([(offsets[d], offsets[d]) for d in range(tensor.ndim)])
        padded = np.pad(tensor, insertion_mask, 'constant', constant_values = 0)
        return padded

    @staticmethod
    def _cross_correlate(self, input, kernel):
        """
        Вычисляет "valid" кросс-корреляцию input[B, C_in, H, W]
        и kernel[C_out, C_in, X, Y].
        Метод не проверяется в тестах -- можно релизовать слой и без
        использования этого метода.
        """
        assert kernel.shape[-1] == kernel.shape[-2]
        assert kernel.shape[-1] % 2 == 1
        self.input = input
        res = np.zeros((input.shape[0], kernel.shape[0], 
                        input.shape[2] - self.parameters[0].shape[-1] + 1, 
                        input.shape[3] - self.parameters[0].shape[-1] + 1))
        # print('input.shape = ', input.shape)
        # print('kernel.shape = ', kernel.shape)
        for b in range(input.shape[0]):
            for c_o in range(kernel.shape[0]):
                for c_i in range(input.shape[1]):
                    for h_place in range(input.shape[2] - self.parameters[0].shape[-1] + 1):
                        for w_place in range(input.shape[3] - self.parameters[0].shape[-1] + 1):
                            # print('input[', b, ', ', c_i, ', (', 
                            #       h_place, '):(', h_place + self.parameters[0].shape[-1], '), (', 
                            #       w_place, '):(', w_place + self.parameters[0].shape[-1], ')])= ', 
                            #       input[b, c_i,
                            #       (h_place):(h_place + self.parameters[0].shape[-1]), 
                            #       (w_place):(w_place + self.parameters[0].shape[-1])])
                            
                            # print('kernel[c_o, c_i, :, :] = ', kernel[c_o, c_i, :, :])
                                
                            res[b, c_o, h_place, w_place] += np.sum(
                                input[b, c_i,
                                h_place:(h_place + self.parameters[0].shape[-1]), 
                                w_place:(w_place + self.parameters[0].shape[-1])
                                ] * kernel[c_o, c_i, :, :])
                res[b, c_o] += self.parameters[1][c_o]
                                
        return res

    def forward(self, input: np.ndarray) -> np.ndarray:
        padding_size = self.parameters[0].shape[-1] // 2
        input_padded = self._pad_zeros(self, input, padding_size)
        return self._cross_correlate(self, input_padded, self.parameters[0])

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        for c_o in range(self.parameters[0].shape[0]):
            self.parameters_grads[1][c_o] = np.sum(output_grad[:, c_o, :, :])
            
        padding_size = self.parameters[0].shape[-1] // 2
        output_grad_padded = self._pad_zeros(self, output_grad, padding_size)
        print('output_grad.shape = ', output_grad.shape)
        grad = np.zeros((output_grad.shape[0], self.parameters[0].shape[1], output_grad.shape[2], output_grad.shape[3]))
        
        w_changed = np.zeros((self.parameters[0].shape[-1], self.parameters[0].shape[-1]))
        
        for b in range(self.input.shape[0]):
            for c_o in range(self.parameters[0].shape[0]):
                for c_i in range(self.input.shape[1]):
                    for h_place in range(self.input.shape[2] - self.parameters[0].shape[-1] + 1):
                        for w_place in range(self.input.shape[3] - self.parameters[0].shape[-1] + 1):
                            for h_kernel in range(self.parameters[0].shape[-1]):
                                for w_kernel in range(self.parameters[0].shape[-1]):
                                    self.parameters_grads[0][c_o][c_i][h_kernel][w_kernel] += output_grad_padded[b, c_o, h_place + h_kernel, w_place + w_kernel]
                                    w_changed[h_kernel, w_kernel] = self.parameters[0][c_o][c_i][self.parameters[0].shape[-1] - 1 - h_kernel][self.parameters[0].shape[-1] - 1 - w_kernel]
                             
                             
                            grad[b, c_i, h_place, w_place] += np.sum(w_changed * output_grad_padded[b, c_o, h_place:(h_place + self.parameters[0].shape[-1]), w_place:(w_place + self.parameters[0].shape[-1])])
                            
                                    
        return grad

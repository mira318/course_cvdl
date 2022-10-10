import numpy as np
from .base import BaseLayer


class MaxPoolLayer(BaseLayer):
    """
    Слой, выполняющий 2D Max Pooling, т.е. выбор максимального значения в окне.
    y[B, c, h, w] = Max[i, j] (x[B, c, h+i, w+j])

    У слоя нет обучаемых параметров.
    Используется channel-first представление данных, т.е. тензоры имеют размер [B, C, H, W].
    Всегда ядро свертки квадратное, kernel_size имеет тип int. Значение kernel_size всегда нечетное.

    В качестве значений padding используется -np.inf, т.е. добавленые pad-значения используются исключительно
     для корректности индексов в любом положении, и никогда не могут реально оказаться максимумом в
     своем пуле.
    Гарантируется, что значения padding, stride и kernel_size будут такие, что
     в input + padding поместится целое число kernel, т.е.:
     (D + 2*padding - kernel_size)  % stride  == 0, где D - размерность H или W.

    Пример корректных значений:
    - kernel_size = 3
    - padding = 1
    - stride = 2
    - D = 7
    Результат:
    (Pool[-1:2], Pool[1:4], Pool[3:6], Pool[5:(7+1)])
    """
    def __init__(self, kernel_size: int, stride: int, padding: int):
        assert(kernel_size % 2 == 1)
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

    @staticmethod
    def _pad_neg_inf(self, tensor, one_size_pad, axis=[-1, -2]):
        """
        Добавляет одинаковый паддинг по осям, указанным в axis.
        Метод не проверяется в тестах -- можно релизовать слой без
        использования этого метода.
        """
        offsets = np.zeros(tensor.ndim).astype(int)
        offsets[axis] = one_size_pad    
        insertion_mask = ([(offsets[d], offsets[d]) for d in range(tensor.ndim)])
        padded = np.pad(tensor, insertion_mask, 'constant', constant_values = -np.inf)
        return padded

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert (input.shape[-1] + 2 * self.padding - self.kernel_size) % self.stride  == 0
        assert (input.shape[-2] + 2 * self.padding - self.kernel_size) % self.stride  == 0
        self.input_shape = input.shape
        input = self._pad_neg_inf(self, input, self.padding)
        res_shape = [input.shape[0], input.shape[1], 
                     int((input.shape[2] - self.kernel_size + self.stride) / self.stride),
                     int((input.shape[3] - self.kernel_size + self.stride) / self.stride)]
        
        res = np.zeros(res_shape)
        self.max_list = []
        
        for b in range(res_shape[0]):
            for c in range(res_shape[1]):
                for h_m in range(res_shape[2]):
                    inp_h = h_m * self.stride
                    for w_m in range(res_shape[3]):
                        inp_w = w_m * self.stride
                        arg = np.argmax(
                            input[b, c, inp_h:(inp_h + self.kernel_size), inp_w:(inp_w + self.kernel_size)]
                        )
                        self.max_list.append((b, c, inp_h + arg // self.kernel_size - self.padding, 
                                                    inp_w + arg % self.kernel_size - self.padding))
                        res[b][c][h_m][w_m] = np.max(
                            input[b, c, inp_h:(inp_h + self.kernel_size), inp_w:(inp_w + self.kernel_size)]
                        )
        print('max_list = ', self.max_list)
        return res

    def backward(self, output_grad: np.ndarray)->np.ndarray:
        print('output_grad = ', output_grad)
        grad = np.zeros(self.input_shape)
        list_index = 0
        for b in range(output_grad.shape[0]):
            for c in range(output_grad.shape[1]):
                for h_out in range(output_grad.shape[2]):
                    for w_out in range(output_grad.shape[3]):
                        grad[self.max_list[list_index]] += output_grad[b][c][h_out][w_out] 
                        list_index += 1
        print('gard = ', grad)
        return grad

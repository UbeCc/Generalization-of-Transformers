import math
import torch
from utils import SE, MSE

def get_task_sampler(
        task_name, 
        n_dims, 
        batch_size,
        k,
        is_void=False,
        **kwargs
):
    task_names_to_classes = {
        "constant": Constant,
        "linear": Linear,
        "square": Square,
        "cube": Cube,
        "biquadrate": Biquadrate,
        "sin": Sin,
        "cos": Cos,
        "exp": Exp,
        'arccos': Arccos,
        'inverse': Inverse,
    }
    assert task_name in task_names_to_classes, f"Unknown task: {task_name}"
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        return lambda **args: task_cls(n_dims, batch_size, k, is_void, **args, **kwargs)
    else:
        print(f"Unknown task: {task_name}")
        raise NotImplementedError

class Task:
    def __init__(self, n_dims, batch_size, k, is_void=False):
        self.n_dims = n_dims
        self.bsize = batch_size
        self.w = torch.randn(self.bsize, self.n_dims, 1)
        self.w = torch.clip(self.w, -1, 1)
        if is_void:
            self.w = torch.full((self.bsize, self.n_dims, 1), 1)
        self.k = k

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train', **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return SE

    @staticmethod
    def get_training_metric():
        return MSE
    
    def get_weight(self, use_local_weight, device):
        if not use_local_weight:
            self.w = torch.full(self.w.shape, 1, device=device)
        return self.w.to(device)
    
class Constant(Task):
    def __init__(self, n_dims, batch_size):
        super(Constant, self).__init__(n_dims, batch_size)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        return scale * self.w[:, :, 0].expand(-1, xs.shape[1]) # xs.shape[1] = 40 for our experiments
    
class Linear(Task):
    def __init__(self, n_dims, batch_size, k, is_void=False):
        super(Linear, self).__init__(n_dims, batch_size, k, is_void)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        device = xs.device if type(xs) == torch.Tensor else 'cpu'
        w = self.get_weight(use_local_weight=not (mode == 'test' and train_type == 'comp'), device=device)
        xs = xs.float()
        if datatype != 'legendre':
            ys = (xs @ torch.abs(w))[:, :, 0]
        else:
            # y_{\text{(1)}}^{\text{l}} = \frac{\sqrt{30}}{50} \cdot \mathbf{x} \cdot |\mathbf{w}|
            ys = (math.sqrt(30) / 50.0) * (xs @ torch.abs(w))[:, :, 0]
        return scale * ys

class Square(Task):
    def __init__(self, n_dims, batch_size, k, is_void=False):
        super(Square, self).__init__(n_dims, batch_size, k, is_void)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        device = xs.device if type(xs) == torch.Tensor else 'cpu'
        w = self.get_weight(use_local_weight=not (mode == 'test' and train_type == 'comp'), device=device)
        xs = xs.float()
        if datatype != 'legendre':
            ys = (xs * xs @ torch.abs(w))[:, :, 0]
        else:
            # y_{\text{(2)}}^{\text{l}} = \frac{\sqrt{2}}{50} \cdot \left( 3 \mathbf{x}^2 - 25 \right) \cdot \mathbf{w}
            ys = ((math.sqrt(2.0) / 50.0) * ((3.0 * xs * xs) - 25) @ w)[:, :, 0]
        return scale * ys
    
class Cube(Task):
    def __init__(self, n_dims, batch_size, k, is_void=False):
        super(Cube, self).__init__(n_dims, batch_size, k, is_void)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        device = xs.device if type(xs) == torch.Tensor else 'cpu'
        w = self.get_weight(use_local_weight=not (mode == 'test' and train_type == 'comp'), device=device)
        xs = xs.float()
        if datatype != 'legendre':
            ys = (xs * xs * xs @ w)[:, :, 0]
        else:
            # y_{\text{(3)}}^{\text{l}} = \frac{\sqrt{70}}{500} \cdot \left( \mathbf{x}^3 - 15 \mathbf{x} \right) \cdot \mathbf{w}
            ys = math.sqrt(70) / 500.0 * ((xs * xs * xs - 15.0 * xs) @ w)[:, :, 0]
        return scale * ys

class Biquadrate(Task):
    def __init__(self, n_dims, batch_size, k, is_void=False):
        super(Biquadrate, self).__init__(n_dims, batch_size, k, is_void)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        device = xs.device if type(xs) == torch.Tensor else 'cpu'
        w = self.get_weight(use_local_weight=not (mode == 'test' and train_type == 'comp'), device=device)
        xs = xs.float()
        if datatype != 'legendre':
            ys = (xs * xs * xs * xs @ w)[:, :, 0]
        else:
            # y_{\text{(4)}}^{\text{l}} = \frac{3 \sqrt{10}}{10000} \cdot \left( 7 \mathbf{x}^4 - 150 \mathbf{x}^2 + 375 \right) \cdot \mathbf{w}
            ys = (3.0 * float(math.sqrt(10)) / 10000.0) * ((7 * (xs * xs * xs * xs) \
                - 150 * (xs * xs) + 375 * torch.ones_like(xs)) @ w)[:, :, 0]
        return scale * ys
    
class Sin(Task):
    def __init__(self, n_dims, batch_size, k, is_void=False):
        super(Sin, self).__init__(n_dims, batch_size, k, is_void)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        device = xs.device if type(xs) == torch.Tensor else 'cpu'
        w = self.get_weight(use_local_weight=not (mode == 'test' and train_type == 'comp'), device=device)
        xs = xs.float()
        if bias: 
            ys = (torch.sin(self.k * xs) @ w)[:, :, 0] + bias
        else:
            ys = -math.pow(-1, int(self.k)) * (torch.sin(self.k * xs.float()) @ torch.abs(w.float()))[:, :, 0]
        return scale * ys

class Cos(Task):
    def __init__(self, n_dims, batch_size, k, is_void=False):
        super(Cos, self).__init__(n_dims, batch_size, k, is_void)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        device = xs.device if type(xs) == torch.Tensor else 'cpu'
        w = self.get_weight(use_local_weight=not (mode == 'test' and train_type == 'comp'), device=device)
        xs = xs.float()
        if bias:
            ys = (torch.cos(self.k * xs) @ w.float())[:, :, 0] + bias
        else:
            ys = -math.pow(-1, int(self.k)) * (torch.cos(self.k * xs.float()) @ torch.abs(w.float()))[:, :, 0]
        return scale * ys
    
class Exp(Task):
    def __init__(self, n_dims, batch_size, k, is_void=False):
        super(Exp, self).__init__(n_dims, batch_size, k, is_void)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        device = xs.device if type(xs) == torch.Tensor else 'cpu'
        if mode == 'test' and train_type == 'comp':
            w = torch.full(self.w.shape, 1, device=device)
            self.w = torch.full(self.w.shape, 1, device=device)
        else:
            w = self.w.to(device)
        xs = xs.float()
        w = w.float()
        ys = (torch.exp(xs) @ w)[:, :, 0]
        # biased version:
        # quarters = 2.0 / (math.e ** 2 - 1.0 / (math.e ** 2))
        # mean = (math.e ** 2 + 1.0 / (math.e ** 2)) / (1.0 * (math.e ** 2 - 1.0 / (math.e ** 2)))
        # ys = (ys * quarters - mean * torch.ones_like(ys))
        return scale * ys

class Arccos(Task):
    def __init__(self, n_dims, batch_size, k, is_void=False):
        super(Arccos, self).__init__(n_dims, batch_size, k, is_void)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        device = xs.device if type(xs) == torch.Tensor else 'cpu'
        w = self.get_weight(use_local_weight=not (mode == 'test' and train_type == 'comp'), device=device)
        xs = xs.float()
        ys = (torch.arccos(xs / 5) @ torch.abs(w))[:, :, 0]
        return scale * ys

class Inverse(Task):
    def __init__(self, n_dims, batch_size, k, is_void=False):
        super(Inverse, self).__init__(n_dims, batch_size, k, is_void)

    def evaluate(self, xs, datatype='basic', scale=1, bias=0, train_type='default', mode='train'):
        if type(xs) != torch.Tensor and len(xs) == 1:
            xs = torch.tensor(xs[0]).unsqueeze(0).unsqueeze(0)
        device = xs.device if type(xs) == torch.Tensor else 'cpu'
        w = self.get_weight(use_local_weight=not (mode == 'test' and train_type == 'comp'), device=device)
        xs = xs.float()
        ys = (1 / (torch.abs(xs) + 0.5 * torch.ones_like(xs)) @ torch.abs(w))[:, :, 0]
        return scale * ys
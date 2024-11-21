import numpy as np
from scipy.optimize import minimize
import random
import torch
import os
from munch import Munch
import torch
import yaml
import models

def SE(ys_pred, ys_ref):
    return (ys_ref - ys_pred).square()

def MSE(ys_pred, ys_ref):
    return (ys_ref - ys_pred).square().mean()

def ACC(ys_pred, ys_ref):
    return (ys_ref == ys_pred.sign()).float()

sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()
def CE(ys_pred, ys_ref):
    output = sigmoid(ys_pred)
    target = (ys_ref + 1) / 2
    return bce_loss(output, target)

def get_model_from_run(run_path, step=-1, model_name='state.pt', only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if step == -1:
        state_path = os.path.join(run_path, model_name)
        state = torch.load(state_path,map_location=device)
        try:
            model.load_state_dict(state["model_state_dict"])
        except:
            model.load_state_dict(state)
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path,map_location=device)
        model.load_state_dict(state_dict)

    return model, conf

class GaussianSampler():
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__()
        self.n_dims = n_dims
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, bsize, datatype, n_dims_truncated=None, limit=1,
                  seeds=None, ood=False, order='pre', iid_count=30, ood_count=10):
        if seeds is None:
            if datatype == 'gaussian':
                xs_b = torch.randn(bsize, n_points, self.n_dims)  # x~N(0,1)
                torch.nn.init.trunc_normal_(xs_b, mean=0, std=1, a=-5, b=5)
                return xs_b
            elif datatype != "uniform":
                raise NotImplementedError

            # x ~ [-limit, limit]
            xs_b = torch.rand(size=(bsize, n_points, self.n_dims), dtype=torch.float32) * 2 * limit \
                    - limit * torch.ones(size=(bsize, n_points, self.n_dims), dtype=torch.float32)

            if ood:
                for i in range(ood_count):
                    for j in range(bsize):
                        ood_x_neg = torch.rand(size=(1,), dtype=torch.float32).item() * limit - 2 * limit
                        ood_x_pos = torch.rand(size=(1,), dtype=torch.float32).item() * limit + limit
                        if random.randint(0, 1):
                            if order == 'pre':
                                xs_b[j][i] = ood_x_neg
                            else:
                                # 0 ~ iid_count - 1, iid_count ~ iid_count + ood_count - 1
                                xs_b[j][i + iid_count] = ood_x_neg
                        else:
                            if order == 'pre':
                                xs_b[j][i] = ood_x_pos
                            else:
                                xs_b[j][i + iid_count] = ood_x_pos
        else:
            xs_b = torch.zeros(bsize, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == bsize
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        
        return xs_b
    
class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_points = args.points.start
        self.n_dims_schedule = args.dims
        self.n_points_schedule = args.points
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc
        return min(var, schedule.end)
    
def get_weight(task):
    if not '+' in task:
        return 1
    try:
        if task.split('+')[1] == 'pi':
            w = torch.pi
        else:
            w = int(task.split('+')[1])
    except Exception as e:
        print(f'Error when parsing weight: {task}')
        w = 1
    return w

def replace_last_occurrence(original_str, old, new):
    last_index = original_str.rfind(old)
    if last_index != -1:
        new_str = (
            original_str[:last_index] + 
            original_str[last_index:].replace(old, new, 1)
        )
        return new_str
    else:
        return original_str
    
def get_limit(raw_limit):
    if raw_limit == 'pi':
        return torch.pi
    else:
        try:
            return float(raw_limit)
        except Exception as e:
            print(f'Error when parsing limit: {e}')
            raise e

def find_extrema(func, bsize, n_points, limit, n_grid=1000):
    bounds = [-limit, limit]
    
    x_grid = torch.linspace(bounds[0], bounds[1], n_grid)
    y_grid = []
    for x in x_grid:
        cur_x = x.unsqueeze(0).unsqueeze(0)
        cur_y = func(cur_x)
        y_grid.append(cur_y.item())
    x_init_max = x_grid[np.argmax(y_grid)]
    x_init_min = x_grid[np.argmin(y_grid)]
    max_result = minimize(lambda x: -func(x), x0=x_init_max, bounds=[bounds], method='L-BFGS-B')
    min_result = minimize(lambda x: func(x), x0=x_init_min, bounds=[bounds], method='L-BFGS-B')
    return -max_result.fun, min_result.fun

    # testcase
    # def f(x):
    #     if isinstance(x, np.ndarray):
    #         return np.sin(x) + np.cos(2*x)
    #     return np.sin(x) + np.cos(2*x)
    # bounds = (0, 2*np.pi)
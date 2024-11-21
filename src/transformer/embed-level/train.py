import os
from random import randint
import random
from tqdm import tqdm
import torch
import yaml
import wandb
from tasks import get_task_sampler
from models import build_model
from omegaconf import DictConfig, OmegaConf
from utils import *
import hydra

torch.backends.cudnn.benchmark = True

def step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def generate_task(
        task_sampler_list, # function classes in python
        task_list, # descriptions for the functions
        xs, # input data
        train_task, # task type: ['baseline', 'combination']
        train_type, # training type: ['legendre', 'basic', 'sinusoidal']
        task_type, # task type: ['add', 'mul', 'comp']
        xs_datatype,
        bias=0, # bias, only for sinusoidal function
        comp_count=1, # number of combination functions
        comp_sampler_list=None, # description for the combination functions, pairwise
        scaleup=1,
    ):
    # We assign indexs to each function. If the index >= number of functions, we assume it is a combination of two functions. 
    if train_type == 'baseline':
        func_idxs = len(task_list) - 1 + comp_count
    elif train_type == 'combination':
        func_idxs = len(task_list) + comp_count # The last index is reserved for the combination of two functions
    else:
        raise NotImplementedError
    
    # For convex combination (addition), we keep the ratio between 1 / 3 ~ 3
    # x / (x + x * r / 3) => 1 / (1 + 1 * r / 3) => 3 / (3 + r)
    # For multiplication and composition, we keep the ratio as 1
    func_idx = random.randint(0, func_idxs)
    r = random.randint(1, 9)
    if func_idx < len(task_list): # base function
        task_sampler = task_sampler_list[func_idx]
        if train_type == 'combination' or random.random() > 0.2:
            scaleup = 1
        ys = scaleup * task_sampler.evaluate(xs=xs, bias=bias)
    else: # combination
        comp_idx = func_idx - len(task_list) - 1
        task_sampler1, task_sampler2 = comp_sampler_list[2 * comp_idx], comp_sampler_list[2 * comp_idx + 1]
        if task_type == 'add':
            ys = 3 / (3 + r) * task_sampler1.evaluate(xs=xs, datatype=train_task, bias=bias) \
                + r / (3 + r) * task_sampler2.evaluate(xs=xs, datatype=train_task, bias=bias)
        elif task_type == 'mul':
            ys = task_sampler1.evaluate(xs=xs, datatype=train_task, bias=bias) \
                * task_sampler2.evaluate(xs=xs,datatype=train_task, bias=bias)
        elif task_type == "comp":
            ys = task_sampler1.evaluate(xs=task_sampler2.evaluate(xs=xs, datatype=train_task, bias=bias)
                                        .unsqueeze(-1), datatype=train_task, bias=bias)
    return ys

def train(model, args):
    device = f"cuda:{args.train.gpu}"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.learning_rate)
    curriculum = Curriculum(args.train.curriculum)
    limit = get_limit(args.train.limit)
    start_step = 0
    state_path = os.path.join(args.output_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    task_kwargs = args.train.task_kwargs
    task_list = args.train.train_list
    n_dims = model.n_dims
    bsize = args.train.batch_size
    data_sampler = GaussianSampler(n_dims)
    task_sampler_list = [
        get_task_sampler(
            task.split('+')[0], 
            n_dims,
            bsize,
            k=get_weight(task.split('+')[1]) if '+' in task else 1,
            **task_kwargs
    )(**task_kwargs) for task in task_list]
                                          
    if args.train.train_type == 'combination':
        comp_sampler_list = [
            get_task_sampler(
                task.split('+')[0], 
                n_dims,
                bsize,
                k=get_weight(task.split('+')[1]) if '+' in task else 1,
                **args.train.task_kwargs
        )(**task_kwargs) for task in args.train.comp_funcs]
    else:
        comp_sampler_list = None

    train_steps = args.train.train_steps
    pbar = tqdm(range(start_step, train_steps))
    
    void_task_sampler_list = [
        get_task_sampler(
            task.split('+')[0], 
            1,
            1,
            k=get_weight(task.split('+')[1]) if '+' in task else 1,
            is_void=True,
            **task_kwargs
    )(**task_kwargs) for task in task_list]
    M, M_prod = -1, 1
    for task_sampler in void_task_sampler_list:
        minv, maxv = find_extrema(task_sampler.evaluate, 1, 1, limit)
        M_i = max(abs(minv), abs(maxv))
        M_prod *= M_i
        M = max(M, M_i)
    if args.train.task_type == 'add':
        scaleup = len(task_sampler_list) * M
    elif args.train.task_type == 'mul':
        scaleup = M_prod / M
    else:
        scaleup = 1

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if train_steps is not None:
            assert train_steps >= bsize, "train_steps must be greater than or equal to batch size"
            seeds = sample_seeds(train_steps, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points, 
            bsize, 
            args.train.xs_datatype, 
            curriculum.n_dims_truncated, 
            limit,
            **data_sampler_args
        )

        ys = generate_task(
            task_sampler_list=task_sampler_list,
            task_list=args.train.train_list,
            xs=xs,
            train_task=args.train.task,
            train_type=args.train.train_type,
            task_type=args.train.task_type,
            xs_datatype=args.train.xs_datatype,
            bias=args.train.bias,
            comp_count=int(args.train.comp_count),
            comp_sampler_list=comp_sampler_list,
            scaleup=scaleup
        )

        loss_func = MSE
        loss, output = step(
            model, 
            xs.to(device), 
            ys.to(device), 
            optimizer, 
            loss_func
        )

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = SE
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - i, 0) 
                for i in range(curriculum.n_points)
            ) / curriculum.n_points
        )

        if args.train.use_wandb and i % args.wandb.log_every_steps == 0:
            wandb.log({
                "overall_loss": loss,
                "excess_loss": loss / baseline_loss,
                "pointwise/loss": dict(zip(point_wise_tags, point_wise_loss.cpu().numpy())),
                "n_points": curriculum.n_points,
                "n_dims": curriculum.n_dims_truncated,
            }, step=i)

        curriculum.update()
        pbar.set_description(f"loss {loss}")

        if i % args.train.save_every_steps == 0:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(model.state_dict(), state_path)

        if args.train.keep_every_steps > 0 and i % args.train.keep_every_steps == 0 and i > 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_{i}.pt"))

def main(args: DictConfig):
    if args.train.use_wandb:
        wandb.init(
            name=args.wandb.name,
            project=args.wandb.project,
            config=OmegaConf.to_container(args, resolve=True),
            resume=True,
        )
    model = build_model(args.model)
    device = f"cuda:{args.train.gpu}"
    print(f"Moving model to device: {device}")
    model.to(device)
    model.train()
    train(model, args)

@hydra.main(config_path="conf", config_name="base.yaml")
def hydra_main(args: DictConfig):
    print(f"Running with config: {OmegaConf.to_yaml(args)}")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    run_function = '_'.join(args.train.train_list)
    run_type = args.train.train_type
    output_dir = os.path.join(args.output_dir, run_function, run_type)
    os.makedirs(output_dir, exist_ok=True)

    args.output_dir = output_dir
    with open(os.path.join(output_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(OmegaConf.to_container(args, resolve=True), yaml_file, default_flow_style=False)
    main(args)

if __name__ == "__main__":
    hydra_main()
    
# python train.py --config-name conf/config.yaml
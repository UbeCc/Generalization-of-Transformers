import yaml

run_name = 'kan-add-sin1-cos1-bl'
gpu = '0'
train_list = ['sin+1', 'sin+2', 'cos+1', 'cos+2']
comp_count = 1
comp_funcs = ['sin+1', 'sin+2'] # pairwise, (idx*2, idx*2+1) is the same pair
train_type = 'combination' # [baseline, combination]
task = 'sinusoidal' # [legendre, basic, sinusoidal]
task_type = 'add' # [add, mul, comp]
xs_datatype = 'gaussian' # [gaussian, uniform]
limit = 'pi'
bias = 0 # type: int
batch_size = 256
learning_rate = 0.0001
use_wandb = False
train_steps = 50001
family = 'kan'
project = 'generalization-of-transformers'
entity = 'anonymous'
n_embd = 256
n_layer = 12
n_head = 8
n_dims = 1
n_positions = 40

config = {
    'output_dir': f'results/{run_name}',
    'train': {
        'gpu': gpu,
        'train_list': train_list,
        'comp_count': comp_count,
        'comp_funcs': comp_funcs,
        'task': task,
        'train_type': train_type,
        'task_type': task_type,
        'xs_datatype': xs_datatype,
        'limit': limit,
        'bias': bias,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'use_wandb': use_wandb,
        'train_steps': train_steps,
        'save_every_steps': 10000,
        'keep_every_steps': 10000,
        'task_kwargs': {},
        'curriculum': {
            'dims': {
                'start': 1,
                'end': 1,
                'inc': 1,
                'interval': 2000,
            },
            'points': {
                'start': 40,
                'end': 40,
                'inc': 2,
                'interval': 2000,
            },
        },
    },
    'wandb': {
        'project': project,
        'entity': entity,
        'name': run_name,
        'log_every_steps': 100,
    },
    'model': {
        'family': family,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_dims': n_dims,        
        'n_positions': n_positions,
    },
}

with open(f'{run_name}.yaml', 'w') as file:
    yaml.dump(config, file)
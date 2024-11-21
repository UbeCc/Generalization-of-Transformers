import matplotlib.ticker as mticker
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import math
from tqdm import trange
import pandas as pd
from scipy.interpolate import interp1d
from utils import *
from tasks import get_task_sampler
import random

# set the style of the plot
sns.set_theme(style='darkgrid')
palette = sns.color_palette('colorblind')
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'Times New Roman'

FUNC_DICT = {
    'linear': '$x$',
    'square': '$x^2$',
    'cube': '$x^3$',
    'biquadrate': '$x^4$',
    'arccos': '$\\arccos(x)$',
    'inverse': '$\\frac{1}{x}$',
    'abs': '$|x|$',
}

def get_rs(n):
    return [i.item() for i in list(torch.FloatTensor(n).uniform_(1, 9))]

def batchMSE(ys_pred, ys, batch_size):
    res = []
    for i in range(batch_size):
        s = ys[i].square().sum()
        res.append([(val / s) for _, val in enumerate((ys_pred[i] - ys[i]) ** 2)])
    return np.array(res).mean(axis=0)

def batchSTD(ys_pred, ys, batch_size):
    res = []
    for i in range(batch_size):
        s = ys[i].square().sum()
        res.append([(val / s) for _, val in enumerate((ys_pred[i] - ys[i]) ** 2)])
    return np.array(res).std(axis=0)

def get_task(task_name, task_sampler_list, task_list, xs, rs, datatype="basic", bias=0):
    s = 0
    if task_name == "add" or task_name == 'comp':
        ys = 0
    elif task_name == 'mul':
        ys = 1
    else:
        raise NotImplementedError(f"Task {task_name} not implemented")
    for i in range(len(task_list)):
        s += rs[i]
    for i in range(len(task_list)):
        r = rs[i] / s
        if task_name == 'add' or task_name == 'mul':
            ys += task_sampler_list[i].evaluate(xs, scale=r, datatype=datatype, bias=bias, mode='test', train_type='default')
        elif task_name == 'comp':
            ys = task_sampler_list[0].evaluate(
                task_sampler_list[1].evaluate(xs, datatype=datatype, bias=bias, train_type='comp', mode='test').unsqueeze(-1), 
                bias=bias, train_type='comp', mode='test'
            )
    return ys, rs

def get_title(test_funcs, is_se, axis=0, points=0, task_sampler_list=None):
    ret = 'Test: Com of '
    if len(test_funcs) == 4:
        ret += f'all base functions'
    else:
        for (i, func_name) in enumerate(test_funcs):
            if func_name in FUNC_DICT:
                func_name = FUNC_DICT[func_name]
            elif 'sin' in func_name or 'cos' in func_name:
                segments = func_name.split('+')
                if segments[1] == '1':
                    func_name = f'{segments[0]}(x)'
                elif segments[1] == 'pi':
                    func_name = f'{segments[0]}(3.14x)'
                else:
                    func_name = f'{segments[0]}({segments[1]}x)'
            if i == 0:
                ret += func_name
            else:
                ret += f', {func_name}'
    if is_se:
        return ret

    # draw params
    ret += f'\nWeights: '
    for (i, task) in enumerate(task_sampler_list):
        weight = task.w[axis].item()
        if i == 0:
            ret += f'{weight:.2f}'
        else:
            ret += f', {weight:.2f}'
    return ret

def get_x_generator(limit, round):
    def generator():
        for i in np.linspace(-limit, limit, round):
            yield torch.tensor(i)
    return generator

def generator_xs(
    distribution='gaussian', 
    batch_size=1024, 
    n_points=40, 
    n_dims=1,
    limit=1,
    ood=False, 
    order='pre', 
    iid_count=30
):
    return GaussianSampler(n_dims).sample_xs(
        bsize=batch_size, 
        n_points=n_points, 
        datatype=distribution,
        limit=limit,
        ood=ood, 
        order=order, 
        iid_count=iid_count
    )

def eval_without_repeat(
    run_dir, functions, batch_size, 
    only_state=True, train_info='default', given_func_pool=None,
):
    for train_data in os.listdir(os.path.join(run_dir, functions)):
        if 'baseline' in train_data:
            continue
        run_path = os.path.join(run_dir, functions, train_data)
        model, conf = get_model_from_run(run_path, step=-1)
        limit = get_limit(conf.train.limit)
        test_func_pool = []
        function_pool = conf.train.train_list

        for num1 in range(len(function_pool)):
            for num2 in range(len(function_pool)):
                if num1 == num2:
                    continue
                if num1 > num2:
                    continue
                test_funcs = [function_pool[num1], function_pool[num2]]
                test_func_pool.append(test_funcs)
        for num in range(len(function_pool)):
            test_func_pool.append([function_pool[num]])
        test_func_pool.append(function_pool)

        if given_func_pool:
            test_func_pool = given_func_pool

        assert given_func_pool
        states = []
        for name in os.listdir(run_path):
            if '.pt' in name:
                states.append(name)

        for state in states:
            if only_state and 'state' not in state:
                continue
            for test_funcs in test_func_pool:
                prefix = f"images/{functions}/{train_info}/{train_data}/{state[:-3]}"
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                rep = replace_last_occurrence(run_path, train_data, "baseline")
                model, conf = get_model_from_run(run_path, step=30000, model_name=state)
                model_baseline, _ = get_model_from_run(rep, step=30000, model_name=state)
                n_dims = conf.model.n_dims

                task_sampler_list = [get_task_sampler(
                    task.split('+')[0], 
                    n_dims,
                    batch_size,
                    k=get_weight(task.split('+')[1]) if '+' in task else 1,
                    **conf.train.task_kwargs)(**conf.train.task_kwargs) 
                for task in test_funcs]

                xs = generator_xs(
                    batch_size=batch_size, 
                    distribution=conf.train.xs_datatype,
                    limit=limit
                )

                cur_rs = get_rs(len(test_funcs))
                ys, _ = get_task(
                    conf.train.task_type, 
                    task_sampler_list, 
                    test_funcs, 
                    xs, rs=cur_rs,
                    datatype=conf.train.task, 
                    bias=conf.train.bias)
                
                with torch.no_grad():
                    pred = model(xs, ys)
                with torch.no_grad():
                    pred_baseline = model_baseline(xs, ys)
                loss = batchMSE(pred, ys, batch_size)
                loss_baseline = batchMSE(pred_baseline, ys, batch_size)
                std = batchSTD(pred, ys, batch_size)
                std_baseline = batchSTD(pred_baseline, ys, batch_size)

                x_range = np.arange(0, 40).ravel()
                x_smooth = np.linspace(0, 39, 500)

                # begin drawing legend
                # plt.clf()
                # plt.plot(loss, lw=2, color='red', label="ComFuncLearner")
                # plt.fill_between(
                #     x_smooth, 
                #     interp1d(x_range, (loss - std), kind='cubic')(x_smooth),
                #     interp1d(x_range, (loss + std), kind='cubic')(x_smooth), 
                #     color='red', 
                #     alpha=0.1
                # )
                # plt.plot(loss_baseline, lw=2, color='blue', label="Baseline")
                # plt.fill_between(
                #     x_smooth, 
                #     interp1d(x_range, (loss_baseline - std_baseline), kind='cubic')(x_smooth), 
                #     interp1d(x_range, (loss_baseline + std_baseline), kind='cubic')(x_smooth), 
                #     color='blue', 
                #     alpha=0.1
                # )
                # handles, labels = plt.gca().get_legend_handles_labels()
                # fig_legend = plt.figure(figsize=(4, 2))
                # ax_legend = fig_legend.add_subplot(111)
                # ax_legend.legend(handles, labels, loc='center', frameon=True, fancybox=True, edgecolor='black', framealpha=0.8)
                # # plt.gca().legend(lines, labels, loc='right', ncol=1)
                # ax_legend.axis('off')
                # fig_legend.savefig(f"{prefix}/se/legend_se.png", bbox_inches='tight')
                # plt.close(fig_legend)
                # end drawing legend

                plt.clf()
                plt.plot(loss, lw=2, color='red', label="ComFuncLearner")
                plt.fill_between(
                    x_smooth, 
                    interp1d(x_range, (loss - std), kind='cubic')(x_smooth),
                    interp1d(x_range, (loss + std), kind='cubic')(x_smooth), 
                    color='red', 
                    alpha=0.1
                )
                plt.plot(loss_baseline, lw=2, color='blue', label="Baseline")
                plt.fill_between(
                    x_smooth, 
                    interp1d(x_range, (loss_baseline - std_baseline), kind='cubic')(x_smooth), 
                    interp1d(x_range, (loss_baseline + std_baseline), kind='cubic')(x_smooth), 
                    color='blue', 
                    alpha=0.1
                )
                xticks = []
                for tcs in range(len(loss)):
                    xticks.append(tcs)
                plt.xticks(np.arange(0, len(xticks), 4), xticks[::4])
                plt.ylim(0, 2 * loss_baseline[5:].mean())
                plt.xlabel("# In-Context Examples")
                plt.ylabel("Squared Error")
                plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
                plt.gca().yaxis.get_offset_text().set_fontsize(15)
                lines, labels = plt.gca().get_legend_handles_labels()
                plt.subplots_adjust(bottom=0.13, left=0.12, right=0.98)
                plt.subplots_adjust(hspace=0.5, wspace=0.15)
                plt.subplots_adjust(top=0.85)
                
                matplotlib.rcParams['text.usetex'] = True  # latex rendering for title
                plt.suptitle(f'{get_title(test_funcs=test_funcs, is_se=True)}', wrap=True, fontsize=23, y=0.96, fontname='Times New Roman', weight='regular')
                matplotlib.rcParams['text.usetex'] = False

                if not os.path.exists(f"{prefix}/se"):
                    os.makedirs(f"{prefix}/se")

                def save_csv(loss_CC, loss_baseline, std_CC, std_baseline):
                    df = pd.DataFrame({'CC': loss_CC, 'baseline': loss_baseline, 'std_CC': std_CC, 'std_bl': std_baseline})
                    df.to_csv(f'{prefix}/se/csv_{functions}_trained_on_{train_data}_tested_on_{test_funcs}.csv', index=True)
                save_csv(loss, loss_baseline, std, std_baseline)
                plt.savefig(f"{prefix}/se/se_{functions}_trained_on_{train_data}_tested_on_{test_funcs}.png", dpi=300)

                # begin drawing legend
                plt.clf()
                axis = random.randint(0, batch_size - 1)
                plt.scatter(xs.squeeze(-1).numpy()[axis][:], ys.numpy()[axis][:], s=50, marker='o',
                            facecolors='none', color='black', label='GT')
                matplotlib.rcParams['text.usetex'] = True
                plt.scatter(xs.squeeze(-1).numpy()[axis][:], pred.numpy()[axis][:], s=20, marker='s',
                            color='blue', label=r'ComFuncLearner')
                            # color='blue', label=r'w/o extra basis func')
                plt.scatter(xs.squeeze(-1).numpy()[axis][:], pred_baseline.numpy()[axis][:], s=20, marker='^',
                            color='red', label=r'Baseline')
                            # color='red', label=r'w/ extra basis func')
                matplotlib.rcParams['text.usetex'] = False
                handles, labels = plt.gca().get_legend_handles_labels()
                fig_legend = plt.figure(figsize=(4, 2))
                ax_legend = fig_legend.add_subplot(111)
                ax_legend.legend(handles, labels, loc='center', frameon=True, fancybox=True, edgecolor='black', framealpha=0.8, ncol=3)
                ax_legend.axis('off')
                fig_legend.savefig(f"{prefix}/curves/legend_curve.png", bbox_inches='tight', dpi=300)
                plt.close(fig_legend)
                # end legend drawing

                # draw curve
                plt.clf()
                axis = random.randint(0, batch_size - 1)
                plt.scatter(xs.squeeze(-1).numpy()[axis][:], ys.numpy()[axis][:], s=50, marker='o',
                            facecolors='none', color='black', label='GT')
                plt.scatter(xs.squeeze(-1).numpy()[axis][:], pred.numpy()[axis][:], s=20, marker='s',
                            color='blue', label='ComFuncLearner')
                plt.scatter(xs.squeeze(-1).numpy()[axis][:], pred_baseline.numpy()[axis][:], s=20, marker='^',
                            color='red', label='Baseline')
                plt.xlabel("x")
                plt.ylabel("y")
                axis = random.randint(0, batch_size - 1)

                lines, labels = plt.gca().get_legend_handles_labels()
                plt.subplots_adjust(bottom=0.13, left=0.12, right=0.98)
                plt.subplots_adjust(hspace=0.5, wspace=0.15)
                plt.subplots_adjust(top=0.85)
                matplotlib.rcParams['text.usetex'] = True
                plt.suptitle(get_title(test_funcs=test_funcs, is_se=False, axis=axis, task_sampler_list=task_sampler_list), wrap=True, fontsize=20, y=0.985, weight='regular')
                matplotlib.rcParams['text.usetex'] = False

                if not os.path.exists(f"{prefix}/curves"):
                    os.makedirs(f"{prefix}/curves")
                plt.savefig(
                    f"{prefix}/curves/curves_{functions}_trained_on_{train_data}_tested_on_{test_funcs}.png", dpi=300)
                print(f"Saved images `{functions} trained on {train_data}, tested on {test_funcs}`")

def eval_ood(
    round, batch_size, run_dir, functions, 
    ood_x=False, all_ood=False, ood_bias=0, 
    noise=1, n_points=40, train_info="default",
    given_func_pool=None
):
    for train_data in os.listdir(os.path.join(run_dir, functions)):
        if 'baseline' in train_data:
            continue

        pred_list, pred_bl_list = np.zeros((round)).tolist(), np.zeros((round)).tolist()
        xs_list, ys_list = np.zeros((round)).tolist(), np.zeros((round)).tolist()
        gt_x, gt_y = np.zeros((n_points)).tolist(), np.zeros((n_points)).tolist()
        pred_list_pre, pred_bl_list_pre = np.zeros((round)).tolist(), np.zeros((round)).tolist()
        gt_x_pre, gt_y_pre = np.zeros((n_points)).tolist(), np.zeros((n_points)).tolist()
        pred_list_suc, pred_bl_list_suc = np.zeros((round)).tolist(), np.zeros((n_points)).tolist()
        gt_x_suc, gt_y_suc = np.zeros((n_points)).tolist(), np.zeros((n_points)).tolist()

        all_ys_list = np.zeros((batch_size, round)).tolist()
        all_other_ys_pre_list, all_other_ys_suc_list = np.zeros((batch_size, round)).tolist(), np.zeros((batch_size, round)).tolist()
        all_pred_list, all_pred_bl_list = np.zeros((batch_size, round)).tolist(), np.zeros((batch_size, round)).tolist()
        all_pred_list_pre, all_pred_list_suc = np.zeros((batch_size, round)).tolist(), np.zeros((batch_size, round)).tolist()
        all_pred_bl_list_pre, all_pred_bl_list_suc = np.zeros((batch_size, round)).tolist(), np.zeros((batch_size, round)).tolist()

        run_path = os.path.join(run_dir, functions, train_data)
        model, conf = get_model_from_run(run_path, step=-1)
        model_baseline, _ = get_model_from_run(run_path.replace(train_data, 'baseline'), step=-1)
        limit = get_limit(conf.train.limit)
        function_pool = conf.train.train_list
        prefix = f"images/{functions}/{train_info}/{train_data}/state"
        os.makedirs(prefix, exist_ok=True)

        test_func_pool = []
        task_name = train_data.split(sep='+')[0]
        for num1 in range(len(function_pool)):
            for num2 in range(len(function_pool)):
                if num1 == num2 or num1 > num2 and task_name != 'comp':
                    continue
                test_funcs = [function_pool[num1], function_pool[num2]]
                test_func_pool.append(test_funcs)
        for num in range(len(function_pool)):
            test_func_pool.append([function_pool[num]])
        test_func_pool.append(function_pool)

        if given_func_pool:
            test_func_pool = given_func_pool

        iid_counts = [30]
        ood_count = 10
        if all_ood:
            iid_counts = [1]
            ood_count = 39

        for test_funcs in test_func_pool:
            for iid_count in iid_counts:
                if all_ood:
                    cur_iid_count = 40
                else:
                    cur_iid_count = iid_count
                n_dims = conf.model.n_dims
                task_sampler_list = [get_task_sampler(
                    task.split('+')[0], 
                    n_dims,
                    batch_size,
                    k=get_weight(task.split('+')[1]) if '+' in task else 1,
                    **conf.train.task_kwargs)(**conf.train.task_kwargs) 
                for task in test_funcs]

                xs = generator_xs(
                    batch_size=batch_size, 
                    distribution=conf.train.xs_datatype,
                    limit=limit,
                    ood=False
                )
                
                if ood_x:
                    xs_ood_pre = generator_xs(
                        batch_size=batch_size, 
                        distribution=conf.train.xs_datatype,
                        limit=limit,
                        ood=True, order='pre', iid_count=iid_count)
                    
                    xs_ood_suc = generator_xs(
                        batch_size=batch_size, 
                        distribution=conf.train.xs_datatype,
                        limit=limit,
                        ood=True, order='suc', iid_count=iid_count)
                else:
                    xs_ood_pre, xs_ood_suc = xs.clone(), xs.clone()
                    
                # now solve xs
                axis = random.randint(0, batch_size - 1)
                x_generator = get_x_generator(limit, round)()
                cur_rs = get_rs(len(test_funcs))
                ys, _ = get_task(conf.train.task_type, task_sampler_list, test_funcs, xs, rs=cur_rs,
                                    datatype=conf.train.task, bias=conf.train.bias)
                if ood_x:
                    ys_pre, _ = get_task(conf.train.task_type, task_sampler_list, test_funcs, xs_ood_pre, rs=cur_rs,
                                                datatype=conf.train.task, bias=conf.train.bias)
                    ys_suc = ys_pre.clone()
                else:
                    ys_pre = ys.clone()
                    ys_suc = ys.clone()

                if ood_x and ood_bias:  # x ood & add label noise
                    for i in range(ood_count):
                        for j in range(batch_size):
                            ys_pre[j][i] += 10
                            # 0 ~ iid_count - 1; iid_count ~ iid_count + iid_count + ood_count - 1
                            ys_suc[j][i + iid_count] += 10

                if not ood_x:
                    for i in range(ood_count):
                        for j in range(batch_size):
                            val = torch.randn(1).item() * noise
                            ys_pre[j][i] += val
                            # 0~iid_count-1; iid_count~iid_count+iid_count+ood_count-1
                            ys_suc[j][i + iid_count] += val

                for j in range(cur_iid_count):
                    gt_x[j], gt_y[j] = xs[axis][j].item(), ys[axis][j].item()

                for j in range(iid_count + ood_count):
                    gt_x_pre[j], gt_y_pre[j] = xs_ood_pre[axis][j].item(), ys_pre[axis][j].item()
                    gt_x_suc[j], gt_y_suc[j] = xs_ood_suc[axis][j].item(), ys_suc[axis][j].item()

                for T in trange(round):
                    cur_x = next(x_generator)
                    for j in range(batch_size):
                        xs_ood_pre[j][iid_count + ood_count - 1] = cur_x
                        xs_ood_suc[j][iid_count + ood_count - 1] = cur_x
                        xs[j][cur_iid_count - 1] = cur_x
                    ys, _ = get_task(conf.train.task_type, task_sampler_list, test_funcs, xs, rs=cur_rs, datatype=conf.train.task, bias=conf.train.bias)
                    if ood_x:
                        ys_pre, _ = get_task(conf.train.task_type, task_sampler_list, test_funcs, xs_ood_pre, rs=cur_rs, datatype=conf.train.task, bias=conf.train.bias)
                        ys_suc, _ = get_task(conf.train.task_type, task_sampler_list, test_funcs, xs_ood_suc, rs=cur_rs, datatype=conf.train.task, bias=conf.train.bias)
                    else:
                        ys_pre, ys_suc = ys.clone(), ys.clone()

                    if ood_x and ood_bias:  # x ood & add label noise
                        for i in range(ood_count):
                            for j in range(batch_size):
                                ys_pre[j][i] += 10
                                # 0 ~ iid_count-1; iid_count ~ iid_count + iid_count + ood_count - 1
                                ys_suc[j][i + iid_count] += 10

                    if not ood_x: 
                        for i in range(ood_count):
                            for j in range(batch_size):
                                val = torch.randn(1).item() * noise
                                ys_pre[j][i] += val
                                # 0~iid_count-1; iid_count~iid_count+iid_count+ood_count-1
                                ys_suc[j][i + iid_count] += val

                    with torch.no_grad():
                        pred = model(xs, ys)
                        pred_baseline = model_baseline(xs, ys)
                        pred_pre = model(xs_ood_pre, ys_pre)
                        pred_baseline_pre = model_baseline(xs_ood_pre, ys_pre)
                        pred_suc = model(xs_ood_suc, ys_suc)
                        pred_baseline_suc = model_baseline(xs_ood_suc, ys_suc)

                    pred_list_pre[T] = pred_pre[axis][iid_count + ood_count - 1].item()
                    pred_list_suc[T] = pred_suc[axis][iid_count + ood_count - 1].item()
                    pred_bl_list_pre[T] = pred_baseline_pre[axis][iid_count + ood_count - 1].item()
                    pred_bl_list_suc[T] = pred_baseline_suc[axis][iid_count + ood_count - 1].item()

                    min_idx, min_val = 0, 10000000
                    max_idx, max_val = 0, -10000000

                    for j in range(batch_size):
                        all_other_ys_pre_list[j][T] = ys_pre[j][iid_count + ood_count - 1].item()
                        all_other_ys_suc_list[j][T] = ys_suc[j][iid_count + ood_count - 1].item()
                        all_pred_list_pre[j][T] = pred_pre[j][iid_count + ood_count - 1].item()
                        all_pred_list_suc[j][T] = pred_suc[j][iid_count + ood_count - 1].item()
                        all_pred_bl_list_pre[j][T] = pred_baseline_pre[j][iid_count + ood_count - 1].item()
                        all_pred_bl_list_suc[j][T] = pred_baseline_suc[j][iid_count + ood_count - 1].item()

                        all_ys_list[j][T] = ys[j][cur_iid_count - 1].item()
                        all_pred_list[j][T] = pred[j][cur_iid_count - 1].item()
                        all_pred_bl_list[j][T] = pred_baseline[j][cur_iid_count - 1].item()
                        if abs(ys[axis][cur_iid_count - 1].item() - pred[j][cur_iid_count - 1].item()) < min_val:
                            min_idx, min_val = j, abs(ys[axis][cur_iid_count - 1].item() - pred[j][cur_iid_count - 1].item())
                        if abs(ys[axis][cur_iid_count - 1].item() - pred_baseline[j][cur_iid_count - 1].item()) > max_val:
                            max_idx, max_val = j, abs(ys[axis][cur_iid_count - 1].item() - pred_baseline[j][cur_iid_count - 1].item())

                    pred_list[T] = pred[min_idx][cur_iid_count - 1].item()
                    pred_bl_list[T] = pred_baseline[max_idx][cur_iid_count - 1].item()
                    ys_list[T] = ys[axis][cur_iid_count - 1].item()
                    xs_list[T] = xs[axis][cur_iid_count - 1].item()

                # draw loss at specific points
                if not os.path.exists(f"{prefix}/points"):
                    os.makedirs(f"{prefix}/points")

                def save_csv(loss_CC, loss_baseline, loss_list_ood, loss_bl_list_ood, info='default'):
                    df = pd.DataFrame({'CC': loss_CC, 'baseline': loss_baseline, f'CC({info})': loss_list_ood, f'baseline({info})': loss_bl_list_ood})
                    df.to_csv(f'{prefix}/points/csv_noise{noise}_all_{all_ood}_{functions}_ood_{ood_bias}_trained_on_{train_data}_tested_on_{test_funcs}_at_point_{iid_count+ood_count-1}_{info}.csv', index=True)

                def savefig_point(all_ys_list, all_other_ys_list, all_pred_list, all_pred_bl_list, all_pred_list_ood, all_pred_bl_list_ood, info='default'):
                    loss_list = batchMSE(torch.tensor(all_pred_list), torch.tensor(all_ys_list), batch_size)
                    loss_bl_list = batchMSE(torch.tensor(all_pred_bl_list), torch.tensor(all_ys_list), batch_size)
                    loss_list_ood = batchMSE(torch.tensor(all_pred_list_ood), torch.tensor(all_ys_list), batch_size)
                    loss_bl_list_ood = batchMSE(torch.tensor(all_pred_bl_list_ood), torch.tensor(all_ys_list), batch_size)
                    save_csv(loss_list, loss_bl_list, loss_list_ood, loss_bl_list_ood, f'{info}')

                    # log-scale
                    def get_rank(val):
                        if 0 <= val <= 1:
                            return val
                        elif 1 <= val <= 5:
                            return 1.0 * (val - 1) / 4 + 1
                        elif 5 <= val <= 25:
                            return 1.0 * (val - 5) / 20 + 2
                        else:
                            return 3

                    plt.clf()
                    plt.plot([get_rank(i) for i in loss_list], lw=2, color='brown', label="ComFuncLearner")
                    plt.plot([get_rank(i) for i in loss_bl_list], lw=2, color='blue', label="Baseline")
                    plt.plot([get_rank(i) for i in loss_list_ood], lw=2, color='green', label=f"ComFuncLearner(bias)")
                    plt.plot([get_rank(i) for i in loss_bl_list_ood], lw=2, color='red', label=f"Baseline(bias)")
                    xticks = []
                    for tcs in range(len(loss_list)):
                        xticks.append(tcs )
                    plt.xticks(np.arange(0, len(xticks), 4), xticks[::4])
                    if ood_x:
                        yticks = [0, 1, 5, 25]
                        plt.yticks([0, 1, 2, 3], yticks)
                    plt.xlabel("# Rounds")
                    plt.ylabel("Squared Error")

                    lines, labels = plt.gca().get_legend_handles_labels()
                    plt.gca().legend(lines, labels, loc='lower center', ncol=4)
                    plt.subplots_adjust(top=0.80, bottom=0.12, left=0.10, right=0.92)
                    plt.suptitle(get_title(test_funcs=test_funcs, is_se=True, points=iid_count), wrap=True, fontsize=20)
                    plt.savefig(f"{prefix}/points/points_noise{noise}_all_{all_ood}_{functions}_ood_{ood_bias}_trained_on_{train_data}_tested_on_{test_funcs}_at_point_{iid_count+ood_count-1}_{info}", dpi=300)

                savefig_point(all_ys_list, all_other_ys_pre_list, all_pred_list, all_pred_bl_list, all_pred_list_pre, all_pred_bl_list_pre, 'pre')
                savefig_point(all_ys_list, all_other_ys_suc_list, all_pred_list, all_pred_bl_list, all_pred_list_suc, all_pred_bl_list_suc, 'suc')
                axis = random.randint(0, batch_size - 1)

                # draw curve at specific points
                if not os.path.exists(f"{prefix}/sp_curves"):
                    os.makedirs(f"{prefix}/sp_curves")

                def savefig_sp_curve(ys_list, pred_list, pred_bl_list, pred_list_ood, pred_bl_list_ood, gt_x_ood, gt_y_ood, info='default'):
                    plt.clf()
                    if ood_x:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                        ax1.scatter(gt_x_ood[0], gt_y_ood[0], s=70, marker='*', label="ICE", color='yellow')
                        for j in range(iid_count + ood_count):
                            if j == 0:
                                continue
                            ax1.scatter(gt_x_ood[j], gt_y_ood[j], s=70+8*j, marker='*', color='yellow')
                        ax1.scatter(np.linspace(-limit, limit, round), ys_list, s=70, marker='o', facecolors='none', color='black', label='GT')
                        ax1.scatter(np.linspace(-limit, limit, round), pred_list, s=20, marker='s', color='brown', label='CFL')
                        ax1.scatter(np.linspace(-limit, limit, round), pred_bl_list, s=20, marker='^', color='blue', label='Baseline')
                        ax1.scatter(np.linspace(-limit, limit, round), pred_bl_list_ood, s=20, marker='p', color='forestgreen', label=f'CFL(bias)')
                        ax1.scatter(np.linspace(-limit, limit, round), pred_list_ood, s=20, marker='*', color='red', label=f'CFL(bias)')

                        ax2.scatter(gt_x[0], gt_y[0], s=70, marker='*', color='yellow')
                        for j in range(iid_count + ood_count):
                            if j == 0:
                                continue
                            ax2.scatter(gt_x_ood[j], gt_y_ood[j], s=70+8*j, marker='*', color='yellow')
                        ax2.scatter(np.linspace(-limit, limit, round), ys_list, s=70, marker='o', facecolors='none', color='black')
                        ax2.scatter(np.linspace(-limit, limit, round), pred_list, s=20, marker='s', color='brown')
                        ax2.scatter(np.linspace(-limit, limit, round), pred_bl_list, s=20, marker='^', color='blue')
                        ax2.scatter(np.linspace(-limit, limit, round), pred_list_ood, s=20, marker='*', color='red')
                        ax2.scatter(np.linspace(-limit, limit, round), pred_bl_list_ood, s=20, marker='p', color='forestgreen')
                        # lines, labels = ax1.get_legend_handles_labels()
                        # fig.legend(lines, labels, loc='lower center', ncol=6)

                    else:
                        plt.scatter(gt_x_ood[0], gt_y_ood[0], s=70, marker='*', label="ICE", color='yellow')
                        for j in range(iid_count + ood_count):
                            if j == 0:
                                continue
                            plt.scatter(gt_x_ood[j], gt_y_ood[j], s=70+8*j, marker='*', color='yellow')
                        plt.scatter(np.linspace(-limit, limit, round), ys_list, s=70, marker='o', facecolors='none', color='black', label='GT')
                        plt.scatter(np.linspace(-limit, limit, round), pred_list, s=20, marker='s', color='brown', label='CFL')
                        plt.scatter(np.linspace(-limit, limit, round), pred_bl_list, s=20, marker='^', color='blue', label='Baseline')
                        plt.scatter(np.linspace(-limit, limit, round), pred_list_ood, s=20, marker='p', color='forestgreen', label=f'CFL(bias)')
                        plt.scatter(np.linspace(-limit, limit, round), pred_bl_list_ood, s=20, marker='*', color='red', label=f'Baseline(bias)')

                    plt.subplots_adjust(bottom=0.10, left=0.10, right=0.98)
                    plt.subplots_adjust(hspace=0.5, wspace=0.2)
                    plt.subplots_adjust(top=0.85)

                    lines, labels = plt.gca().get_legend_handles_labels()
                    plt.gca().legend(lines, labels, loc='center right', ncol=1, bbox_to_anchor=(2, 0.5))
                    plt.subplots_adjust(bottom=0.10, left=0.12, right=0.50)
                    plt.subplots_adjust(hspace=0.5, wspace=0.15)
                    plt.subplots_adjust(top=0.85)
                    plt.suptitle(f'{get_title(test_funcs=test_funcs, is_se=True)}', wrap=True, fontsize=20)

                    plt.suptitle(get_title(test_funcs=test_funcs, is_se=False, axis=axis, points=iid_count, task_sampler_list=task_sampler_list), wrap=True, fontsize=20)
                    plt.savefig(f"{prefix}/sp_curves/sp_curves_noise{noise}_all_{all_ood}_{functions}_ood_{ood_bias}_trained_on_{train_data}_tested_on_{test_funcs}_at_point_{iid_count+ood_count-1}_{info}", dpi=300)
                    plt.savefig(f"{prefix}/sp_curves/sp_curves_noise{noise}_all_{all_ood}_{functions}_ood_{ood_bias}_trained_on_{train_data}_tested_on_{test_funcs}_at_point_{iid_count + ood_count - 1}_{info}.pdf")

                savefig_sp_curve(ys_list, pred_list, pred_bl_list, pred_list_pre, pred_bl_list_pre, gt_x_pre, gt_y_pre, info='pre')
                savefig_sp_curve(ys_list, pred_list, pred_bl_list, pred_list_suc, pred_bl_list_suc, gt_x_suc, gt_y_suc, info='suc')

                print(f"saved image at {prefix}/sp_curves/curves_{functions}_trained_on_{train_data}_tested_on_{test_funcs}")

# sort_type: 
# 0 -> random
# 1 -> sort by value
# 2 -> sort by abs
def eval_with_repeat(
    round, batch_size, run_dir, functions, 
    n_points=40, train_info="default", sort_type=0, given_func_pool=None
):
    points = [19, 39]
    pred_list = np.zeros((len(points), round)).tolist()
    pred_bl_list = np.zeros((len(points), round)).tolist()
    ys_list = np.zeros((len(points), round)).tolist()
    xs_list = np.zeros((len(points), round)).tolist()
    gt_x = np.zeros((len(points), n_points)).tolist()
    gt_y = np.zeros((len(points), n_points)).tolist()

    all_ys = np.zeros((len(points), batch_size, round)).tolist()
    all_pred = np.zeros((len(points), batch_size, round)).tolist()
    all_pred_bl = np.zeros((len(points), batch_size, round)).tolist()

    for train_data in os.listdir(os.path.join(run_dir, functions)):
        if 'baseline' in train_data:
            continue

        run_path = os.path.join(run_dir, functions, train_data)
        model, conf = get_model_from_run(run_path, step=-1)
        model_baseline, _ = get_model_from_run(run_path.replace(train_data, 'baseline'), step=-1)
        limit = get_limit(conf.train.limit)
        function_pool = conf.train.train_list
        prefix = f"images/{functions}/{train_info}/{train_data}/state"
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        test_func_pool = []
        task_name = train_data.split(sep='+')[0]
        for num1 in range(len(function_pool)):
            for num2 in range(len(function_pool)):
                if num1 == num2:
                    continue
                if num1 > num2 and task_name != 'Com':
                    continue
                test_funcs = [function_pool[num1], function_pool[num2]]
                test_func_pool.append(test_funcs)
        for num in range(len(function_pool)):
            test_func_pool.append([function_pool[num]])
        test_func_pool.append(function_pool)

        if given_func_pool:
            test_func_pool = given_func_pool

        for test_funcs in test_func_pool:
            model, conf = get_model_from_run(run_path)
            n_dims = conf.model.n_dims
            n_points = conf.train.curriculum.points.end
            task_sampler_list = [get_task_sampler(
                task.split('+')[0], 
                n_dims,
                batch_size,
                k=get_weight(task.split('+')[1]) if '+' in task else 1,
                **conf.train.task_kwargs)(**conf.train.task_kwargs) 
            for task in test_funcs]
            x = generator_xs(
                batch_size=batch_size, 
                distribution=conf.train.xs_datatype,
                limit=limit
            )
            if sort_type == 1: # sort by value
                x, _ = x.sort(dim=1)
            elif sort_type == 2: # sort by abs
                x_abs = torch.abs(x)
                for i in range(batch_size):
                    tmp = x_abs[i]
                    _, indices = tmp.sort(dim=0)
                    tmp = x[i].clone()
                    for j in range(n_points):
                        x[i][j] = tmp[indices[j]]
            xs = torch.zeros(torch.Size([len(points), batch_size, n_points, n_dims]))

            for i in range(len(points)):
                xs[i] = torch.clone(x)

            axis = random.randint(0, batch_size - 1)
            x_generator = get_x_generator(limit, round)()
            cur_rs = get_rs(len(test_funcs))
            ys, rs = get_task(conf.train.task_type, task_sampler_list, test_funcs, x, rs=cur_rs,
                             datatype=conf.train.task, bias=conf.train.bias)

            for i in range(len(points)):
                for j in range(points[i]):
                    gt_x[i][j] = xs[i][axis][j].item()
                    gt_y[i][j] = ys[axis][j].item()
            for T in trange(round):
                cur_x = next(x_generator)
                try:
                    for i in range(len(points)):
                        for j in range(batch_size):
                            xs[i][j][points[i]] = cur_x
                        ys, rs = get_task(conf.train.task_type, task_sampler_list, test_funcs, xs[i], rs=cur_rs,
                                            datatype=conf.train.task, bias=conf.train.bias)
                        with torch.no_grad():
                            pred = model(xs[i], ys)
                        with torch.no_grad():
                            pred_baseline = model_baseline(xs[i], ys)

                        pred_list[i][T] = pred[axis][points[i]].item()
                        pred_bl_list[i][T] = pred_baseline[axis][points[i]].item()
                        ys_list[i][T] = ys[axis][points[i]].item()
                        xs_list[i][T] = xs[i][axis][points[i]].item()

                        for j in range(batch_size):
                            all_ys[i][j][T] = ys[j][points[i]].item()
                            all_pred[i][j][T] = pred[j][points[i]].item()
                            all_pred_bl[i][j][T] = pred_baseline[j][points[i]].item()

                except Exception as e:
                    print(str(e), "error occurs")
                    return

            # draw loss at specific points
            if not os.path.exists(f"{prefix}/points"):
                os.makedirs(f"{prefix}/points")

            for i in range(len(points)):
                loss = batchMSE(torch.tensor(all_pred[i]), torch.tensor(all_ys[i]), batch_size)
                loss_baseline = batchMSE(torch.tensor(all_pred_bl[i]), torch.tensor(all_ys[i]), batch_size)
                std = batchSTD(torch.tensor(all_pred[i]), torch.tensor(all_ys[i]), batch_size)
                std_baseline = batchMSE(torch.tensor(all_pred_bl[i]), torch.tensor(all_ys[i]), batch_size)

                def save_csv(loss_CC, loss_baseline, std_CC, std_baseline):
                    df = pd.DataFrame({'CC': loss_CC, 'baseline': loss_baseline, 'std_CC': std_CC, 'std_bl': std_baseline})
                    df.to_csv(f'{prefix}/points/csv_{functions}_trained_on_{train_data}_tested_on_{test_funcs}_at_points_{points[i]}.csv', index=True)

                save_csv(loss, loss_baseline, std, std_baseline)

                plt.clf()
                plt.plot(np.array(loss), lw=2, color='brown', label="CFL")
                plt.plot(np.array(loss_baseline), lw=2, color='blue', label="Baseline")
                xticks = []
                for tcs in range(len(loss)):
                    xticks.append(tcs)
                plt.xticks(np.arange(0, len(xticks), 4), xticks[::4])
                plt.ylim(0, np.array(loss_baseline[4:-4]).mean() * 2)
                plt.xlabel("# Rounds")
                plt.ylabel("Squared Error")
                # lines, labels = plt.gca().get_legend_handles_labels()
                # plt.gca().legend(lines, labels, loc='lower center', ncol=2)
                plt.subplots_adjust(bottom=0.10, left=0.08, right=0.92)
                plt.subplots_adjust(hspace=0.5, wspace=0.15)
                plt.subplots_adjust(top=0.85)

                x_range = np.arange(0, round).ravel()
                x_smooth = np.linspace(0, round - 1, 500)

                plt.fill_between(x_smooth, interp1d(x_range, loss - std, kind='cubic')(x_smooth),
                                 interp1d(x_range, loss + std, kind='cubic')(x_smooth), color='brown', alpha=0.1)
                plt.plot(loss_baseline, lw=2, color='blue', label="Baseline")
                plt.fill_between(x_smooth, interp1d(x_range, loss_baseline - std_baseline, kind='cubic')(x_smooth),
                                 interp1d(x_range, loss_baseline + std_baseline, kind='cubic')(x_smooth),
                                 color='blue', alpha=0.1)

                plt.suptitle(get_title(test_funcs=test_funcs, points=points[i], is_se=True), wrap=True, fontsize=20)

                plt.savefig(f"{prefix}/points/points_{functions}_trained_on_{train_data}_tested_on_{test_funcs}_at_point_{points[i]}", dpi=300)

            # draw curve at specific points
            if not os.path.exists(f"{prefix}/sp_curves"):
                os.makedirs(f"{prefix}/sp_curves")

            for i in range(len(points)):
                plt.clf()
                cur_xs = gt_x[i][0:points[i]]
                plt.scatter(gt_x[i][0], gt_y[i][0], s=70, marker='*', label="ICE", color='yellow')
                for j in range(len(cur_xs)):
                    if j == 0:
                        continue
                    plt.scatter(cur_xs[j], gt_y[i][j], s=70+8*j, marker='*', color='yellow')
                plt.scatter(np.linspace(-limit, limit, round), ys_list[i], s=50, marker='o',
                            facecolors='none', color='green', label='GT')
                plt.scatter(np.linspace(-limit, limit, round), pred_list[i], s=20, marker='s',
                            color='red', label='CFL')
                plt.scatter(np.linspace(-limit, limit, round), pred_bl_list[i], s=20, marker='^',
                            color='blue', label='Baseline')
                plt.xlabel("x")
                plt.ylabel("y")
                axis = random.randint(0, batch_size - 1)
                # lines, labels = plt.gca().get_legend_handles_labels()
                # plt.gca().legend(lines, labels, loc='lower center', ncol=3)
                plt.subplots_adjust(bottom=0.10, left=0.08, right=0.92)
                plt.subplots_adjust(hspace=0.5, wspace=0.15)
                plt.subplots_adjust(top=0.85)
                plt.suptitle(get_title(test_funcs=test_funcs, is_se=False, axis=axis, points=points[i], task_sampler_list=task_sampler_list), wrap=True, fontsize=20)

                plt.savefig(
                    f"{prefix}/sp_curves/points_{functions}_trained_on_{train_data}_tested_on_{test_funcs}_at_point_{points[i]}", dpi=300)
            print(f"saved image at {prefix}/sp_curves/curves_{functions}_trained_on_{train_data}_tested_on_{test_funcs}")

def eval_multidim(run_dir, functions, batch_size, only_state=True, train_info='default'):
    ckpt = "multidim"
    for train_data in os.listdir(run_dir + "/" + functions):
        if 'baseline' in train_data:
            continue
        run_path = os.path.join(run_dir, functions, train_data)
        print("LOADING:", run_path, ckpt)
        model, conf = get_model_from_run(run_path, step=ckpt)
        limit = get_limit(conf.train.limit)
        test_func_pool = []
        if conf.train.task == "sinusoidal":
            function_pool = ["sin+1", "cos+1", "sin+2", "cos+2"]
        elif conf.train.task == "legendre":
            function_pool = ["linear", "cube", "square", "Fourth"]
        elif conf.train.task == "basic":
            function_pool = ["linear", "square", "inverse", "arccos"]
        else:
            raise NotImplementedError

        if "constant" in conf.train.train_list:
            function_pool.append("constant")

        for num1 in range(len(function_pool)):
            for num2 in range(len(function_pool)):
                if num1 == num2:
                    continue
                if num1 > num2:
                    continue
                test_funcs = [function_pool[num1], function_pool[num2]]
                test_func_pool.append(test_funcs)
        for num in range(len(function_pool)):
            test_func_pool.append([function_pool[num]])
        test_func_pool.append(function_pool)

        states = []
        for name in os.listdir(run_path):
            if '.pt' in name:
                states.append(name)

        def replace_last_occurrence(string, old, new):
            parts = string.rsplit(old, 1)
            return new.join(parts)

        for state in states:
            if only_state and 'state' not in state:
                continue
            for test_funcs in test_func_pool:
                prefix = f"../images/{functions}/{train_info}/{train_data}/{state[:-3]}"
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                model, conf = get_model_from_run(run_path, step=ckpt, model_name=state)
                model_baseline, conf_baseline = get_model_from_run(replace_last_occurrence(run_path, train_data, 'baseline'), step=ckpt,
                                                                   model_name=state)
                n_dims = conf.model.n_dims
                task_sampler_list = [get_task_sampler(
                    task.split('+')[0], 
                    n_dims,
                    batch_size,
                    k=get_weight(task.split('+')[1]) if '+' in task else 1,
                    **conf.train.task_kwargs)(**conf.train.task_kwargs) 
                for task in test_funcs]
                
                xs = generator_xs(
                    batch_size=batch_size, 
                    n_dims=n_dims, 
                    distribution=conf.train.xs_datatype,
                    limit=limit
                )
                if n_dims == 2:
                    for batch in range(batch_size):
                        for n_points in range(40):
                            xs[batch][n_points][1] = xs[batch][n_points][0]

                cur_rs = get_rs(len(test_funcs))
                ys, rs = get_task(conf.train.task_type, task_sampler_list, test_funcs, xs, rs=cur_rs,
                                 datatype=conf.train.task, bias=conf.train.bias)
                with torch.no_grad():
                    pred = model(xs, ys)
                with torch.no_grad():
                    pred_baseline = model_baseline(xs, ys)
                loss = batchMSE(pred, ys, batch_size)
                loss_baseline = batchMSE(pred_baseline, ys, batch_size)
                std = batchSTD(pred, ys, batch_size)
                std_baseline = batchSTD(pred_baseline, ys, batch_size)

                x_range = np.arange(0, 40).ravel()
                x_smooth = np.linspace(0, 39, 500)

                # draw Squared Error of 40 points
                plt.clf()
                plt.plot(loss, lw=2, color='red', label="CFL")
                plt.fill_between(x_smooth, interp1d(x_range, loss - std, kind='cubic')(x_smooth),
                                 interp1d(x_range, loss + std, kind='cubic')(x_smooth), color='red', alpha=0.1)
                plt.plot(loss_baseline, lw=2, color='blue', label="Baseline")
                plt.fill_between(x_smooth, interp1d(x_range, loss_baseline - std_baseline, kind='cubic')(x_smooth),
                                 interp1d(x_range, loss_baseline + std_baseline, kind='cubic')(x_smooth),
                                 color='blue', alpha=0.1)
                xticks = []
                for tcs in range(len(loss)):
                    xticks.append(tcs)
                plt.xticks(np.arange(0, len(xticks), 4), xticks[::4])
                plt.ylim(0, 2 * loss_baseline[5:].mean())
                plt.xlabel("# In-Context Examples")
                plt.ylabel("Squared Error")

                # lines, labels = plt.gca().get_legend_handles_labels()
                # plt.gca().legend(lines, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3))
                axis = random.randint(0, batch_size - 1)
                plt.subplots_adjust(bottom=0.10, left=0.12, right=0.92)
                plt.subplots_adjust(hspace=0.5, wspace=0.15)
                plt.subplots_adjust(top=0.85)
                plt.suptitle(get_title(test_funcs=test_funcs, is_se=False, axis=axis, task_sampler_list=task_sampler_list, n_dims=n_dims), wrap=True, fontsize=20)

                if not os.path.exists(f"{prefix}/se"):
                    os.makedirs(f"{prefix}/se")

                def save_csv(loss_CC, loss_baseline, std_CC, std_baseline):
                    df = pd.DataFrame({'CC': loss_CC, 'baseline': loss_baseline, 'std_CC': std_CC, 'std_bl': std_baseline})
                    df.to_csv(f'{prefix}/se/csv_{functions}_trained_on_{train_data}_tested_on_{test_funcs}.csv', index=True)

                save_csv(loss, loss_baseline, std, std_baseline)

                plt.savefig(
                    f"{prefix}/se/se_{functions}_trained_on_{train_data}_tested_on_{test_funcs}", 
                    dpi=300
                )

                if n_dims == 2:
                    # draw 3D curve for 2D
                    plt.clf()
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    xs_0, xs_1 = xs.split(1, dim=2)
                    axis = random.randint(0, batch_size - 1)
                    xs_0, xs_1 = xs_0.numpy().squeeze(-1)[axis], xs_1.numpy().squeeze(-1)[axis]
                    ys, pred, pred_baseline = ys.numpy()[axis], pred.numpy()[axis], pred_baseline.numpy()[axis]
                    ax.scatter(xs_0, xs_1, ys, s=50, marker='o', facecolors='none', edgecolors='green', label='GT')
                    ax.scatter(xs_0, xs_1, pred, s=20, marker='s', color='blue', label='CFL')
                    ax.scatter(xs_0, xs_1, pred_baseline, s=20, marker='^', color='red', label='Baseline')
                    
                    ax.set_xlabel('x1')
                    ax.set_ylabel('x2')
                    ax.set_zlabel('y')
                    # plt.gca().legend(lines, labels, loc='lower center', ncol=3)
                    plt.subplots_adjust(bottom=0.10, left=0.12, right=0.92)
                    plt.subplots_adjust(hspace=0.5, wspace=0.15)
                    plt.subplots_adjust(top=0.85)
                    plt.suptitle(get_title(test_funcs=test_funcs, is_se=False, axis=axis, task_sampler_list=task_sampler_list, n_dims=n_dims), wrap=True, fontsize=20)

                    if not os.path.exists(f"{prefix}/curves"):
                        os.makedirs(f"{prefix}/curves")
                    plt.savefig(
                        f"{prefix}/curves/curves_{functions}_trained_on_{train_data}_tested_on_{test_funcs}", dpi=300)
                print(f"Saved images `{functions} trained on {train_data}, tested on {test_funcs}`")


if __name__ == "__main__":
    run_dir = '/path/to/your/runs'
    sinusoidal = [
        ['sin+1'], ['cos+1'], ['sin+2'], ['cos+2'],
        ['sin+1', 'cos+1'], ['sin+1', 'sin+2'], ['sin+1', 'cos+2'], ['cos+1', 'sin+2'], ['cos+1', 'cos+2'], ['sin+2', 'cos+2'],
        ['sin+1', 'cos+1', 'sin+2', 'cos+2']
    ]

    polynomial = [
        ['linear'], ['square'], ['cube'], ['biquadrate'],
        ['linear', 'square'], ['linear', 'cube'], ['linear', 'biquadrate'], ['square', 'cube'], ['square', 'biquadrate'], ['cube', 'biquadrate'],
        ['linear', 'square', 'cube', 'biquadrate']
    ]

    functions = 'kan'
    eval_without_repeat(run_dir, functions, 128, train_info='default', given_func_pool=sinusoidal)
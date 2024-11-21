import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import random
import os
import re
import json
import copy
import argparse
from preprocess import generate_sft_data, create_instruction
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sns.set_theme(style='darkgrid')
palette = sns.color_palette('colorblind')
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'Times New Roman'

# use scripts/server-starter.sh to deploy your models locally
bl_ports = [18008, 18009, 18010, 18011]
bl_clients = [OpenAI(api_key="got", base_url=f"http://localhost:{port}/v1") for port in bl_ports]
cc_ports = [18012, 18013, 18014, 18015]
cc_clients = [OpenAI(api_key="got", base_url=f"http://localhost:{port}/v1") for port in cc_ports]

FUNC_DICT = {
    'linear': '$x$',
    'square': '$x^2$',
    'cube': '$x^3$',
    'biquadrate': '$x^4$',
    'arccos': '$\\arccos(x)$',
    'inverse': '$\\frac{1}{x}$',
    'abs': '$|x|$',
}

def extract_answer(output):
    answer_pattern = re.compile(r"<Answer>(.*?)<\/Answer>")
    prediction_match = answer_pattern.search(output)
    if prediction_match:
        return int(prediction_match.group(1))
    print("ERROR", output)
    return None

def call(prompt, model, type='bl'):
    if type == 'bl':
        client = random.sample(bl_clients, 1)[0]
    elif type =='cc':
        client = random.sample(cc_clients, 1)[0]
    response = client.chat.completions.create(
        model=model,
        temperature=1e-6,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

def process_single_x(x, points, limit, bl_model, cc_model, answer_pattern):
    local_points = copy.deepcopy(points)
    
    for i in range(len(local_points)):
        if x == local_points[i][0]:
            local_points[i], local_points[2 * limit - 1] = local_points[2 * limit - 1], local_points[i]
            break
    instruction = create_instruction(local_points[:2 * limit], 2 * limit)
    bl_output = call(instruction, bl_model, type='bl')
    cc_output = call(instruction, cc_model, type='cc')
    bl_predicted_y = answer_pattern.search(bl_output)
    cc_predicted_y = answer_pattern.search(cc_output)
    if bl_predicted_y != None and cc_predicted_y != None:
        bl_predicted_y = int(bl_predicted_y.group(1))
        cc_predicted_y = int(cc_predicted_y.group(1))
        return x, local_points[2 * limit - 1][1], bl_predicted_y, cc_predicted_y
    else:
        print(f"Failed to extract answer for x={x}")
        return None

def process_entry(limit, entry, bl_model, cc_model, answer_pattern, bsize, max_inner_workers=8):
    results = []
    
    for round in range(bsize):
        print(f'Processing entry {entry["function"]} round {round+1}/{bsize}')
        xs, ref_ys, bl_predicted_ys, cc_predicted_ys = [], [], [], []
        points = entry['points']

        with ThreadPoolExecutor(max_workers=max_inner_workers) as executor:
            futures = {
                executor.submit(process_single_x, x, points, limit, bl_model, cc_model, answer_pattern): x
                for x in range(-limit, limit + 1, 1)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing xs (bsize run)", leave=False):
                result = future.result()
                if result:
                    x, ref_y, bl_predicted_y, cc_predicted_y = result
                    xs.append(x)
                    ref_ys.append(ref_y)
                    bl_predicted_ys.append(bl_predicted_y)
                    cc_predicted_ys.append(cc_predicted_y)

        if xs:
            results.append({
                'function': entry['function'],
                'xs': xs,
                'ref_ys': ref_ys,
                'bl_predicted_ys': bl_predicted_ys,
                'cc_predicted_ys': cc_predicted_ys,
            })

    return results

def run_infer(limit, sft_data, bl_model, cc_model, bsize, max_workers=32):
    answer_pattern = re.compile(r"<Answer>(.*?)<\/Answer>")
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(process_entry, limit, entry, bl_model, cc_model, answer_pattern, bsize): entry
            for entry in sft_data
        }
        
        for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="Processing entries"):
            entry = future_to_entry[future]
            try:
                results = future.result()
                if results:
                    entry['results'] = results
                    all_results.append(entry)
            except Exception as exc:
                print(f"Entry generated an exception: {exc}")

    return all_results

def save_extracted_data(extracted_data, json_output):
    with open(json_output, 'w') as f:
        json.dump(extracted_data, f, indent=4)
    print(f"Extracted data saved to {json_output}")

def get_title(function):
    ret = 'Test: Com of '
    if len(function) == 4:
        ret += f'all base functions'
    else:
        for (i, (weight, func_name)) in enumerate(function):
            if func_name in FUNC_DICT:
                func_name = FUNC_DICT[func_name]
            if i == 0:
                ret += func_name
            else:
                ret += f', {func_name}'

    # draw params
    ret += f'\nWeights: '
    for (i, (weight, func_name)) in enumerate(function):
        if i == 0:
            ret += f'{weight}'
        else:
            ret += f', {weight}'
    return ret

def show_result(json_output, output_folder):
    with open(json_output, 'r') as f:
        results = json.load(f)
    print(len(results))
    for entry in results:
        print(f"Function: {entry['function']}")
        print(f"Points: {entry['points']}")
        for i, result in enumerate(entry['results']):
            print(f"Run {i+1}:")
            print(f"  Xs: {result['xs']}")
            print(f"  Reference Ys: {result['ref_ys']}")
            print(f"  Baseline Predicted Ys: {result['bl_predicted_ys']}")
            print(f"  ComFuncLearner Predicted Ys: {result['cc_predicted_ys']}")
        print()

    results = [d['results'] for d in results]
    for result in results:
        xs, ys, bl_pred, cc_pred = [], [], [], []
        axis = random.randint(0, len(result) - 1)
        for run in result:
            xs.append(run['xs'])
            ys.append(run['ref_ys'])
            bl_pred.append(run['bl_predicted_ys'])
            cc_pred.append(run['cc_predicted_ys'])
        function = result[0]['function']    
        xs = np.array(xs[axis])
        ys = np.array(ys[axis])
        bl_pred = np.array(bl_pred[axis])
        cc_pred = np.array(cc_pred[axis])
        
        prefix = f'{output_folder}/{"_".join([func_name for (_, func_name) in function])}'
        plt.clf()
        plt.scatter(xs, ys, s=50, marker='o',
                    facecolors='none', color='black', label='GT')
        plt.scatter(xs, cc_pred, s=20, marker='s',
                    color='blue', label='ComFuncLearner')
        plt.scatter(xs, bl_pred, s=20, marker='^',
                    color='red', label='Baseline')
        plt.xlabel("x")
        plt.ylabel("y")

        ax = plt.gca()
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        
        lines, labels = plt.gca().get_legend_handles_labels()
        # plt.gca().legend(lines, labels, loc='right', ncol=1)
        plt.subplots_adjust(bottom=0.13, left=0.12, right=0.98)
        plt.subplots_adjust(hspace=0.5, wspace=0.15)
        plt.subplots_adjust(top=0.85)
        matplotlib.rcParams['text.usetex'] = True
        plt.suptitle(get_title(function), wrap=True, fontsize=20)
        matplotlib.rcParams['text.usetex'] = False

        if not os.path.exists(f"{prefix}/curves"):
            os.makedirs(f"{prefix}/curves")
        plt.savefig(
            f"{prefix}/curves/curves_{function}.png", dpi=300)
        print(f"Saved images `{function}`")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT data and simulate OpenAI API calls.")
    parser.add_argument('--json_output', type=str, required=True, help="The output path for the extracted data.")
    parser.add_argument('--n_functions', type=int, default=1, help='The number of functions to generate (default: 1).')
    parser.add_argument('--n_points', type=int, default=20, help="The number of points to use for each function (default: 19).")
    parser.add_argument('--limit', type=int, default=20, help="The limit for x values (default: 10).")
    parser.add_argument('--coef_ranges', type=str, help="The coefficient ranges for each function, e.g., 'linear:-10,10 square:1,5'.")
    parser.add_argument('--num_examples', type=int, default=10, help="Number of examples to generate (default: 10).")
    parser.add_argument('--bsize', type=int, default=5, help="Number of times to evaluate each function (default: 5).")
    parser.add_argument('--bl_model', type=str, help="The baseline model to use.")
    parser.add_argument('--cc_model', type=str, help="The combination model to use.")
    parser.add_argument('--max_workers', type=int, default=32, help="The maximum number of worker threads for concurrent execution (default: 32).")
    parser.add_argument('--show_result', action='store_true', help="Show the extracted data.")
    parser.add_argument('--comp', action='store_true', help="Whether to generate composition of two functions.")
    parser.add_argument('--output_folder', type=str, default='output', help="The output folder for the images.")
    args = parser.parse_args()

    if args.show_result:
        show_result(args.json_output, args.output_folder)
        exit(0)

    coef_ranges = [
        {
            func.split(':')[0]: tuple(map(int, func.split(':')[1].split(','))) 
            for func in coef_range.split()
        } for coef_range in args.coef_ranges.split(';')
    ]
    
    for i, ranges in enumerate(coef_ranges):
        generate_sft_data(f"temp_sft_data_{i}.json", ranges, args.comp, args.n_functions, args.n_points, args.limit, args.num_examples)
    
    sft_data = []
    for i in range(len(coef_ranges)):
        with open(f"temp_sft_data_{i}.json", 'r') as f:
            sft_data += json.load(f)
        os.remove(f"temp_sft_data_{i}.json")
    
    for i in range(len(sft_data)):
        del sft_data[i]["input"]
        del sft_data[i]["output"]

    results = run_infer(args.limit, sft_data, args.bl_model, args.cc_model, args.bsize, max_workers=args.max_workers)
    save_extracted_data(results, args.json_output)
    
# python evaluation.py \
#     --json_output abs.json \
#     --n_functions 1 \
#     --n_points 20 \
#     --limit 40 \
#     --coef_ranges "abs:-256,-128 square:1,16;; square:1,16 abs:-256,-128; abs:-256,-128; square:1,16" \
#     --num_examples 1 \
#     --bl_model /workspace/saves/llama2-10k-bl/ \
#     --cc_model /workspace/LLaMA-Factory/saves/llama-comp-cc/ \
#     --max_workers 64 \
#     --bsize 1

# python evaluation.py \
#     --json_output abs.json \
#     --output_folder output/abs \
#     --show_result
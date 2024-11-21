import random
import numpy as np
import json
import argparse

def linear(x):
    return x

def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def biquadrate(x):
    return x ** 4

def abs_value(x):
    return abs(x)

FUNCTIONS = {
    'linear': linear,
    'square': square,
    'cube': cube,
    'biquadrate': biquadrate,
    'abs': abs_value
}

class FunctionCombination:
    def __init__(self, terms):
        self.terms = terms

    def evaluate(self, x, comp=False):
        result = 0
        if not comp:
            for coef, func_name in self.terms:
                func = FUNCTIONS[func_name]
                result += coef * func(x)
        else: # composition of two functions
            assert len(self.terms.items()) == 2, "Composition of two functions only."
            coef1, func_name1 = self.terms[0]
            coef2, func_name2 = self.terms[1]
            func1 = FUNCTIONS[func_name1]
            func2 = FUNCTIONS[func_name2]
            result = coef1 * func1(coef2 * func2(x))
        return result

    def __str__(self):
        terms_str = [f"{coef} * {func_name}" for coef, func_name in self.terms]
        return " + ".join(terms_str)

def generate_custom_function(coef_ranges):
    terms = []
    for func_name, coef_range in coef_ranges.items():
        if func_name in FUNCTIONS:
            coef = random.randint(*coef_range)
            terms.append((coef, func_name))
        else:
            raise ValueError(f"Function {func_name} is not defined.")
    return FunctionCombination(terms)

def generate_multiple_functions(n, coef_ranges):
    return [generate_custom_function(coef_ranges) for _ in range(n)]

def generate_function_points(func_comb, x_values):
    return [(int(x), int(func_comb.evaluate(x))) for x in x_values]

def create_instruction(points, n_points):
    example_points = [(-7, -70), (34, -340), (0, 0), (2, -20), (11, -110), 
                      (31, -310), (38, -380), (12, -120), (23, -230), (14, -140), 
                      (30, -300), (-25, -250), (33, -330), (-30, -300), (-24, -240), 
                      (16, -160), (6, -60), (13, -130), (-28, -280), (-31,)]
    example_prediction = -310
    example_str = ', '.join([f"({x}, {y[0]})" if len(point) == 2 else f"({x},)" for point in example_points for x, *y in [point]])
    task_points_str = ', '.join([f"({x}, {y})" for x, y in points[:-1]]) + f", ({points[-1][0]},)"    
    instruction = f"Now you are a proficient function learner, who is a master of learning a math function from (x, y) pairs. Your task is to learn a function from (x, y) pairs from given points and predict a y given x.\n" \
                  f"Specifically, we'll give you {n_points} points (x, y) pairs, and you need to predict the y value of the {n_points}-th point.\n" \
                  f"Please note that you should answer your prediction wrapped in <Answer></Answer> like <Answer>41</Answer>.\n" \
                  f"Here are examples. Points formatted as (x, y):\n" \
                  f"# Example: {example_str}\n" \
                  f"# Prediction: <Answer>{example_prediction}</Answer>\n\n" \
                  f"# Task: {task_points_str}\n" \
                  f"# Prediction\n"
    return instruction

def generate_sft_data(json_output, coef_ranges, n_functions=1, n_points=19, limit=40, num_examples=10000):
    sft_data = []
    for idx in range(num_examples):
        random_funcs = generate_multiple_functions(n_functions, coef_ranges)
        for func_comb in random_funcs:
            x_values = np.linspace(-limit, limit, 2 * limit + 1)
            raw_points = generate_function_points(func_comb, x_values)
            points = random.sample(raw_points, n_points)
            instruction = create_instruction(points[:n_points], n_points)
            sft_data.append({
                "input": instruction,
                "output": f"<Answer>{points[:n_points][-1][1]}</Answer>",
                "points": raw_points,
                "function": func_comb.terms
            })
    with open(json_output, 'w') as f:
        json.dump(sft_data, f, indent=4)
    print(f"SFT data saved to {json_output}")

def parse_coef_ranges(coef_ranges_str):
    coef_ranges = {}
    for pair in coef_ranges_str.split():
        func_name, range_str = pair.split(':')
        coef_min, coef_max = map(int, range_str.split(','))
        coef_ranges[func_name] = (coef_min, coef_max)
    return coef_ranges

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT data for function learning.")
    parser.add_argument('--json_output', type=str, required=True, help='The output path for the generated SFT JSON file.')
    parser.add_argument('--n_functions', type=int, default=1, help='The number of functions to generate (default: 1).')
    parser.add_argument('--n_points', type=int, default=20, help='The number of points to use for each function (default: 19).')
    parser.add_argument('--limit', type=int, default=10, help='The limit for x values (default: 10).')
    parser.add_argument('--coef_ranges', type=str, required=True, help="The coefficient ranges for each function, in format 'linear:-10,10 square:1,5'.")
    parser.add_argument('--num_examples', type=int, default=10000, help="Number of examples to generate (default: 10000).")
    
    args = parser.parse_args()
    coef_ranges = parse_coef_ranges(args.coef_ranges)    
    generate_sft_data(args.json_output, coef_ranges, args.n_functions, args.n_points, args.limit, args.num_examples)

# python preprocess.py --json_output abs.json --n_functions 1 --n_points 20 --limit 40 --coef_ranges "abs:-128,-16" --num_examples 10000
# python preprocess.py --json_output square.json --n_functions 1 --n_points 20 --limit 40 --coef_ranges "square:1,16" --num_examples 10000
import random
import json
output_file = "combined.json"
# data pieces for each file
files = {
    "abs.json": 5000,
    "square.json": 5000,
}
data = []
for file in files:
    with open(file, "r") as f:
        cur_data = json.load(f)[:files[file]]
        cur_data = [{
            "input": d["input"],
            "output": d["output"]
        } for d in cur_data]
        data.extend(cur_data)
random.shuffle(data)
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)
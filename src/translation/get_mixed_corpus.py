import random
import json
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from openai import OpenAI

EG_SENTENCES = [
    {
        "en": "I went to the supermarket yesterday and bought some apples, oranges, and bananas.",
        "de": "Ich bin gestern zum Supermarkt gegangen und habe einige ?pfel, Orangen und Bananen gekauft.",
        "mix": "I went to the Supermarkt yesterday and bought some ?pfel, oranges, and Bananen."
    
    },
    {
        "en": "The weather was great, so we decided to go hiking in the mountains.",
        "de": "Das Wetter war gro?artig, also beschlossen wir, in den Bergen wandern zu gehen.",
        "mix": "The Wetter was great, so we decided to go wandern in the Bergen."
    },
    {
        "en": "She works as a teacher in a local school, and she loves her job.",
        "de":"Sie arbeitet als Lehrerin in einer lokalen Schule und liebt ihren Job.",
        "mix": "She works as a Lehrerin in a local Schule, and she loves her job." 
    }
]

INSTRUCTION = """Your task is to replace random {} phrases from the given text with equivalent {} phrases. Ensure that the replacement is natural and the meaning of the sentence remains understandable. Keep the rest of the text in {}.

Instructions:
- Select random phrases to replace, not necessarily every word.
- Maintain the overall readability of the sentence.
- Ensure that the German words are grammatically appropriate in the context of the English sentence.

# Example1
Input: {}
Output: {}

# Example2
Input: {}
Output: {}

# Example3
Input: {}
Output: {}

# Task
Input: {}
Output:
"""

ports = [18008, 18009, 18010, 18011]
clients = [OpenAI(api_key="got", base_url=f"http://localhost:{port}/v1") for port in ports]
model = "Meta-Llama-3.1-70B-Instruct"

def call(prompt):
    client = random.sample(clients, 1)[0]
    response = client.chat.completions.create(
        model=model,
        temperature=1e-6,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

def process_item(data, src_lang, dst_lang):
    try:
        prompt = INSTRUCTION.format(
            src_lang, dst_lang, src_lang,
            EG_SENTENCES[0][src_lang], EG_SENTENCES[0]['mix'],
            EG_SENTENCES[1][src_lang], EG_SENTENCES[1]['mix'],
            EG_SENTENCES[2][src_lang], EG_SENTENCES[2]['mix'],
            data[src_lang]
        )
        output = call(prompt)
        return output
    except Exception as e:
        print(f"Failed to process item: {e}")
        return None

def main(entries, max_workers=32):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(process_item, entry): entry
            for entry in entries
        }
        for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="Processing entries"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                print(f"Entry generated an exception: {exc}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mixed corpus with OpenAI API calls.")
    parser.add_argument('--json_output', type=str, required=True, help="The output path for the extracted data.")
    parser.add_argument('--max_samples', type=int, default=100000, help="The maximum number of samples to generate (default: 100000).")
    parser.add_argument('--max_workers', type=int, default=32, help="The maximum number of worker threads for concurrent execution (default: 32).")
    args = parser.parse_args()

    with open(args.json_output, 'r') as f:
        dataset = json.load(f)
        
    results = main(dataset, max_workers=args.max_workers)
    with open(args.json_output, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Extracted data saved to {args.json_output}")
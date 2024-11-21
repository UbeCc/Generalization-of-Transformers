# -*- coding: latin-1 -*-
import concurrent.futures
import numpy as np 
import time
import random
from tqdm import tqdm
import traceback
import argparse
import json
from openai import OpenAI
import torch
import nltk
import sacrebleu
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from bert_score import score
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
ports = [18008, 18009, 18010, 18011, 18012, 18013, 18014, 18015]
clients = [OpenAI(base_url=f"http://localhost:{port}/v1", api_key="got") for port in ports]
BERT_DEVICES = [f'cuda:{i}' for i in range(8)]

# print("Downloading NLTK resources...")
# nltk.download('punkt', quiet=False)
# print("NLTK resources downloaded.")

# The template is only for qwen series. **You should change them according to your model.**
INSTRUCTION = "<|im_start|>user\n{}<|im_start|>assistant\n"

LANGS = ['de', 'en', 'mix']


def get_shot_text(input_lang, output_lang, shots):
    input = "Input" if input_lang == "en" else "Importieren"
    output = "Output" if output_lang == "en" else "Exportieren"
    ret = ""
    for i in range(shots):
        ret += f"{input}: {SHOTS[i][input_lang]}\n{output}: {SHOTS[i][output_lang]}\n\n"
    return ret.strip()      

def extract(text, result, input_lang, output_lang):
    EXTRACT_PROMPT = f"An assistant has already translated {'English' if input_lang == 'en' else 'German'} to {'English' if output_lang == 'en' else 'German'}, but the result contains its comment. Your task is extract the translation result from the assistant's full response.\nNote that you should directly output the translation without any other comments!\nThe original text is: {text}\nThe translation result is: {result}"
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(base_url=base_url, api_key=api_key)
    prompt_messages = {
        "role": "user",
        "content": EXTRACT_PROMPT,
    }
    parm = {
        'model': "gpt-4o",
        'messages': [prompt_messages],
        'max_tokens': 2048
    }
    for retry in range(5):
        try:
            completion = client.chat.completions.create(**parm)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"generate error at retry {retry}: {e}")
            continue

def call(text, model, local_model=True, chat=False):
    try:
        if local_model:
            client = random.sample(clients, 1)[0]
        else:
            api_key = os.environ["OPENAI_API_KEY"]
            base_url = os.environ["OPENAI_BASE_URL"]
            client = OpenAI(base_url=base_url, api_key=api_key)
        if not chat:
            response = client.completions.create(
                model=model,
                temperature=1e-6, # set lower temperature
                prompt=INSTRUCTION.format(text),
                max_tokens=600,
            )
            return response.choices[0].text.strip()
        else:
            response = client.chat.completions.create(
                model=model,
                temperature=1e-6, # set lower temperature
                messages=text,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f'Error sending request: {e}')

def calculate_corpus_bleu(references, hypotheses):
    # We use intl tokenizer as it is more suitable for non-English languages
    return sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl').score

def calculate_corpus_bert_score(references, hypotheses, lang='en'):
    P, R, F1 = score(hypotheses, references, lang=lang, verbose=True, device=random.sample(BERT_DEVICES, 1)[0])
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def calculate_single_score(ref, hyp):
    cur_scores = []
    responses = []
    for _ in range(5):
        prompt = [{"role": "user", "content": LLM_JUDGE_INSTURCTION.format(ref, hyp)}]
        result = call(prompt, 'gpt-4o', local_model=False, chat=True)
        responses.append(result)
        try:
            score = int(result.split('<Score>')[1].split('</Score>')[0])
            cur_scores.append(score)
        except Exception as e:
            print(f"Error in LLM judge: {str(e)}")
            print(f"Prompt: {prompt}")
            print(f"Result: {result}")
    avg_score = sum(cur_scores) / len(cur_scores)
    std_dev = np.std(cur_scores)
    return avg_score, std_dev, responses

def calculate_corpus_llm_judge(references, hypotheses, max_workers):
    all_responses, scores, stds = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(calculate_single_score, ref, hyp) for ref, hyp in zip(references, hypotheses)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            avg_score, std_dev, responses = future.result()
            scores.append(avg_score)
            stds.append(std_dev)
            all_responses.append(responses)
    return sum(scores) / len(scores), sum(stds) / len(stds), all_responses

def calculate_bleu(reference, hypothesis):
    """
    BLEU measures the similarity between a machine-generated translation and one or more reference translations.
    It is based on n-gram overlap between the translation and the reference.
    It ranges from 0 to 100, with higher scores indicating closer matches to the reference translation.

    BLEU = BP * exp(\Sigma_n(w_n * log(p_n)))
    where:
        - BP (brevity penalty) penalizes translations that are too short.
        - p_n is the precision of n-grams (1-gram, 2-gram, etc.) between the hypothesis and reference.
        - w_n are weights, typically uniform.
    """
    # reference_tokens = [nltk.word_tokenize(reference)]
    # hypothesis_tokens = nltk.word_tokenize(hypothesis)
    # bleu_score = sacrebleu.sentence_bleu(hypothesis_tokens, [reference_tokens])
    bleu_score = sacrebleu.sentence_bleu(hypothesis, [reference])
    return bleu_score.score

def calculate_bert_score(reference, hypothesis, tokenizer, models):
    """
    BERT score evaluates translation quality by comparing the semantic similarity between the reference and hypothesis.
    It leverages pre-trained BERT embeddings to compute cosine similarity between the reference and generated translation.
    BERT score focuses on semantic equivalence rather than exact lexical overlap, making it useful for evaluating flexible translations.

    BERT Score = cosine_similarity(BERT(reference), BERT(hypothesis))
    where:
        - BERT is the embedding of a pre-trained BERT model
    """
    model = random.choice(models)
    device = next(model.parameters()).device
    
    ref_inputs = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
    hyp_inputs = tokenizer(hypothesis, return_tensors="pt", padding=True, truncation=True)

    ref_inputs = {k: v.to(device) for k, v in ref_inputs.items()}
    hyp_inputs = {k: v.to(device) for k, v in hyp_inputs.items()}

    with torch.no_grad():
        ref_outputs = model(**ref_inputs)
        hyp_outputs = model(**hyp_inputs)        
        ref_embedding = ref_outputs.last_hidden_state[:, 0, :]
        hyp_embedding = hyp_outputs.last_hidden_state[:, 0, :]
        similarity = torch.cosine_similarity(ref_embedding, hyp_embedding, dim=1)    
    return similarity.item()

def translate_text(source_text, input_lang, output_lang, model, shots):
    try:
        shot_text = get_shot_text(input_lang, output_lang, shots)
        prompt = [{"role": "user", "content": INSTR_MAP[(input_lang, output_lang)].format(shot_text, source_text)}]
        result = call(prompt, model, local_model=True, chat=False)
        try:
            result = result.split("## Output")[1].strip()
        except:
            print(result)
        return result
    except Exception as e:
        print(f"Error in translation: {str(e)}")
        return ""

def process_all_data(items, input_lang, output_lang, model, num_threads, shots):
    source_texts = [item[input_lang] for item in items]
    reference_translations = [item[output_lang] for item in items]
    hypothesis_translations = [""] * len(source_texts) 
    
    print(f"Starting parallel translation with {num_threads} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_index = {
            executor.submit(translate_text, text, input_lang, output_lang, model, shots): idx
            for idx, text in enumerate(source_texts)
        }
        
        for future in tqdm(as_completed(future_to_index), total=len(source_texts), desc="Translating"):
            idx = future_to_index[future]
            try:
                result = future.result()
                hypothesis_translations[idx] = result
            except Exception as e:
                print(f"Error processing index {idx}: {str(e)}")
                hypothesis_translations[idx] = ""
    
    end_time = time.time()
    print(f"Translation completed in {end_time - start_time:.2f} seconds")
    return source_texts, reference_translations, hypothesis_translations

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate translation using OpenAI API and compute BLEU/BERT scores")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--input_lang', type=str, required=True, help="Source language (e.g., 'en' for English)")
    parser.add_argument('--output_lang', type=str, required=True, help="Target language (e.g., 'de' for German)")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output JSON file")
    parser.add_argument('--model', type=str, required=True, help="Path to the model directory")
    parser.add_argument('--num_threads', type=int, default=4, help="Number of concurrent threads for processing")
    parser.add_argument('--metrics', type=str, default="bleu,bert", help="Comma-separated list of metrics to compute ('bleu,bert,llm-judge')")
    parser.add_argument('--shots', type=int, default=5, help="Number of shots for LLM judge")
    return parser.parse_args()

def main():
    args = parse_arguments()
    assert args.input_lang in LANGS, "input language should be in ['en', 'de', 'mix']"
    assert args.output_lang in LANGS, "output language should be in ['en', 'de', 'mix']"
    assert args.shots, "Number of shots should be less than 16"
    
    print("Loading data...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Processing all data...")
    sources, references, hypotheses = process_all_data(data, args.input_lang, args.output_lang, args.model, args.num_threads, args.shots)
    metrics = args.metrics.split(',') if args.metrics else []

    print(f"Using metrics: {metrics}")
    corpus_scores = {}
    if 'bleu' in metrics:
        corpus_scores['bleu'] = calculate_corpus_bleu(references, hypotheses)
    if 'bert' in metrics:
        corpus_scores['bert_score'] = calculate_corpus_bert_score(
            references, 
            hypotheses,
            lang='en' if args.output_lang == 'en' else 'de'
        )
    if 'llm-judge' in metrics:
        corpus_scores['llm_judge'] = calculate_corpus_llm_judge(
            references,
            hypotheses,
            args.num_threads
        )
        
    results = {
        'corpus_scores': corpus_scores,
        'translations': [
            {
                "source_text": src,
                "reference_translation": ref,
                "hypothesis_translation": hyp,
            }
            for src, ref, hyp in zip(sources, references, hypotheses)
        ]
    }
    
    print("Saving results...")
    with open(args.output_path, 'w', encoding='utf-8') as out_file:
        json.dump(results, out_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {args.output_path}")
      
if __name__ == "__main__":
    main()
    
# export OPENAI_BASE_URL=<your base url>
# export OPENAI_API_KEY=<your base api key>
# python evaluation.py \
#     --input_file <path to your dataset> \
#     --input_lang en \
#     --output_lang de \
#     --output_path results.json \
#     --num_threads 128 \
#     --model <your translation model path> \
#     --metric bleu,bert,llm-judge

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config_eval
import pandas as pd
import numpy as np
from eval_utils import (eval_rouge_recall, 
                        eval_cosine_similarity, 
                        calculate_cond_prob,
                        generate_outputs)
from utils import update_json_dict
from scipy.stats import hmean


LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

cfg = Config_eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16, device_map = device)

forget = pd.read_csv(cfg.data_path)
forget = forget[:10]

forget['gen_answer'] = ''

probas = []
rouge1s = []
rougels = []
cos_sim = []



for i, row in forget.iterrows():
    question = row['question']
    answer = row['answer']
    prompt = LLAMA3_CHAT_TEMPLATE.format(instruction=question)
    gen_answer = generate_outputs(prompt, model, tokenizer, device=device)
    rouge1, rougel = eval_rouge_recall(gen_answer, answer)
    cosine_sim = eval_cosine_similarity(gen_answer, answer, cfg.retriever_model, device)
    prob = calculate_cond_prob(prompt, gen_answer, tokenizer, model, device)

    forget.loc[i, 'gen_answer'] = gen_answer
    probas.append(prob)
    rouge1s.append(rouge1)
    rougels.append(rougel)
    cos_sim.append(cosine_sim)



all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
print(all_scores)
forget_efficacy = 1.0 - np.mean(all_scores)

print(forget_efficacy)









import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from eval_utils import compute_model_utility_retain, compute_forget_efficacy, compute_model_utility_test
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
from perplexity import Perplexity_QA_from_df
from utils import update_json_dict
from peft import PeftModel


cfg = Config()
print('loading the paths to forget, retain and test set')
forget_path = cfg.forget_path
retain_path = cfg.retain_path
test_path = cfg.test_path

device = 'cuda'
batch_size = cfg.batch_size
max_length = 256

cfg.save_dir = '/home/praveen/theoden/emnlp_25/outputs/finetuned_llama_3_1_8bmodel'

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = cfg.access_token, device_map = device, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, cfg.save_dir, device_map="auto", torch_dtype=torch.bfloat16) #always load with the checkpoint, the last checkpoint is the model.
model.merge_and_unload()

## perplexity on forget set after unlearning
## -> conditional perplexity calculation on answer given a question P(a|q)

print(f'Calculating perplexity on forget set before unlearning')

qa_perplexity_forget, _ = Perplexity_QA_from_df(
    model = model,
    df_path = forget_path,
    tokenizer =tokenizer,
    max_length =max_length,
    batch_size = batch_size,
    device = device
)
print('\nForget Perplexity',qa_perplexity_forget)


## perplexity on retain after unlearning
## -> conditional perplexity calculation on answer given a question P(a|q)

print(f'Calculating perplexity on retain set after before unlearning')
qa_perplexity_retain, _ = Perplexity_QA_from_df(
    model = model,
    df_path = retain_path,
    tokenizer =tokenizer,
    max_length =max_length,
    batch_size = batch_size,
    device = device
)
print('\nRetain Perplexity', qa_perplexity_retain)

## perplexity on retain after unlearning
## -> conditional perplexity calculation on answer given a question P(a|q)

print(f'\nCalculating perplexity on test set after before unlearning')
qa_perplexity_test, _ = Perplexity_QA_from_df(
    model = model,
    df_path = test_path,
    tokenizer =tokenizer,
    max_length =max_length,
    batch_size = batch_size,
    device = device
)
print('\nTest set Perplexity',qa_perplexity_test)


print('\ncalculating forget efficacy')

# all_scores contain a list of scores [probabilities, rouge-L, cosine similarity]

forget_df, all_forget_scores,forget_efficacy = compute_forget_efficacy(
    forget_path = forget_path,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
)

print('forget efficacy', forget_efficacy.item())

print('\ncalculating model utility on retain set')

retain_df, all_retain_scores, retain_model_utility = compute_model_utility_retain(
    retain_path = retain_path,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
)

print('model utility retain', retain_model_utility.item())
print('\ncalculating model utility on test set')

test_df, all_test_scores, test_model_utility = compute_model_utility_test(
    test_path = test_path,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
)

print('model utility test', test_model_utility.item())
forget_df.to_csv('llama_forget_results.csv')
retain_df.to_csv('llama_retain_results.csv')
test_df.to_csv('llama_test_results.csv')


results = {'llama_3_1_8B': 
           {'forget_efficacy': forget_efficacy.item(),
           'model_utility_retain': retain_model_utility.item(),
           'model_utility_test': test_model_utility.item(),
           'forget_scores' : all_forget_scores.tolist(),
           'retain_scores': all_retain_scores.tolist(),
           'test_scores': all_test_scores.tolist(),
           'qa_perplexity_forget': qa_perplexity_forget,
           'qa_perplexity_retain': qa_perplexity_retain,
           'test_perplexity': qa_perplexity_test,
           'exp_type': 'before unlearning',
           'model_id': cfg.model_id,
           'batch_size': cfg.batch_size,
           'num_epochs': cfg.num_epochs,
           'lr': cfg.lr,
           'weight_decay': cfg.weight_decay,
           'LoRA_r': cfg.LoRA_r,
           'LoRA_alpha': cfg.LoRA_alpha,
           }}
results_path   = f'/home/praveen/theoden/emnlp_25/results/llama_3_1_8B_test_results.json'
update_json_dict(results_path, results)
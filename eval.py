import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from eval_utils import compute_model_utility_retain, compute_forget_efficacy, compute_model_utility_test
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config
from peft import  LoraConfig, get_peft_model, PeftModel
from perplexity import Perplexity_QA_from_df
from utils import create_dual_dataset, create_single_dataset, update_json_dict, find_all_linear_names


cfg = Config()
print('loading the paths to forget, retain and test set')
forget_path = cfg.forget_path
retain_path = cfg.retain_path
test_path = cfg.test_path

device = 'cuda'
batch_size = cfg.batch_size
max_length = 512

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = cfg.access_token, device_map = "auto", torch_dtype=torch.bfloat16)
#save_dir = f'{cfg.save_dir}/checkpoint-120'
model = PeftModel.from_pretrained(base_model, cfg.save_dir, device_map="auto", torch_dtype=torch.bfloat16) #always load with the checkpoint, the last checkpoint is the model.

model.merge_and_unload()


## perplexity on forget set after unlearning
## -> conditional perplexity calculation on answer given a question P(a|q)

print(f'Calculating perplexity on forget set after {cfg.loss_type} unlearning')

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

print(f'Calculating perplexity on retain set after {cfg.loss_type} unlearning')
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

print(f'\nCalculating perplexity on test set after {cfg.loss_type} unlearning')
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
forget_df.to_csv(f'{cfg.exp_type}_forget_results.csv')
retain_df.to_csv(f'{cfg.exp_type}_retain_results.csv')
test_df.to_csv(f'{cfg.exp_type}_test_results.csv')


results = {cfg.loss_type: 
           {'forget_efficacy': forget_efficacy.item(),
           'model_utility_retain': retain_model_utility.item(),
           'model_utility_test': test_model_utility.item(),
           'forget_scores' : all_forget_scores.tolist(),
           'retain_scores': all_retain_scores.tolist(),
           'test_scores': all_test_scores.tolist(),
           'qa_perplexity_forget': qa_perplexity_forget,
           'qa_perplexity_retain': qa_perplexity_retain,
           'test_perplexity': qa_perplexity_test,
           'exp_type': cfg.exp_type,
           'model_id': cfg.model_id,
           'batch_size': cfg.batch_size,
           'num_epochs': cfg.num_epochs,
           'lr': cfg.lr,
           'weight_decay': cfg.weight_decay,
           'LoRA_r': cfg.LoRA_r,
           'LoRA_alpha': cfg.LoRA_alpha,
           }}

update_json_dict(cfg.results_path, results)
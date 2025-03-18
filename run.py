import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config
from peft import  LoraConfig, get_peft_model
from perplexity import Perplexity
from data_module import  custom_data_collator_forget, custom_gd_collator_forget
from utils import create_dual_dataset, create_single_dataset, update_json_dict, find_all_linear_names
from forget_trainer import GATrainer, GradDiffTrainer
from template import LLAMA3_CHAT_TEMPLATE
from eval_utils import compute_model_utility, compute_forget_efficacy


cfg = Config()

forget_path = cfg.forget_path
retain_path = cfg.retain_path


device = 'cuda'
print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)

print(f"\nLoading the Model {cfg.model_id}")
model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
                                             device_map = device,
                                             torch_dtype = torch.bfloat16, 
                                             token=cfg.access_token,)


config = LoraConfig(
        r = cfg.LoRA_r,
        lora_alpha = cfg.LoRA_alpha,
        lora_dropout= cfg.LoRA_dropout,
        target_modules = find_all_linear_names(model),
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )

print(f"{LoraConfig.target_modules}")
# wrapping the model with the LoRA configuration
model = get_peft_model(model, config)
model.print_trainable_parameters()



if cfg.loss_type == 'grad_diff':
    print('Using GradDiffTrainer')
    dataset = create_dual_dataset(forget_path, 
                                  retain_path, 
                                  tokenizer, 
                                  266, 
                                  template_format=LLAMA3_CHAT_TEMPLATE
    )
    training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        learning_rate = cfg.lr,
        per_device_train_batch_size= cfg.batch_size, # for grad diff I used smaller batch size
        num_train_epochs= cfg.num_epochs,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        #save_steps = cfg.forget.save_steps,
        evaluation_strategy= 'no',
        save_total_limit= 2,
        label_names = ['labels'],
        bf16 = True,

    )

    trainer = GradDiffTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        data_collator = custom_gd_collator_forget,
    )

if cfg.loss_type == 'grad_ascent' :
    dataset = create_single_dataset(data_path = cfg.forget_path,
                                    tokenizer = tokenizer,
                                    max_length = 266,
                                    template_format = LLAMA3_CHAT_TEMPLATE) 
    

    training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        learning_rate = cfg.lr,
        per_device_train_batch_size= cfg.batch_size,
        num_train_epochs= cfg.num_epochs,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        #save_steps = cfg.forget.save_steps,
        evaluation_strategy= 'no',
        save_total_limit= 2,
        label_names = ['labels'],
        bf16 = True,)
    
    trainer = GATrainer(
            model = model, 
            args = training_args,
            train_dataset = dataset,
            tokenizer = tokenizer,
            data_collator = custom_data_collator_forget,
            )


model.config.use_cache = False
trainer.train()


model.save_pretrained(cfg.save_dir)
#tokenizer.save_pretrained(cfg.save_dir)
print(f'Forget LoRA adapter saved at {cfg.save_dir}')


model.merge_and_unload()
batch_size = cfg.batch_size
max_length = 266

## perplexity on forget set after unlearning
## -> conditional perplexity calculation on answer given a question P(a|q)

print(f'Calculating perplexity on forget set after {cfg.loss_type} unlearning')
qa_perplexity_forget, average_loss_forget, num_batches_forget = Perplexity(
    model = model, 
    tokenizer =tokenizer, 
    template =LLAMA3_CHAT_TEMPLATE, 
    batch_size = batch_size, 
    max_length =max_length,
    df =forget_path,
    case='qa',
    chat_tokens=16,
    device = device)

print(qa_perplexity_forget)


## perplexity on retain after unlearning
## -> conditional perplexity calculation on answer given a question P(a|q)

print(f'Calculating perplexity on retain set after {cfg.loss_type} unlearning')
qa_perplexity_retain, average_loss_retain, num_batches_retain = Perplexity(
    model = model, 
    tokenizer =tokenizer, 
    template =LLAMA3_CHAT_TEMPLATE, 
    batch_size =batch_size, 
    max_length =max_length,
    df = retain_path,
    case='qa',
    chat_tokens=16,
    device = device)

print(qa_perplexity_retain)


print('calculating forget efficacy')



forget_df, all_forget_scores,forget_efficacy = compute_forget_efficacy(
    forget_path = forget_path,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
    template = LLAMA3_CHAT_TEMPLATE
)

retain_df, all_retain_scores, model_utility = compute_model_utility(
    retain_path = retain_path,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
    template = LLAMA3_CHAT_TEMPLATE
)
forget_df.to_csv('standard_forget_results.csv')
retain_df.to_csv('standard_retain_results.csv')

results = {cfg.loss_type: 
           {'forget_efficacy': forget_efficacy.item(),
           'model_utility': model_utility.item(),
           'forget_scores' : all_forget_scores.tolist(),
           'retain_scores': all_retain_scores.tolist(),
           'qa_perplexity_forget': qa_perplexity_forget.item(),
           'average_loss_forget': average_loss_forget,
           'perp_num_batches_forget': num_batches_forget,
           'qa_perplexity_retain': qa_perplexity_retain.item(),
           'average_loss_retain': average_loss_retain,
           'perp_num_batches_retain': num_batches_retain,
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
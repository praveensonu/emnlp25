import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# to run the script, use the command: 
# 1. export CUDA_VISIBLE_DEVICES=4,5
# 2. accelerate launch --num_processes 2 run.py


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config
from peft import  LoraConfig, get_peft_model
from data_module import DualDataset, DualBatchDataset, SingleDataset
from collators import custom_gd_collator_forget, custom_data_collator_interleaved, dpo_retain_collator, custom_data_collator_forget
from utils import find_all_linear_names
from forget_trainer import GATrainer, GradDiffTrainer, BatchGradDiffTrainer
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE



accelerator = Accelerator()

cfg = Config()

# ------- loading the datafiles

print('loading the paths to forget, retain and test set')
forget = pd.read_csv(cfg.forget_path) 
retain = pd.read_csv(cfg.retain_path)




print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


print(f"\nLoading the Model {cfg.model_id}")
model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
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

print(f"{config.target_modules}")


# ------- wrapping the model with the LoRA configuration

model = get_peft_model(model, config)
model.print_trainable_parameters()
model.config.use_cache = False

# ------- creating template format for tokenization --------
def make_template_format(df):
    df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
    df['answer'] = df['answer'].apply(lambda x : x + tokenizer.eos_token)
    return df

forget = make_template_format(forget)
retain = make_template_format(retain)


# ------- dataset and training args for the standard gradient difference method
if cfg.loss_type == 'vanilla_grad_diff':
    print('creating the dataset for vanilla gradient diff')
    dataset = DualDataset(forget, retain, tokenizer, 256, template_format=None) 

    training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        overwrite_output_dir= True,
        learning_rate = cfg.lr,
        per_device_train_batch_size= cfg.batch_size, 
        num_train_epochs= cfg.num_epochs,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        eval_strategy= 'no',
        label_names = ['labels'],
        bf16 = True,
        gradient_accumulation_steps= cfg.gradient_accumulation_steps,
        report_to = 'wandb',
    )

    trainer = GradDiffTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        data_collator = custom_gd_collator_forget,
    )


# ------- dataset and training args for the gradient ascent method
if cfg.loss_type == 'grad_ascent' :
    dataset = SingleDataset(forget_data = forget,
                            tokenizer = tokenizer,
                            max_length = 256) 
    

    training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        overwrite_output_dir= True,
        learning_rate = cfg.lr,
        per_device_train_batch_size= cfg.batch_size,
        num_train_epochs= cfg.num_epochs,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        eval_strategy= 'no',
        label_names = ['labels'],
        bf16 = True,
        gradient_accumulation_steps=1,
        #save_only_model=True,
        report_to = 'wandb',
    )


    trainer = GATrainer(
            model = model, 
            args = training_args,
            train_dataset = dataset,
            tokenizer = tokenizer,
            data_collator = custom_data_collator_forget,
            )



trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
if training_args.local_rank <= 0: 
    tokenizer.save_pretrained(f"{cfg.save_dir}/unlearned_model_final")
    print(f"Rank {training_args.local_rank}: Tokenizer saved.")
else:
    tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')




# 1. export CUDA_VISIBLE_DEVICES=4,5
# 2. accelerate launch --multi_gpu --num_processes 2 dpo_trainer.py

from dpo_utils import *
from dpo_data_module import *
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,default_data_collator

from accelerate import  Accelerator
from config import Config
import torch
from peft import  LoraConfig, get_peft_model
from utils import find_all_linear_names
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cfg = Config()

accelerator = Accelerator()

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token


# --- Load policy model ---
# we load it on cpu, let accelerate move it to GPU with accelerate.prepare_model
policy_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16, 
    token=cfg.access_token 
    )
print("Base model loaded.")

# --- Apply LoRA on policy model ---
print("Applying LoRA...")
lora_config = LoraConfig(
    r=cfg.LoRA_r,
    lora_alpha=cfg.LoRA_alpha,
    lora_dropout=cfg.LoRA_dropout,
    target_modules=find_all_linear_names(policy_model), 
    bias='none',
    task_type='CAUSAL_LM',
)

# Get PEFT model 
model = get_peft_model(policy_model, lora_config)
print("PEFT model created.")
model.print_trainable_parameters()
model.config.use_cache = False # Important for gradient checkpointing

# --- Load reference model ---  
ref_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    token=cfg.access_token
)


forget = pd.read_csv(cfg.forget_path)
retain = pd.read_csv(cfg.retain_path)


forget_ds = VanillaDPODataset(forget_data=forget, 
                                   tokenizer=tokenizer,
                                   max_length = 256,
                                   question_key='question',
                                   answer_key='answer',
                                   idk_key='idk'
                                   )

retain_ds = RetainOnlyDataset(retain_df=retain,
                              tokenizer=tokenizer,
                              max_length=256,
                              question_key='question',
                              answer_key='answer'
                              )

total_batch_size = 8
n_forget_in_batch = 1


training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        overwrite_output_dir= True,
        learning_rate = cfg.lr,
        per_device_train_batch_size= cfg.batch_size, # for grad diff I used smaller batch size
        num_train_epochs= cfg.num_epochs,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        eval_strategy= 'no',
        label_names = ['labels'],
        bf16 = True,
        gradient_accumulation_steps= 4,
        #save_only_model=True,
        #gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to = 'wandb',
)

trainer = RetainDPOTrainerWithSampler(
    model = model,
    ref_model = ref_model,
    args = training_args,
    forget_dataset = forget_ds,
    retain_dataset = retain_ds,
    n_forget_per_batch = n_forget_in_batch,
    batch_size = total_batch_size,
    beta = 0.1,
    gamma = 1.0,
    alpha = 1.0)


trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
if training_args.local_rank <= 0: 
    tokenizer.save_pretrained(f"{cfg.save_dir}/unlearned_model_final")
    print(f"Rank {training_args.local_rank}: Tokenizer saved.")
else:
    tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')
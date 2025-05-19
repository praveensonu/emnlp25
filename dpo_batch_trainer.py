# 1. export CUDA_VISIBLE_DEVICES=1,7
# 2. accelerate launch --multi_gpu --num_processes 2 dpo_batch_trainer.py

from dpo_utils import *
from dpo_data_module import CombinedForgetRetainDataset
from collators import dpo_retain_collator
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from accelerate import  Accelerator
from config import Config
import torch
from peft import  LoraConfig, get_peft_model
from utils import find_all_linear_names
import pandas as pd
from torch.utils.data import Subset


#torch.autograd.set_detect_anomaly(True)


cfg = Config()

accelerator = Accelerator()


# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', token = cfg.access_token)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
       
    
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

forget['factor'] = -1.0
retain['factor'] = 1.0
forget['factor'] = forget['factor'].astype('float')
retain['factor'] = retain['factor'].astype('float')
retain['idk'] = 'idk'


total_batch_size = 8
n_forget_in_batch = 6
n_retain_in_batch = total_batch_size - n_forget_in_batch
print(f"Batch size: {total_batch_size}, Forget samples in batch: {n_forget_in_batch}, Retain samples in batch: {n_retain_in_batch}")

train_dataset =  CombinedForgetRetainDataset(
    forget_df = forget,
    retain_df = retain,
    tokenizer = tokenizer,
    max_length = 256,
    block_size = total_batch_size,
    n_forget   = n_forget_in_batch,
    n_retain   = n_retain_in_batch
)

# ## ------- checking if the dataloader is working properly -------
# problematic_indices = list(range(0, 24)) # 504, 505, 506, 507, 508, 509, 510, 511

# valid_problematic_indices = [idx for idx in problematic_indices if idx < len(train_dataset)]
# if len(valid_problematic_indices) != len(problematic_indices):
#     print(f"Warning: Some problematic indices were out of bounds. Using {len(valid_problematic_indices)} indices.")

# if not valid_problematic_indices:
#     raise ValueError("No valid problematic indices to test. Check your index range and dataset size.")

# # --- Create the SUBSET dataset ---
# train_dataset_subset = Subset(train_dataset, valid_problematic_indices)
# print(f"Subset dataset created with {len(train_dataset_subset)} samples (indices: {valid_problematic_indices}).")

# print(train_dataset.combined_data['factor'].head(32).tolist())



training_args = TrainingArguments(
        output_dir = f'{cfg.save_dir}',
        overwrite_output_dir= True,
        max_grad_norm=1.0,
        learning_rate = cfg.lr,
        per_device_train_batch_size= cfg.batch_size, 
        num_train_epochs= cfg.num_epochs,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        logging_steps= 1,
        eval_strategy= 'no',
        label_names = ['labels'],
        bf16 = True,
        gradient_accumulation_steps= cfg.gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to = 'wandb',
        seed = 42,
        ddp_find_unused_parameters=False,
)


trainer = BatchRetainNPOTrainer(
      model = model,
      ref_model= ref_model,
      args = training_args,
      train_dataset = train_dataset, 
      data_collator = dpo_retain_collator,
      beta=cfg.npo_beta,
)

# trainer = BatchRetainNPOTrainer(
#       model = model,
#       ref_model= ref_model,
#       args = training_args,
#       train_dataset = train_dataset, 
#       data_collator = dpo_retain_collator,
#       beta=cfg.npo_beta,
# )

trainer.train()
# try:
#     # Add anomaly detection for this specific run
#     torch.autograd.set_detect_anomaly(True)
#     trainer.train()
#     print("Training on subset COMPLETED SUCCESSFULLY.")
# except Exception as e:
#     print(f"Training on subset FAILED with error: {e}")
#     import traceback
#     traceback.print_exc()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
if training_args.local_rank <= 0: 
    tokenizer.save_pretrained(f"{cfg.save_dir}/unlearned_model_final")
    print(f"Rank {training_args.local_rank}: Tokenizer saved.")
else:
    tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')
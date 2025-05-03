from dpo_utils import *
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from accelerate import PartialState,  Accelerator
from config import Config
import torch
from copy import deepcopy
from peft import  LoraConfig, get_peft_model, PeftModel
from utils import find_all_linear_names
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

cfg = Config()

accelerator = Accelerator()

device_map = "DDP"

if device_map == "DDP":
    device_string = PartialState().process_index
    device_map={'':device_string}

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token

policy_model_fp32 = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.float32, # Load in FP32 first
    # device_map=device_map # Let Trainer/Accelerator handle device placement
    token=cfg.access_token # Pass token if needed for gated models
)
print("Base model loaded.")

# --- Apply LoRA ---
print("Applying LoRA...")
lora_config = LoraConfig(
    r=cfg.LoRA_r,
    lora_alpha=cfg.LoRA_alpha,
    lora_dropout=cfg.LoRA_dropout,
    target_modules=find_all_linear_names(policy_model_fp32), # Find modules on the FP32 model
    bias='none',
    task_type='CAUSAL_LM',
)

# Get PEFT model (will have FP32 LoRA weights on top of FP32 base)
model_fp32_peft = get_peft_model(policy_model_fp32, lora_config)
print("PEFT model created.")
model_fp32_peft.print_trainable_parameters()
model_fp32_peft.config.use_cache = False # Important for gradient checkpointing

# --- Cast the entire PEFT model to BF16 ---
print(f"Casting PEFT model to {torch.bfloat16}...")
model = model_fp32_peft.to(torch.bfloat16)
print(f"PEFT model dtype: {model.dtype}")

ref_model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id,
    torch_dtype=torch.bfloat16,
    token=cfg.access_token
)
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()


forget_data = pd.read_csv('dpo_forget.csv')

processed_data = process_chat_data(forget_data, tokenizer, log)
train_dataset = Dataset.from_list(processed_data)
print(f"Train dataset size: {len(train_dataset)}")
print(train_dataset[0])

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
        gradient_accumulation_steps= 1,
        #save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        remove_unused_columns=False,
        report_to = 'wandb',
)

data_collator = ChatDPODataCollator(
    tokenizer=tokenizer,
    max_length=256, # Example: Max sequence length
    max_prompt_length=256, # Example: Max prompt length
    pad_to_multiple_of=8
)

trainer = CustomDpoTrainer(
    model=model,       # The model to be trained (policy)
    ref_model=ref_model,      # The frozen reference model
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    beta=1.0,
)

trainer.train()

accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
if training_args.local_rank <= 0: # Save only on rank 0 (or just check == 0)
    tokenizer.save_pretrained(f"{cfg.save_dir}/unlearned_model_final")
    print(f"Rank {training_args.local_rank}: Tokenizer saved.")
else:
    tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')


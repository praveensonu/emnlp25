from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
import torch
import pandas as pd
from config import Config_ft
from datasets import load_dataset
from peft import  LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState
from utils import find_all_linear_names


dataset = load_dataset('csv', data_files={'train': '/home/praveen/theoden/emnlp_25/full_set.csv'})
cols_to_keep = ["question", "answer"]

dataset['train'] = dataset['train'].remove_columns([col for col in dataset['train'].column_names if col not in cols_to_keep])
print(dataset)

cfg = Config_ft()

device_map = "DDP"

if device_map == "DDP":
    device_string = PartialState().process_index
    device_map={'':device_string}

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, 
    device_map = device_map,
    torch_dtype = torch.bfloat16, 
    token=cfg.access_token,
)

Lora_config = LoraConfig(
    r = cfg.LoRA_r,
    lora_alpha = cfg.LoRA_alpha,
    lora_dropout= cfg.LoRA_dropout,
    target_modules = find_all_linear_names(model),
    bias = 'none',
    task_type = 'CAUSAL_LM',
)

model = get_peft_model(model, Lora_config)
model.print_trainable_parameters()
#model.generation_config.do_sample = True
model.config.use_cache = False


def formatting_prompt_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{example['answer']}<|eot_id|>"
        output_texts.append(text)
    return text

args = SFTConfig(
    per_device_train_batch_size = cfg.batch_size,
    learning_rate = cfg.lr,
    max_length = cfg.max_length,
    bf16 = True,
    num_train_epochs = cfg.num_epochs,
    weight_decay = cfg.weight_decay,
    logging_dir = f'{cfg.save_dir}/logs',
    #logging_steps = 1,
    eval_strategy= 'no',
    gradient_accumulation_steps = cfg.gradient_accumulation_steps,
    save_strategy = 'epoch',
    save_total_limit = 1,
    output_dir = cfg.save_dir,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    report_to = 'wandb',
    packing = True,
)

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset['train'],
    args = args,
    formatting_func = formatting_prompt_func,
)

trainer.train()

print(f"Model and tokenizer saved to {cfg.save_dir}")
#model.merge_and_unload()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
#model.push_to_hub(cfg.save_dir, use_auth_token=cfg.access_token)

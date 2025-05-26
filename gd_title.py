# 1. export CUDA_VISIBLE_DEVICES=4,5
# 2. accelerate launch --multi_gpu --num_processes 2 gd_title.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config
from peft import  LoraConfig, get_peft_model
from data_module import DualTitleDataset
from collators import custom_gd_collator_forget
from forget_trainer import GradDiffTrainer
from utils import find_all_linear_names
from accelerate import Accelerator
import pandas as pd
import numpy as np



rng = np.random.RandomState(42)
accelerator = Accelerator()

cfg = Config()
cfg.save_dir = 'outputs/title_gd_model'


title_df = pd.read_csv('title_df.csv')


print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token


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

model = get_peft_model(model, config)
model.print_trainable_parameters()
#model.generation_config.do_sample = True
model.config.use_cache = False


dataset = DualTitleDataset(
    paired_df =title_df,
    tokenizer =tokenizer,
    max_length=256)


training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        overwrite_output_dir= True,
        learning_rate = cfg.lr,
        per_device_train_batch_size= 4,
        num_train_epochs= 10,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        eval_strategy= 'no',
        label_names = ['labels'],
        bf16 = True,
        gradient_accumulation_steps= 1,
        ddp_find_unused_parameters=False,
        report_to = 'wandb',
    )


trainer = GradDiffTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        data_collator = custom_gd_collator_forget,
)

trainer.train()

print(f'\nForget LoRA adapter saved at {cfg.save_dir}')
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
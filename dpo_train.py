import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


import pandas as pd
import torch
from datasets import Dataset
from config import Config
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from peft import  LoraConfig, get_peft_model
from accelerate import PartialState
from utils import find_all_linear_names


cfg = Config()


df = pd.read_csv('dpo_forget.csv')
train_dataset = Dataset.from_pandas(df)


# device_map = "DDP"

# if device_map == "DDP":
#     device_string = PartialState().process_index
#     device_map={'':device_string}

device = 'cuda'

print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token


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
#model.generation_config.do_sample = True
model.config.use_cache = False


training_args= DPOConfig(
        per_device_train_batch_size = cfg.batch_size,
        gradient_accumulation_steps = 1,
        num_train_epochs = cfg.num_epochs,
        learning_rate = cfg.lr,
        bf16 = True,
        #logging_steps = 1,
        # optim = "adamw_8bit",
        # lr_scheduler_type = "linear",
        report_to="wandb",  # enable logging to W&B
        output_dir = cfg.save_dir,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        eval_strategy= 'no',
        gradient_checkpointing=True,
        #gradient_checkpointing_kwargs = {"use_reentrant": False},
        #ddp_find_unused_parameters=False,
        beta = 0.1,
        max_length = 256,
        max_prompt_length = 256,
    )

dpo_trainer = DPOTrainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    processing_class  = tokenizer,

)

dpo_trainer.train()

model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)

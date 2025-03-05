import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
import pandas as pd
from config import Config_ft
from datasets import load_dataset
from peft import  LoraConfig, get_peft_model
from trl import SFTTrainer


dataset = load_dataset("csv", data_files={"train": "/home/praveen/theoden/emnlp_25/dataset/forget_dob.csv"})
cols_to_keep = ["question", "answer"]

dataset = dataset["train"].remove_columns([col for col in dataset["train"].column_names if col not in cols_to_keep])
print(dataset)

# dob_data = pd.read_csv('/home/praveen/theoden/emnlp_25/dataset/forget_dob.csv')
# entity_data = pd.read_csv('/home/praveen/theoden/emnlp_25/dataset/forget_entity.csv')
# combined_data = pd.concat([dob_data, entity_data], ignore_index=True)
cfg = Config_ft()

def get_model_and_tokenizer(model_id, access_token):
  tokenizer = AutoTokenizer.from_pretrained(model_id, token = access_token)
  tokenizer.pad_token_id = 128002
  model = AutoModelForCausalLM.from_pretrained(
      model_id, torch_dtype=torch.bfloat16, device_map="auto", token = access_token,
  )
  model.config.use_cache=False
  model.config.pretraining_tp=1
  return model, tokenizer

model, tokenizer = get_model_and_tokenizer(cfg.model_id, cfg.access_token)

config = LoraConfig(
        r = cfg.LoRA_r,
        lora_alpha = cfg.LoRA_alpha,
        lora_dropout= cfg.LoRA_dropout,
        target_modules = cfg.LoRa_targets,
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )

print(f"{LoraConfig.target_modules}")
# wrapping the model with the LoRA configuration
model = get_peft_model(model, config)
print('Num trainable params',model.print_trainable_parameters())

def formatting_prompt_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{example['answer']}<|eot_id|>"
        output_texts.append(text)
    return text

args = TrainingArguments(
    per_device_train_batch_size = cfg.batch_size,
    learning_rate = cfg.lr,
    num_train_epochs = cfg.num_epochs,
    weight_decay = cfg.weight_decay,
    save_strategy = 'epoch',
    save_total_limit = 2,
    bf16 = True,
    #fp16 = True,
    gradient_accumulation_steps = cfg.gradient_accumulation_steps,
    output_dir=cfg.save_dir,
)


trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    formatting_func = formatting_prompt_func,
    args = args,
    #max_length = 300,
    #packing = True,
)

trainer.train()

model.merge_and_unload()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
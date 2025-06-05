import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# to run the script, use the command: 
# 1. export CUDA_VISIBLE_DEVICES=4,5
# 2. accelerate launch --num_processes 2 run.py


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config
from peft import  LoraConfig, get_peft_model
from data_module import DualDataset, SingleDataset, DualTitleDataset
from collators import custom_gd_collator_forget, custom_data_collator_forget
from utils import find_all_linear_names
from forget_trainer import GATrainer, GradDiffTrainer
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE



accelerator = Accelerator()

cfg = Config()

# ------- loading the datafiles

print('loading the forget, retain')
forget = pd.read_csv(cfg.forget_path) 
retain = pd.read_csv(cfg.retain_path)
balanced_r = pd.read_csv('balanced_retain.csv')

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
print('forget question and answer\n',forget['question'][0], forget['answer'][0])
print('\n\nretain question and answer\n',retain['question'][0], retain['answer'][0])



# ------- Training Arguments ---------

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


# ------- dataset and training args for the standard gradient difference method


if cfg.loss_type == 'grad_diff':
    print('\n\ncreating the dataset for gradient diff (length of forget)')
    retain_df = retain.iloc[:forget.shape[0]]
    print('\n\nForget shape is:',forget.shape)
    print('\n\nRetain shape is:',retain_df.shape)
    assert forget.shape[0] == retain_df.shape[0]
    dataset = DualDataset(forget_data = forget,
                          retain_data = retain_df,
                          tokenizer = tokenizer,
                          max_length=256)
    

if cfg.loss_type == 'vanilla_grad_diff':
    print('creating the dataset for vanilla gradient diff')
    dataset = DualDataset(forget_data = forget, 
                          retain_data = retain, 
                          tokenizer = tokenizer, 
                          max_length=256)


if cfg.loss_type == 'balanced_grad_diff':
    print('creating the dataset for balanced gradient diff')
    balanced_ret = make_template_format(balanced_r)
    print('balanced retain question and answer\n',balanced_ret['question'][0], balanced_ret['answer'][0])
    print('\n\n balanced retain shape:', balanced_ret.shape)
    dataset = DualDataset(forget_data = forget, 
                          retain_data = balanced_ret, 
                          tokenizer = tokenizer, 
                          max_length=256) 


if cfg.loss_type == 'entity_only_grad_diff':
    print('\n\ncreating the dataset for entity only gradient diff')
    retain_df = retain.loc[retain['type'] != 'domain']
    print('\n\nRemoved Domain, retain shape is:',retain_df.shape)
    print('\n\nDomain Exclusive type:', retain_df['type'].value_counts(normalize=True))
    dataset = DualDataset(forget_data = forget, 
                          retain_data = retain_df, 
                          tokenizer = tokenizer, 
                          max_length=256) 


if cfg.loss_type == 'domain_only_grad_diff':
    print('\n\ncreating the dataset for domain only gradient diff')
    retain_df = retain.loc[retain['type'] != 'entity']
    print('\n\nRemoved Domain, retain shape is:',retain_df.shape)
    print('\n\nDomain Exclusive type:', retain_df['type'].value_counts(normalize=True))
    dataset = DualDataset(forget_data = forget, 
                          retain_data = retain_df, 
                          tokenizer = tokenizer, 
                          max_length=256) 


if cfg.loss_type == 'title_gd':
    print('\n\ncreating the dataset for title gradient diff')
    title_df = pd.read_csv('title_df.csv')
    def make_template_format(df):
        df['question_forget'] = df['question_forget'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
        df['answer_forget'] = df['answer_forget'].apply(lambda x : x + tokenizer.eos_token)
        df['question_retain'] = df['question_retain'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
        df['answer_retain'] = df['answer_retain'].apply(lambda x : x + tokenizer.eos_token)
        return df

    title_df = make_template_format(title_df)
    print('\n\nTitle df shape is:',title_df.shape)
    print('\n\nForget question and answer\n',title_df['question_forget'][0], title_df['answer_forget'][0])
    print('\n\nRetain df question and answer\n',title_df['question_retain'][0], title_df['answer_retain'][0])
    dataset = DualTitleDataset(paired_df=title_df,
                          tokenizer = tokenizer, 
                          max_length=256)
    print('\n\nLength of tokenized dataset', len(dataset))
    

    trainer = GradDiffTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        data_collator = custom_gd_collator_forget,
    )


# ------- dataset and training args for the gradient ascent method
if cfg.loss_type == 'grad_ascent' :
    print('\n\ncreating the dataset for gradient ascent')
    dataset = SingleDataset(forget_data = forget,
                            tokenizer = tokenizer,
                            max_length = 256) 

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
#tokenizer.save_pretrained(cfg.save_dir)

if training_args.local_rank <= 0: 
    tokenizer.save_pretrained(cfg.save_dir)
    print(f"Rank {training_args.local_rank}: Tokenizer saved.")
else:
    tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')




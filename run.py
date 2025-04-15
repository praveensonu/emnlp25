# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
# to run the script, use the command: 
# 1. export CUDA_VISIBLE_DEVICES=4,5
# 2. deepspeed --num_gpus=2 run.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config
from peft import  LoraConfig, get_peft_model
from perplexity import Perplexity_QA_from_df
from data_module import  custom_data_collator_forget, custom_gd_collator_forget
from utils import create_dual_dataset, create_single_dataset, update_json_dict, find_all_linear_names
from forget_trainer import GATrainer, GradDiffTrainer
from eval_utils import compute_model_utility_retain, compute_forget_efficacy, compute_model_utility_test



cfg = Config()

# loading the paths

print('loading the paths to forget, retain and test set')
forget_path = cfg.forget_path
retain_path = cfg.retain_path
test_path = cfg.test_path


device = 'cuda'
print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)

print(f"\nLoading the Model {cfg.model_id}")
model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
                                             #device_map = device,
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
                                  512, 
                                  template_format=None
    )
    training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        learning_rate = cfg.lr,
        #per_device_train_batch_size= cfg.batch_size, # for grad diff I used smaller batch size
        num_train_epochs= cfg.num_epochs,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        #save_steps = cfg.forget.save_steps,
        evaluation_strategy= 'no',
        save_total_limit= 2,
        label_names = ['labels'],
        deepspeed= cfg.ds_path,
        bf16 = True,)

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
                                    max_length = 512,
                                    template_format = None) 
    

    training_args = TrainingArguments(
        output_dir = cfg.save_dir,
        learning_rate = cfg.lr,
        #per_device_train_batch_size= cfg.batch_size,
        num_train_epochs= cfg.num_epochs,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{cfg.save_dir}/logs',
        #save_steps = cfg.forget.save_steps,
        evaluation_strategy= 'no',
        save_total_limit= 2,
        label_names = ['labels'],
        deepspeed= cfg.ds_path,
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
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')


model.merge_and_unload()
batch_size = cfg.batch_size
max_length = 512

## perplexity on forget set after unlearning
## -> conditional perplexity calculation on answer given a question P(a|q)

print(f'Calculating perplexity on forget set after {cfg.loss_type} unlearning')

qa_perplexity_forget, _ = Perplexity_QA_from_df(
    model = model,
    df_path = forget_path,
    tokenizer =tokenizer,
    max_length =max_length,
    batch_size = batch_size,
    device = device
)
print('\nForget Perplexity',qa_perplexity_forget)


## perplexity on retain after unlearning
## -> conditional perplexity calculation on answer given a question P(a|q)

print(f'Calculating perplexity on retain set after {cfg.loss_type} unlearning')
qa_perplexity_retain, _ = Perplexity_QA_from_df(
    model = model,
    df_path = retain_path,
    tokenizer =tokenizer,
    max_length =max_length,
    batch_size = batch_size,
    device = device
)
print('\nRetain Perplexity', qa_perplexity_retain)

## perplexity on retain after unlearning
## -> conditional perplexity calculation on answer given a question P(a|q)

print(f'\nCalculating perplexity on test set after {cfg.loss_type} unlearning')
qa_perplexity_test, _ = Perplexity_QA_from_df(
    model = model,
    df_path = test_path,
    tokenizer =tokenizer,
    max_length =max_length,
    batch_size = batch_size,
    device = device
)
print('\nTest set Perplexity',qa_perplexity_test)



print('\ncalculating forget efficacy')

forget_df, all_forget_scores,forget_efficacy = compute_forget_efficacy(
    forget_path = forget_path,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
)

print('forget efficacy', forget_efficacy.item())

print('\ncalculating model utility on retain set')

retain_df, all_retain_scores, retain_model_utility = compute_model_utility_retain(
    retain_path = retain_path,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
)

print('model utility retain', retain_model_utility.item())
print('\ncalculating model utility on test set')

test_df, all_test_scores, test_model_utility = compute_model_utility_test(
    test_path = test_path,
    model = model,
    tokenizer = tokenizer,
    retriever_model= cfg.retriever_model,
    device = device,
)

print('model utility test', test_model_utility.item())
forget_df.to_csv(f'{cfg.exp_type}_forget_results.csv')
retain_df.to_csv(f'{cfg.exp_type}_retain_results.csv')
test_df.to_csv(f'{cfg.exp_type}_test_results.csv')


results = {cfg.loss_type: 
           {'forget_efficacy': forget_efficacy.item(),
           'model_utility_retain': retain_model_utility.item(),
           'model_utility_test': test_model_utility.item(),
           'forget_scores' : all_forget_scores.tolist(),
           'retain_scores': all_retain_scores.tolist(),
           'test_scores': all_test_scores.tolist(),
           'qa_perplexity_forget': qa_perplexity_forget.item(),
           'qa_perplexity_retain': qa_perplexity_retain.item(),
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
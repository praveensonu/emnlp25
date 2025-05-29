
# 1. export CUDA_VISIBLE_DEVICES=1,6
# 2. accelerate launch --num_processes 2 run_hpo.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config
from peft import  LoraConfig, get_peft_model
from data_module import DualDataset
from collators import custom_gd_collator_forget
from utils import find_all_linear_names
from forget_trainer import GradDiffTrainer
from accelerate import Accelerator
import pandas as pd
import optuna
from eval_utils import compute_model_utility_retain, compute_forget_efficacy, compute_model_utility_test


accelerator = Accelerator()

cfg = Config()

# loading the paths

print('loading the paths to forget, retain and test set')
forget = pd.read_csv(cfg.forget_path) #cfg.forget_path
retain = pd.read_csv(cfg.retain_path) #cfg.retain_path
test = pd.read_csv(cfg.test_path)

titles = ['Siegfried Lenz', 'Benedetto Varchi', 'Rudolf Christoph Eucken']
forget_df = forget.loc[forget['title'].isin(titles)]
test_df = test.loc[test['title'].isin(titles)]
retain_df = retain.loc[retain['title'].isin(titles)]

print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token

def model_init(trial=None): # trial is optional here if not tuning LoRA HPs
    print(f"Initializing model for trial...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        token=cfg.access_token,
    )
    lora_config = LoraConfig(
        r=cfg.LoRA_r,
        lora_alpha=cfg.LoRA_alpha,
        lora_dropout=cfg.LoRA_dropout,
        target_modules=find_all_linear_names(base_model),
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False
    return model



print('Creating the dataset for Gradient Difference')
# For GradDiff, you use DualDataset with forget and retain
train_dataset_for_gd = DualDataset(forget_df, retain_df, tokenizer, 256, template_format=None)
trainer_for_hpo_objective_gd = None


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True),
        "num_epochs": trial.suggest_int('num_train_epochs',1, 10 ),
    }
def gd_objective(trial: optuna.Trial):
    global trainer_for_hpo_objective_gd

    
    lr = trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True) 
    num_epochs = trial.suggest_int("num_train_epochs", 1, 10) 
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    #per_device_train_batch_size = trial.suggest_int('per_device_train_batch_size', 1,4)


    training_args = TrainingArguments(
        output_dir=f"{cfg.save_dir}/hpo_gd_trial_{trial.number}", 
        overwrite_output_dir=True,
        learning_rate=lr,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        weight_decay=weight_decay,
        logging_strategy="no",
        evaluation_strategy="no", # Manual evaluation at the end of the trial
        save_strategy="no", # No intermediate saves during HPO
        label_names=['labels'], # As per your original GradDiff setup
        bf16=True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps, # Use your config
        report_to="none",
        remove_unused_columns=False, # Often needed for custom datasets/collators
    )
    if torch.cuda.is_available():
        print(f"Trial {trial.number} - Before train: GPU mem allocated: {torch.cuda.memory_allocated(eval_device)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(eval_device)/1024**2:.2f} MB")
    # Initialize GradDiffTrainer for THIS trial
    trainer = GradDiffTrainer(
        model_init=model_init, # model_init will be called with this trial
        args=training_args,
        train_dataset=train_dataset_for_gd,
        tokenizer=tokenizer,
        data_collator=custom_gd_collator_forget,
        # If GradDiffTrainer takes retain_strength_lambda:
        # retain_lambda=retain_strength_lambda, # Pass it here
    )
    trainer_for_hpo_objective_gd = trainer

    print(f"\n--- Starting GD HPO Trial {trial.number} ---")
    trainer.train()
    print(f"--- GD HPO Trial {trial.number} Training Finished ---")
    
    if torch.cuda.is_available():
        print(f"Trial {trial.number} - After train: GPU mem allocated: {torch.cuda.memory_allocated(eval_device)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(eval_device)/1024**2:.2f} MB")
    
    print(f"--- Evaluating GD HPO Trial {trial.number} ---")
    current_model = trainer.model # Model after training in this trial
    try:
        eval_device = next(current_model.parameters()).device
    except StopIteration:
        print("Warning: couldnt infer the device from model params directly, falling back to accelerateor.device")
        eval_device = accelerator.device
    fe = compute_forget_efficacy(forget = forget_df,
                                    model = current_model,
                                    tokenizer = tokenizer,
                                    retriever_model= cfg.retriever_model,
                                    device = eval_device)
    mu_t = compute_model_utility_test(test = test_df,
                                      model = current_model, 
                                      tokenizer = tokenizer, 
                                      retriever_model= cfg.retriever_model,
                                      device = eval_device)

    # Objective for Gradient Difference:
    # Prioritize getting *some* MU-T, as FE is usually high.
    # If MU-T is very low (e.g., < 0.01, indicating collapse), heavily penalize.
    # Otherwise, try to maximize a combination.
    min_acceptable_fe = 0.39
    min_acceptable_mu_t = 0.59
    fe_penalty = 0
    if fe < min_acceptable_fe:
        fe_penalty = -100 * (min_acceptable_fe - fe)

    mu_t_penalty = 0
    if mu_t < min_acceptable_mu_t:
        mu_t_penalty = -200 * (min_acceptable_mu_t - mu_t)

    base_objective = (fe * 0.4) + (mu_t * 0.6)

    objective_value = base_objective + fe_penalty + mu_t_penalty

    if torch.cuda.is_available():
        print(f"Trial {trial.number} - After eval: GPU mem allocated: {torch.cuda.memory_allocated(eval_device)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(eval_device)/1024**2:.2f} MB")
    # Another way to structure the penalization: this is better
    # if fe < min_acceptable_fe or mu_t < min_acceptable_mu_t:
    #     # If any threshold is not met, assign a very low score,
    #     # potentially differentiating by how badly it missed.
    #     # This makes it a "harder" constraint.
    #     objective_value = -1000 + (fe - min_acceptable_fe) + (mu_t - min_acceptable_mu_t)
    # else:
    #     # Both thresholds met, now optimize the weighted sum
    #     objective_value = (fe * 0.4) + (mu_t * 0.6) # Or fe + mu_t
    del current_model
    del trainer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"GD Trial {trial.number}: LR={lr:.2e}, Epochs={num_epochs}, WD={weight_decay:.3f} -> FE={fe:.3f}, MU-T={mu_t:.3f}, Objective={objective_value:.3f}")
    return objective_value


# --- Dummy Trainer for `hyperparameter_search` structure ---
placeholder_training_args_gd = TrainingArguments(
    output_dir=f"{cfg.save_dir}/hpo_gd_placeholder",
    report_to="none",
    per_device_train_batch_size=1, num_train_epochs=1, eval_strategy="no",
)

trainer_for_hpo_search_gd = GradDiffTrainer( # Use GradDiffTrainer here
    model=None, # Will be set by model_init in objective
    model_init=model_init,
    args=placeholder_training_args_gd,
    train_dataset=train_dataset_for_gd, # Provide a train_dataset
    tokenizer=tokenizer,
    data_collator=custom_gd_collator_forget
)

# --- 3. RUN HYPERPARAMETER SEARCH for Gradient Difference ---
print("\n--- Starting Gradient Difference Hyperparameter Search ---")
best_trial_gd = trainer_for_hpo_search_gd.hyperparameter_search(
    hp_space=None,
    compute_objective=gd_objective,
    n_trials= 20, # e.g., 20-50 trials for GD cfg.hpo_n_trials
    direction="maximize",
    backend="optuna",
)

print("\n--- Gradient Difference Hyperparameter Search Finished ---")
print("Best GD trial results:")
print(best_trial_gd)


# --- Train final GD model with best hyperparameters ---
print("\n--- Training Final Gradient Difference Model with Best Hyperparameters ---")
best_lr_gd = best_trial_gd.hyperparameters["learning_rate"]
best_epochs_gd = best_trial_gd.hyperparameters["num_train_epochs"]
best_weight_decay_gd = best_trial_gd.hyperparameters["weight_decay"]
# best_retain_lambda_gd = best_trial_gd.hyperparameters.get("retain_lambda") # If tuned

final_gd_training_args = TrainingArguments(
    output_dir=cfg.save_dir_best_model_gd, 
    overwrite_output_dir=True,
    learning_rate=best_lr_gd,
    per_device_train_batch_size=cfg.batch_size,
    num_train_epochs=best_epochs_gd,
    weight_decay=best_weight_decay_gd,
    logging_dir=f'{cfg.save_dir_best_model_gd}/logs',
    evaluation_strategy="no", 
    label_names=['labels'],
    bf16=True,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    report_to="wandb", 
    remove_unused_columns=False,

)

final_gd_model = model_init(None) 

final_gd_trainer = GradDiffTrainer(
    model=final_gd_model,
    args=final_gd_training_args,
    train_dataset=train_dataset_for_gd,
    tokenizer=tokenizer,
    data_collator=custom_gd_collator_forget,
    # retain_lambda=best_retain_lambda_gd, # If tuned
)

final_gd_trainer.train()
print("\n--- Final Gradient Difference Model Training Finished ---")

print("Evaluating final best GD model:")
final_gd_model_to_eval = final_gd_trainer.model
eval_device = accelerator.device
final_fe_gd = compute_forget_efficacy(forget = forget_df,
                                model = final_gd_model_to_eval,
                                tokenizer = tokenizer,
                                retriever_model= cfg.retriever_model,
                                device = eval_device)
final_mu_t_gd = compute_model_utility_test(test = test_df,
                                    model = final_gd_model_to_eval, 
                                    tokenizer = tokenizer, 
                                    retriever_model= cfg.retriever_model,
                                    device = eval_device)

print(f"Final Best GD Model: FE={final_fe_gd:.4f}, MU-T={final_mu_t_gd:.4f}")


accelerator.wait_for_everyone()
final_gd_trainer.save_model(cfg.save_dir_best_model_gd)
if accelerator.is_main_process:
    tokenizer.save_pretrained(cfg.save_dir_best_model_gd)
    print(f"Best GD model and tokenizer saved at {cfg.save_dir_best_model_gd}")




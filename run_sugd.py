# run_sugd.py

import torch
import pandas as pd
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig # Optional: for quantization
)
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, DatasetDict
from peft import LoraConfig, get_peft_model
from config import Config
from tqdm import tqdm
from sugd_runner import SequentialUnlearningRunner, convert_raw_data_to_model_format
from utils import find_all_linear_names

# --- Configuration (Replace with your actual config loading) ---
# Instantiate config
cfg = Config()

forget_path = cfg.forget_path
retain_path = cfg.retain_path
test_path = cfg.test_path
# --- Helper Function for Data Preparation ---
def prepare_hf_dataset(csv_path: str,
                       tokenizer: AutoTokenizer,
                       max_length: int,
                       question_key: str,
                       answer_key: str) -> HFDataset:
    """Loads data from CSV, tokenizes it, and returns a Hugging Face Dataset."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[question_key, answer_key]) # Drop rows with missing Q/A
    print(f"Loaded {len(df)} rows.")

    tokenized_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    print("Tokenizing data...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        processed = convert_raw_data_to_model_format(
            tokenizer=tokenizer,
            max_length=max_length,
            question=row[question_key],
            answer=row[answer_key],
            template_format=None # Use default chat template or simple concat
        )
        tokenized_data["input_ids"].append(processed["input_ids"])
        tokenized_data["attention_mask"].append(processed["attention_mask"])
        tokenized_data["labels"].append(processed["labels"])

    print("Creating Hugging Face Dataset...")
    hf_dataset = HFDataset.from_dict(tokenized_data)
    print("Dataset created.")
    return hf_dataset

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Tokenizer
    print(f"Loading tokenizer: {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    # Set padding token if necessary
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
    # Recommended padding side for Causal LM
    tokenizer.padding_side = "left" # Often better for Causal LMs

    # 3. Load Model (with optional quantization)
    print(f"Loading model: {cfg.model_id}")
    quantization_config = None
    if cfg.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization (QLoRA).")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=quantization_config,
        # device_map="auto", # Use if loading large models across GPUs without DeepSpeed
        torch_dtype=torch.bfloat16 if cfg.use_bf16 and not cfg.use_4bit else torch.float16, # Set dtype
        trust_remote_code=True # If needed for specific models
    )

    # Resize token embeddings if pad token was added (important!)
    if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.vocab_size != len(tokenizer):
         print(f"Resizing token embeddings from {len(tokenizer)} to match potentially added pad token.")
         model.resize_token_embeddings(len(tokenizer))


    # Set gradient checkpointing if enabled
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")


    # 4. Apply LoRA (Optional)
    if cfg.use_lora:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=cfg.LoRA_r,
            lora_alpha=cfg.LoRA_alpha,
            target_modules= find_all_linear_names(model),
            lora_dropout=cfg.LoRA_dropout,
            bias="none",
            task_type= 'CAUSAL_LM'
        )
        model = get_peft_model(model, lora_config)
        print("LoRA applied. Trainable parameters:")
        model.print_trainable_parameters()
    elif not cfg.use_lora:
         print("LoRA not used. Fine-tuning full model (or subset if layer freezing is used).")


    # 5. Load and Prepare Data
    forget_hf_dataset = prepare_hf_dataset(
        forget_path, tokenizer, 512, 'question', 'answer'
    )
    retain_hf_dataset = prepare_hf_dataset(
        retain_path, tokenizer, 512, 'question', 'answer'
    )

    # Basic check if datasets are loaded
    if not forget_hf_dataset or not retain_hf_dataset:
         raise ValueError("Failed to load or process one or both datasets.")
    print(f"Forget dataset size: {len(forget_hf_dataset)}")
    print(f"Retain dataset size: {len(retain_hf_dataset)}")

    # 6. Define Training Arguments
    N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
    PER_DEVICE_BS = 1 # As per paper's strategy for fine-grained control
    TARGET_EFFECTIVE_BATCH_SIZE = 1 + cfg.forget_retain_ratio # e.g., 1 + 7 = 8

    gradient_accumulation_steps = max(1, TARGET_EFFECTIVE_BATCH_SIZE // (PER_DEVICE_BS * N_GPUS))

    print(f"Number of GPUs: {N_GPUS}")
    print(f"Per Device BS: {PER_DEVICE_BS}")
    print(f"Target Effective BS (1 Forget + {cfg.forget_retain_ratio} Retain): {TARGET_EFFECTIVE_BATCH_SIZE}")
    print(f"Calculated Gradient Accumulation Steps: {gradient_accumulation_steps}")

    training_args = TrainingArguments(
        output_dir=cfg.save_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs_per_chunk, # Epochs PER CHUNK
        weight_decay=cfg.weight_decay,
        logging_dir=f'{cfg.save_dir}/main_logs', # Main log dir
        logging_strategy="steps",
        logging_steps=10, # Log frequently within chunks
        save_strategy="no", # Runner handles final save
        # save_total_limit=1, # Only if saving per-chunk checkpoints
        label_names=['labels'],
        # deepspeed=cfg.ds_path, # Add if using DeepSpeed config file
        bf16=cfg.use_bf16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        fp16=not cfg.use_bf16 if torch.cuda.is_available() else False,
        gradient_checkpointing=cfg.gradient_checkpointing, # Use configured value
        optim=cfg.optim,
        # ddp_find_unused_parameters=False, # Set if needed for DDP and LoRA
        report_to="none", # Disable wandb/etc. unless configured
        seed=42 # Set a seed for reproducibility
    )

    # 7. Instantiate and Run the Runner
    runner = SequentialUnlearningRunner(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        forget_dataset=forget_hf_dataset,
        retain_dataset=retain_hf_dataset,
        chunk_size=cfg.chunk_size,
        forget_retain_ratio=cfg.forget_retain_ratio,
        # compute_metrics=your_compute_metrics_func, # Optional
        # preprocess_logits_for_metrics=your_preprocess_func # Optional
    )

    # 8. Train
    runner.train()

    # 9. Save Final Model and Summary
    runner.save_model() # Saves to cfg.output_dir / "final_model"
    runner.save_summary() # Saves to cfg.output_dir / "training_summary.json"

    print("SUGD training script finished.")
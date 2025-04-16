# sugd_runner.py

import datasets
import torch
import inspect
import json
import math
import pandas as pd

from torch.utils.data import Dataset, SequentialSampler, DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    AutoModelForCausalLM 
)
from datasets import Dataset as HFDataset 
from datasets import DatasetDict, concatenate_datasets
from typing import Optional, Dict, List, Tuple, Any
from itertools import cycle
from tqdm.auto import tqdm

# --- Helper function for Tokenization ---

def convert_raw_data_to_model_format(tokenizer: PreTrainedTokenizer,
                                     max_length: int,
                                     question: str, 
                                     answer: str, 
                                     template_format: Optional[str] = None) -> Dict[str, List[int]]:
    """
    Tokenizes question/answer pair into sequence-to-sequence/SFT format.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        max_length (int): Maximum sequence length for padding/truncation.
        question (str): The input prompt or question.
        answer (str): The target completion or answer.
        template_format (str, optional): A custom format string like "Instruction: {instruction}\nAnswer:"
                                         If None, uses tokenizer.apply_chat_template if available,
                                         otherwise defaults to simple concatenation.

    Returns:
        Dict[str, List[int]]: Dictionary containing 'input_ids', 'attention_mask', and 'labels'.
    """
    # 1. Format the prompt part
    if template_format:
        prompt_text = template_format.format(instruction=question) # Adapt if keys differ
    elif hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
         # Use chat template if available
         messages = [{"role": "user", "content": question}]
         prompt_text = tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
    else:
        # Fallback to simple concatenation (adjust if needed)
        prompt_text = question + "\n" # Simple separator

    # 2. Tokenize prompt to find its length
    prompt_encoding = tokenizer(prompt_text, add_special_tokens=False) # Don't add special tokens yet
    prompt_len = len(prompt_encoding['input_ids'])

    # 3. Tokenize answer
    answer_encoding = tokenizer(answer, add_special_tokens=False)
    answer_len = len(answer_encoding['input_ids'])

    # 4. Combine, add special tokens (like BOS/EOS depending on model)
    # Some models expect BOS at the start
    needs_bos = tokenizer.bos_token_id is not None
    bos_token_list = [tokenizer.bos_token_id] if needs_bos else []
    eos_token_list = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    full_ids = prompt_encoding['input_ids'] + answer_encoding['input_ids'] + eos_token_list
    # Adjust prompt_len if BOS was added
    prompt_len_adjusted = prompt_len + (1 if needs_bos else 0)

    # 5. Create attention mask (attend to everything initially)
    full_attention_mask = [1] * len(full_ids)

    # 6. Create labels (mask prompt tokens)
    labels = [-100] * prompt_len + answer_encoding['input_ids'] + eos_token_list

    # 7. Pad or truncate
    current_len = len(full_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have a pad token or EOS token set.")

    if current_len < max_length:
        pad_len = max_length - current_len
        input_ids = full_ids + [pad_token_id] * pad_len
        attention_mask = full_attention_mask + [0] * pad_len 
    elif current_len > max_length:
        input_ids = full_ids[:max_length]
        attention_mask = full_attention_mask[:max_length]
        labels = labels[:max_length]
        # Ensure last token is EOS if truncated and EOS exists
        if eos_token_list and input_ids[-1] != tokenizer.eos_token_id:
             input_ids[-1] = tokenizer.eos_token_id
             # Ensure label for the new EOS is correct (might need adjustment based on exact logic)
             if labels[-1] != -100: # Only overwrite if it wasn't masked padding
                labels[-1] = tokenizer.eos_token_id
    else:
        input_ids = full_ids
        attention_mask = full_attention_mask
        labels = labels

    # Ensure length consistency (sanity check)
    assert len(input_ids) == max_length, f"Final input_ids length {len(input_ids)} != {max_length}"
    assert len(attention_mask) == max_length, f"Final attention_mask length {len(attention_mask)} != {max_length}"
    assert len(labels) == max_length, f"Final labels length {len(labels)} != {max_length}"


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# --- Core SUGD Classes ---

class AdvSupervisedDataset(Dataset):
    """
    Dataset for interleaved forget/retain samples for SUGD, adapted from paper's concept.
    This version explicitly interleaves based on the forget:retain ratio.
    Assumes input forget_chunk and retain_chunk are already Hugging Face Datasets.
    """
    def __init__(
        self,
        forget_chunk: HFDataset,
        retain_chunk: HFDataset,
        forget_retain_ratio: int, 
        positive_factor: float = 1.0,
    ):
        super(AdvSupervisedDataset, self).__init__()

        len_forget = len(forget_chunk)
        len_retain = len(retain_chunk)

        if len_retain < len_forget * forget_retain_ratio:
             print(f"Warning: Retain chunk size ({len_retain}) is less than required "
                   f"for the ratio ({len_forget * forget_retain_ratio}). Using all available retain samples.")

        self.data = [] 

        retain_idx_counter = 0
        for i in range(len_forget):
            forget_sample = forget_chunk[i]
            self.data.append({
                "input_ids": forget_sample["input_ids"],
                "labels": forget_sample["labels"],
                "attention_mask": forget_sample["attention_mask"],
                "factor": -1.0
            })

            num_retain_to_add = forget_retain_ratio
            for _ in range(num_retain_to_add):
                if retain_idx_counter < len_retain:
                    retain_sample = retain_chunk[retain_idx_counter]
                    self.data.append({
                        "input_ids": retain_sample["input_ids"],
                        "labels": retain_sample["labels"],
                        "attention_mask": retain_sample["attention_mask"],
                        "factor": positive_factor
                    })
                    retain_idx_counter += 1
                else:
                    break

        print(f"Created AdvSupervisedDataset: {len(forget_chunk)} forget, {retain_idx_counter} retain (ratio ~1:{forget_retain_ratio}). Total: {len(self)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return self.data[i]

    def summary(self) -> Dict[str, int]:
        factors = [item['factor'] for item in self.data]
        summary = {
            "retain_samples" : sum(1 for f in factors if f > 0),
            "forget_samples" : sum(1 for f in factors if f < 0),
            "total_samples" : len(self)
        }
        return summary


class AscentPlusDescentDataCollator(DataCollatorForSeq2Seq):
    """Collator that also handles the 'factor' column."""
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        if "factor" in features[0]:
            batch["factor"] = torch.tensor([f["factor"] for f in features], dtype=torch.float32)
        return batch


class AscentPlusDescentTrainer(Trainer):
    """Trainer that uses the 'factor' to adjust the loss."""
    def compute_loss(self, model, inputs, return_outputs=False):
        if "factor" not in inputs:
            print("Warning: 'factor' not found in inputs. Using standard loss.")
            return super().compute_loss(model, inputs, return_outputs)

        factors = inputs.pop("factor").to(self.args.device)
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = loss.view(shift_logits.size(0), -1)

            valid_counts = (shift_labels != -100).sum(dim=-1).float()
            valid_counts = torch.max(valid_counts, torch.tensor(1.0, device=valid_counts.device))

            per_sequence_loss = loss.sum(dim=-1) / valid_counts
        else:
             print("Warning: Labels not found in inputs during loss computation.")
             per_sequence_loss = torch.tensor(0.0, device=logits.device).repeat(logits.size(0))

        adjusted_loss = (per_sequence_loss * factors).mean()

        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        print("Using SequentialSampler for training.")
        # Ensure the dataset being sampled is the train_dataset attribute
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return SequentialSampler(self.train_dataset)


    def _set_signature_columns_if_needed(self):
         # Override to ensure 'factor' is *not* passed to model.forward if it's not an arg
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            model_signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            label_columns = list(set(["label", "label_ids"] + self.label_names))

            # Columns for the trainer: model args + label args
            self._signature_columns = model_signature_columns + label_columns

            # Add 'factor' if it's not already in the combined list (it shouldn't be in model forward)
            if 'factor' not in self._signature_columns:
                 self._signature_columns.append('factor')

            # Make sure the final list doesn't contain duplicates if 'factor' somehow was in model args
            self._signature_columns = list(set(self._signature_columns))
            print(f"Trainer signature columns set to: {self._signature_columns}")


# --- Orchestration Class ---

class SequentialUnlearningRunner:
    """
    Manages the sequential unlearning process with gradient difference.
    """
    def __init__(
            self,
            model: AutoModelForCausalLM, # More specific type hint
            tokenizer: PreTrainedTokenizer,
            training_args: TrainingArguments,
            forget_dataset: HFDataset,
            retain_dataset: HFDataset,
            chunk_size: int = 32,
            forget_retain_ratio: int = 7,
            compute_metrics=None,
            preprocess_logits_for_metrics=None
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.forget_dataset = forget_dataset # Assumes already HFDataset
        self.retain_dataset = retain_dataset # Assumes already HFDataset
        self.chunk_size = chunk_size
        self.forget_retain_ratio = forget_retain_ratio
        self.data_collator = AscentPlusDescentDataCollator(tokenizer=tokenizer, model=model)
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        self.retain_next_index = 0
        self.log_history = []
        self.total_runtime = 0.0
        self.total_flos = 0.0

        # Pre-convert retain dataset to list for faster cycling if it's large
        print("Converting retain dataset to list for efficient cyclic sampling...")
        self._retain_list = self.retain_dataset.to_list()
        self._total_retain_samples = len(self._retain_list)
        print(f"Retain list created with {self._total_retain_samples} samples.")
        if not self._retain_list:
             print("Warning: Retain dataset is empty.")


    def _get_cycled_retain_samples(self, required_size: int) -> HFDataset:
        """
        Cycle through retain samples sequentially, returning a Dataset.
        Uses the pre-converted list for efficiency.
        """
        if self._total_retain_samples == 0:
            print("Warning: Cannot sample from empty retain dataset.")
             # Return an empty dataset with the correct schema
            return HFDataset.from_dict({col: [] for col in self.retain_dataset.column_names})

        cycled_samples_list = []
        current_idx = self.retain_next_index

        for _ in range(required_size):
            cycled_samples_list.append(self._retain_list[current_idx])
            current_idx = (current_idx + 1) % self._total_retain_samples

        next_index = current_idx
        self.retain_next_index = next_index

        if not cycled_samples_list:
             cycled_samples_dict = {col: [] for col in self.retain_dataset.column_names}
        else:
             # Ensure all keys from the original schema are present, even if empty
             all_keys = self.retain_dataset.column_names
             cycled_samples_dict = {key: [row.get(key) for row in cycled_samples_list] for key in all_keys}
             # Handle potential None values if keys were missing in some rows (shouldn't happen with .to_list())
             # This part might need adjustment based on actual data structure after .to_list()

        cycled_retain = HFDataset.from_dict(cycled_samples_dict)
        # Preserve features (schema) from original retain dataset
        cycled_retain = cycled_retain.cast(self.retain_dataset.features)

        return cycled_retain


    def train(self):
        """
        Run the sequential unlearning process.
        """
        num_forget_samples = len(self.forget_dataset)
        if num_forget_samples == 0:
            print("Forget dataset is empty. Skipping training.")
            return

        # Ensure chunk size isn't larger than the dataset
        actual_chunk_size = min(self.chunk_size, num_forget_samples)
        if actual_chunk_size <= 0:
             print(f"Invalid chunk size {self.chunk_size} or empty forget dataset. Exiting.")
             return

        n_chunks = math.ceil(num_forget_samples / actual_chunk_size) # Use ceil for last partial chunk


        print(f"Starting sequential unlearning...")
        print(f"Total forget samples: {num_forget_samples}")
        print(f"Chunk size: {actual_chunk_size}")
        print(f"Number of chunks: {n_chunks}")
        print(f"Forget:Retain ratio per forget sample: 1:{self.forget_retain_ratio}")

        chunk_iterator = tqdm(range(n_chunks), desc="Processing Chunks")

        for i in chunk_iterator:
            start_idx = i * actual_chunk_size
            end_idx = min((i + 1) * actual_chunk_size, num_forget_samples)
            current_forget_chunk_size = end_idx - start_idx

            if current_forget_chunk_size <= 0: continue # Should not happen with ceil

            partial_forget_set = self.forget_dataset.select(range(start_idx, end_idx))

            retain_chunk_size_needed = current_forget_chunk_size * self.forget_retain_ratio
            partial_retain_set = self._get_cycled_retain_samples(retain_chunk_size_needed)

            chunk_iterator.set_description(f"Chunk {i+1}/{n_chunks}")
            print(f"\n--- Chunk {i+1} ---")
            print(f"Forget indices: {start_idx}-{end_idx-1} (Size: {current_forget_chunk_size})")
            print(f"Required retain samples: {retain_chunk_size_needed}, Actual sampled: {len(partial_retain_set)}")
            print(f"Retain sampling ended at index: {self.retain_next_index}")

            chunk_train_dataset = AdvSupervisedDataset(
                forget_chunk=partial_forget_set,
                retain_chunk=partial_retain_set,
                forget_retain_ratio=self.forget_retain_ratio,
                positive_factor=1.0
            )
            print(f"Chunk Train Dataset Summary: {chunk_train_dataset.summary()}")

            # Skip training if the combined dataset for the chunk is empty
            if len(chunk_train_dataset) == 0:
                print(f"Skipping training for chunk {i+1} as the combined dataset is empty.")
                continue

            # Modify output_dir per chunk if saving checkpoints per chunk
            chunk_output_dir = f"{self.training_args.output_dir}/chunk_{i+1}"
            # It's safer to create a copy if modifying TrainingArguments per chunk
            from dataclasses import replace
            chunk_training_args = replace(self.training_args, output_dir=chunk_output_dir)
            # Ensure log dir is also specific if needed
            chunk_training_args.logging_dir = f"{chunk_output_dir}/logs"

            trainer = AscentPlusDescentTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=chunk_training_args,
                train_dataset=chunk_train_dataset,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
            )

            print(f"Training on chunk {i+1}...")
            train_result = trainer.train()

            # Accumulate history and stats
            # Filter out the final summary dict
            chunk_log = [log for log in trainer.state.log_history if 'train_runtime' not in log]
            self.log_history.extend(chunk_log)

            # Get runtime/flops from train_result metrics
            report = train_result.metrics
            chunk_runtime = report.get("train_runtime", 0)
            chunk_flos = report.get("total_flos", 0)
            self.total_runtime += chunk_runtime
            self.total_flos += chunk_flos

            print(f"Chunk {i+1} trained. Runtime: {chunk_runtime:.2f}s, FLOS: {chunk_flos}")

        print("\nSequential unlearning finished.")


    def save_model(self, final_save_path=None):
        """Save the final model state."""
        # Use the original output_dir from training_args unless overridden
        output_dir = final_save_path if final_save_path else f"{self.training_args.output_dir}/final_model"
        print(f"Saving final trained model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model and tokenizer saved.")

    def save_summary(self, summary_path=None):
        """Save the accumulated training log and stats."""
        output_file = summary_path if summary_path else f"{self.training_args.output_dir}/training_summary.json"
        summary_data = {
            "log_history": self.log_history,
            "total_runtime": self.total_runtime,
            "total_flos": self.total_flos,
            "final_retain_index": self.retain_next_index,
            "chunk_size": self.chunk_size,
            "forget_retain_ratio": self.forget_retain_ratio,
            "training_args": self.training_args.to_dict() # Include args used
        }
        print(f"Saving training summary to {output_file}")
        # Ensure output directory exists
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            # Handle non-serializable items in training_args if necessary
            try:
                json.dump(summary_data, f, indent=4)
            except TypeError as e:
                 print(f"Warning: Could not serialize all training args: {e}. Saving partial summary.")
                 # Try saving without args or selectively removing problematic keys
                 del summary_data["training_args"]
                 json.dump(summary_data, f, indent=4)
        return summary_data
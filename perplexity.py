import torch
import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch import Tensor
from typing import Tuple, Generator, List, Dict
from tqdm.auto import tqdm # Optional: for progress bar

# --- Revised Data Preparation ---
# (Combines formatting and tokenization more efficiently)

def prepare_inputs_for_perplexity(
    questions: List[str],
    answers: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    device: str = 'cpu' # Allow specifying device here
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Prepares input_ids, attention_mask, and labels for QA perplexity calculation.
    Labels are masked for the question part. Uses tokenizer's batch encoding and padding.

    Args:
        questions (List[str]): List of questions.
        answers (List[str]): List of corresponding answers.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        max_length (int): Maximum sequence length for truncation and padding.
        device (str): Device to place tensors on ('cpu' or 'cuda').

    Returns:
        Tuple[Tensor, Tensor, Tensor]: input_ids, attention_mask, labels tensors.
    """
    if not questions or not answers or len(questions) != len(answers):
        raise ValueError("Questions and answers lists must be non-empty and have the same length.")

    # 1. Format prompts using chat template (batch processing friendly)
    prompts_formatted = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        # Keep add_generation_prompt=True if your model expects it (like Llama, Mistral instruct)
        # Set to False if the model expects only raw user query + eos
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts_formatted.append(formatted)

    # 2. Tokenize prompts *only* to get their lengths accurately AFTER chat templating
    #    Use temporary tokenization without padding/truncation here.
    prompt_tokenized = tokenizer(prompts_formatted, add_special_tokens=False) # Avoid double special tokens
    prompt_lengths = [len(ids) for ids in prompt_tokenized['input_ids']]

    # 3. Create full texts (prompt + answer)
    full_texts = [p + a + tokenizer.eos_token_id for p, a in zip(prompts_formatted, answers)]
    # Note: Added eos_token_id to the end of the answer. Loss will be computed on it too.

    # 4. Tokenize full texts with padding and truncation
    inputs = tokenizer(
        full_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length", # Let tokenizer handle padding
        return_tensors="pt",
        add_special_tokens=True # Add BOS if tokenizer configured to do so
    )

    # 5. Create labels: start with input_ids, then mask
    labels = inputs.input_ids.clone()

    # Mask padding tokens
    labels[labels == tokenizer.pad_token_id] = -100

    # Mask the prompt tokens (including special tokens added by chat template)
    # Need to account for potential BOS token added by the main tokenizer call
    bos_len = 1 if tokenizer.bos_token_id and inputs.input_ids[0, 0] == tokenizer.bos_token_id else 0

    for i in range(len(labels)):
        # Calculate actual prompt length in the final tokenized sequence
        # This is tricky because the main tokenization might add BOS differently than the prompt-only one
        # A safer way: tokenize prompt and answer *separately* then combine.
        # Let's stick to the original idea but be careful:
        # The prompt length in the *final* sequence includes BOS + template tokens
        # We measured prompt_lengths *without* BOS/EOS from apply_chat_template.

        # Re-tokenize the formatted prompt WITH special tokens to get accurate length
        # This is slightly less efficient but more robust to tokenizer behavior
        prompt_with_special_tokens = tokenizer(prompts_formatted[i], add_special_tokens=True)
        actual_prompt_len_in_final = len(prompt_with_special_tokens['input_ids'])
        
        # If the prompt was truncated during full text tokenization, adjust
        mask_len = min(actual_prompt_len_in_final, max_length)

        labels[i, :mask_len] = -100

        # Sanity check: Ensure we don't mask everything if prompt+answer < max_length
        # and prompt itself >= max_length
        if (labels[i] == -100).all():
             print(f"Warning: All labels masked for index {i}. Prompt length might exceed max_length or tokenizer settings mismatch.")
             # Optionally, unmask the last valid token if needed, depends on desired behavior
             # if mask_len > 0: labels[i, mask_len - 1] = inputs.input_ids[i, mask_len-1]


    return (
        inputs.input_ids.to(device),
        inputs.attention_mask.to(device),
        labels.to(device)
    )


# --- Revised Perplexity Calculation ---
# (More accurate loss averaging)

def calculate_perplexity_qa(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    batch_size: int,
    device: str
) -> Tuple[float, float]:
    """
    Calculates QA Perplexity using pre-processed inputs.

    Args:
        model (PreTrainedModel): The model to evaluate.
        input_ids (Tensor): Input IDs.
        attention_mask (Tensor): Attention mask.
        labels (Tensor): Labels tensor (with prompt masked).
        batch_size (int): Batch size for evaluation.
        device (str): Device model is on.

    Returns:
        Tuple[float, float]: Perplexity score, Average loss per token.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0 # Count of actual tokens loss is computed on

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Perplexity"): # Add progress bar
            b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]

            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )

            loss = outputs.loss # This is usually the MEAN loss over tokens in the batch
            num_active_tokens = (b_labels != -100).sum()

            # To get total loss, multiply mean loss by number of tokens it was averaged over
            if num_active_tokens > 0:
                total_loss += loss.item() * num_active_tokens
                total_tokens += num_active_tokens.item()
            # Handle case where a batch might have zero active tokens (e.g., only padding/masked)
            elif loss.item() == 0.0: # Or check if num_active_tokens == 0
                 pass # No contribution to loss or token count
            # else: # Potentially handle unexpected loss values if needed
            #     print(f"Warning: Loss is {loss.item()} but num_active_tokens is {num_active_tokens}")


    if total_tokens == 0:
        print("Warning: No valid tokens found to calculate perplexity.")
        return float('inf'), float('inf')

    average_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(average_loss)).item() # Use .item() to get float

    print(f"Total Loss: {total_loss:.4f}, Total Tokens: {total_tokens}")
    print(f"Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.4f}")

    return perplexity, average_loss

# --- Simplified Wrapper ---

def Perplexity_QA_from_df(
    model: PreTrainedModel,
    df_path: str, # Take path instead of df
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int,
    device: str
) -> Tuple[float, float]:
    """
    Wrapper function to compute QA perplexity from a CSV file.

    Args:
        model (PreTrainedModel): Model to evaluate.
        df_path (str): Path to the CSV file with 'question' and 'answer' columns.
        tokenizer (PreTrainedTokenizer): Tokenizer.
        max_length (int): Maximum sequence length.
        batch_size (int): Evaluation batch size.
        device (str): Device ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: Perplexity score, Average loss per token.
    """
    print(f"Loading data from: {df_path}")
    df = pd.read_csv(df_path)
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("DataFrame must contain 'question' and 'answer' columns.")

    # Ensure answers are strings (sometimes read as float/int)
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)


    print("Preparing inputs...")
    input_ids, attention_mask, labels = prepare_inputs_for_perplexity(
        df['question'].tolist(),
        df['answer'].tolist(),
        tokenizer,
        max_length,
        device # Pass device here if you want tensors created directly on GPU
    )

    print(f"Calculating perplexity with batch size {batch_size}...")
    perplexity, avg_loss = calculate_perplexity_qa(
        model,
        input_ids,
        attention_mask,
        labels,
        batch_size,
        device
    )

    return perplexity, avg_loss
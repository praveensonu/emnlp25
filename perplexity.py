import torch
from torch import Tensor
from typing import Generator, Tuple, Union
import pandas as pd
from transformers import PreTrainedTokenizer, PreTrainedModel
#from template import get_llama3_chat_template

def prepare_inputs_from_dataframe(
        df: pd.DataFrame, 
        max_length: int, 
        template: str, 
        tokenizer: PreTrainedTokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    """
    Prepare input tensors (input_ids, attention_mask, labels) for model training or evaluation.

    Args:
        df (pd.DataFrame): DataFrame containing the columns 'question' and 'answer'.
        max_length (int): Maximum sequence length. This includes max_new_tokens + token length of question.
        template (str): Template string for the input. The template should contain a placeholder {instruction} for the question.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing input_ids, labels, and attention_mask.
    
    """

    inputs, labels, attention_masks = [], [], []


    for _,row in df.iterrows():
        question, answer = row['question'], row['answer']
        formatted_input = template.format(instruction=question) 
        full_text = formatted_input + answer

        # get the number of tokens in the question part. We use this later to create the labels.
        num_question_len = len(tokenizer.encode(formatted_input, add_special_tokens=True))

        # tokenization
        encoded = tokenizer(
            full_text, 
            max_length=max_length, 
            truncation=True, 
            add_special_tokens=True)

        # taking pad length to pad the input_ids and attention_mask
        pad_length = max_length - len(encoded['input_ids'])

        # pad input ids and attention mask
        pad_input_ids = encoded['input_ids'] + [128009] + [128004] * (pad_length -1)
        pad_attention_mask = encoded['attention_mask'] + [0]*pad_length

        # get labels
        if len(encoded['input_ids']) == max_length:
            current_labels = encoded['input_ids']
        else:
            current_labels = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)

        # mask the questions based on num_question_len
        for i in range(num_question_len):
            current_labels[i] = -100
        
        # append the tensors
        inputs.append(torch.tensor(pad_input_ids))
        attention_masks.append(torch.tensor(pad_attention_mask))
        labels.append(torch.tensor(current_labels))

    return (
        torch.stack(inputs), 
        torch.stack(labels), 
        torch.stack(attention_masks),)


def create_batches(
        input_ids: Tensor, 
        labels: Tensor, 
        attention_mask: Tensor, 
        batch_size: int) -> Generator[Tuple[Tensor, Tensor, Tensor], None, None]:
    
    """
    Splits the data into batches.

    Args:
        input_ids (Tensor): Input ids tensor.
        labels (Tensor): Labels tensor.
        attention_mask (Tensor): Attention mask tensor.
        batch_size (int): Batch size.
    
    Returns:
        Generator[Tuple[Tensor, Tensor, Tensor], None, None]: Generator of batches.
    
    """
    for i in range(0, len(input_ids), batch_size):
        yield(
            input_ids[i:i+batch_size],
            labels[i:i+batch_size],
            attention_mask[i:i+batch_size]
        )


def calculate_perplexity(
        batches: Generator[Tuple[Tensor, Tensor, Tensor], None, None],
        model: PreTrainedModel, 
        case: str, 
        chat_tokens: int) -> float:
    
    """
    Calculates Perplexity for a given dataset (batches) and model.

    Args:
        batches (Generator[Tuple[Tensor, Tensor, Tensor], None, None]): Generator of batches.
        model (PreTrainedModel): Model to calculate perplexity.
        case (str): Case to calculate perplexity for.
        chat_tokens (int): Number of bos and eos tokens in the chat_template to ignore to calculate perplexity.
    
    Returns:
        float: Overall Perplexity.
    """


    total_loss = 0.0
    num_batches = 0
    print(f'calculating perplexity for {case}! Please change this if this is not the case')
    for input_ids_batch, labels_batch, attention_mask_batch in batches:
        if case == "next_token":

            # create labels for next token prediction (start from the chat_tokens+1 token)
            labels_batch = input_ids_batch.clone()
            labels_batch[:,:chat_tokens] = -100 # ignoring the chat_tokens 
        else:
            labels_batch = labels_batch
        
        with torch.no_grad():
            outputs = model(
                input_ids = input_ids_batch, 
                attention_mask = attention_mask_batch, 
                labels = labels_batch)
            total_loss += outputs.loss.item()
        num_batches += 1
    
    # if there is a division by zero error
    if num_batches == 0:
        raise ValueError("No batches found, run the create_batches function again. Generators are one-time use only.")

    
    average_loss = total_loss / num_batches
    print(f"Average loss for {num_batches} batches: {average_loss}")

    # calculate perplexity
    overall_perplexity = torch.exp(torch.tensor(average_loss))
    return overall_perplexity, average_loss, num_batches


def Perplexity(
        model: PreTrainedModel, 
        df: str, 
        tokenizer: PreTrainedTokenizer, 
        max_length: int, 
        template: str, 
        batch_size: int, # 14 for llama 3 chat template
        chat_tokens: int, 
        case: str) -> float:
    
    """
    Wrapper function to compute perplexity for a given model from a DtaFrame.

    Args:
        model (PreTrainedModel): Model to calculate perplexity.
        df (pd.DataFrame): DataFrame containing the columns 'question' and 'answer'.
        tokenizer (PreTrainedTokenizer): Tokenizer to tokenize the input.
        max_length (int): Maximum sequence length. This includes max_new_tokens + token length of question.
        template (str): Template to format the input.
        batch_size (int): Batch size.
        chat_tokens (int): Number of bos and eos tokens in the chat_template to ignore to calculate perplexity.
        case (str): Case to calculate perplexity for next_token or qa.
    
    Returns:
        float: Perplexity score.
    """
    df = pd.read_csv(df)
    input_ids, labels, attention_mask = prepare_inputs_from_dataframe(
        df, max_length, template, tokenizer)
    batches = create_batches(input_ids, labels, attention_mask, batch_size)
    return calculate_perplexity(batches, model, case, chat_tokens)


def predict(model, tokenizer, question):
    inputs = tokenizer(question, return_tensors='pt')
    outputs = model.generate(
        input_ids = inputs['input_ids'],
        attention_mask = inputs['attention_mask'],
        max_new_tokens = 100,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

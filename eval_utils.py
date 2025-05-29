import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm import tqdm
import pandas as pd
from torch import nn
from scipy.stats import ks_2samp, hmean
from typing import Generator, Tuple, Union
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer):
    question = str(question)
    answer = str(answer)
    full_text = question + answer
    num_question_tokens = len(tokenizer.tokenize(question, add_special_tokens=False))
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)



def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(gen_outputs, ground_truths)

    return rouge_scores['rouge1'].recall, rouge_scores['rougeL'].recall


def eval_cosine_similarity(gen_outputs, ground_truths, model_name, device):
    model = SentenceTransformer(model_name, device=device)
    with torch.no_grad():
        gen_embedding = model.encode(gen_outputs, show_progress_bar=False)
        gt_embedding = model.encode(ground_truths, show_progress_bar=False)
        cosine_sim = cosine_similarity([gen_embedding], [gt_embedding])[0][0]
        similarity = cosine_sim.item()
    return max(0, similarity)


def get_probs(outputs, labels):
    loss = get_batch_loss(outputs.logits, labels) # we use batch loss, which return sum loss of the sequence
    ## since get batch_loss give sum loss for each sequence in a batch
    T = (labels[0] != -100).sum().float()  # convert count to float
    loss_avg = loss / T  # average loss per answer token
    
    # Compute conditional probability p(y|x) = exp(-T * loss_avg)
    p_y_given_x = torch.exp(-T * loss_avg)
    
    return p_y_given_x, T.item(), loss_avg.item()


def calculate_cond_prob(prompt, answer, tokenizer, model, device):
    """
    Calculate conditional probability of the answer given the prompt.
    """
    # Format the prompt using the chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Create the full text with the formatted prompt and answer
    full_text = formatted_prompt + answer + tokenizer.eos_token
    
    # Tokenize the full text
    encoded = tokenizer(full_text, return_tensors='pt', add_special_tokens=True).to(device)
    full_input_ids = encoded['input_ids']
    
    # Tokenize just the prompt to get its length
    prompt_encoded = tokenizer(formatted_prompt, return_tensors='pt', add_special_tokens=True).to(device)
    prompt_len = prompt_encoded['input_ids'].size(1)
    
    # Create labels, masking the prompt tokens
    labels = full_input_ids.clone()
    labels[0, :prompt_len] = -100
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(full_input_ids)
    
    # Calculate probabilities
    p_y_given_x, T, loss_avg = get_probs(outputs, labels)
    
    return p_y_given_x



def generate_outputs(text, model, tokenizer, device):
    """
    Generate model outputs for the given text using the chat template.
    """
    # Format the input using the chat template
    messages = [{"role": "user", "content": text}]
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the formatted input
    inputs = tokenizer(formatted_input, return_tensors="pt", add_special_tokens=True).to(device)
    
    # Generate outputs
    outputs = model.generate(**inputs, max_new_tokens = 50)
    
    # Decode the outputs
    full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Extract the answer part
    # This assumes the assistant's response comes after the formatted prompt
    # The exact split might need adjustment based on the specific chat template used
    assistant_prefix = "assistant\n\n"
    if assistant_prefix in full_output:
        answer = full_output.split(assistant_prefix)[-1]
    else:
        # If the assistant prefix isn't found, try to extract the answer portion
        # based on the length of the formatted input
        prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if full_output.startswith(prompt_text):
            answer = full_output[len(prompt_text):].strip()
        else:
            answer = full_output  # Fallback to returning the full output
    
    return answer

# def generate_outputs(text, model, tokenizer, device):
#     inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
#     outputs = model.generate(**inputs)
#     full_output = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     answer = full_output.split("assistant\n\n")[-1]
#     return answer



def compute_forget_efficacy(forget, model, tokenizer, retriever_model, device):
    """
    Evaluate the forget efficacy by generating answers for each row in the provided DataFrame,
    computing evaluation metrics, and updating the DataFrame with generated answers.

    Parameters:
        forget (pd.DataFrame): DataFrame containing at least 'question' and 'answer' columns.
        model: The model used for generating answers.
        tokenizer: The tokenizer corresponding to the model.
        cfg: Configuration object that must include 'retriever_model' for cosine similarity evaluation.
        device: The device to run model computations on (e.g., "cpu" or "cuda").

    Returns:
        tuple: A tuple containing:
            - forget (pd.DataFrame): The updated DataFrame with a new 'gen_answer' column.
            - forget_efficacy (float): The computed forget efficacy score.
    """
    # Initialize the 'gen_answer' column and lists for evaluation metrics
    
    #forget = pd.read_csv(forget_path)
    forget['gen_answer'] = ''
    forget['probs'] = ''
    forget['rouge_l'] = ''
    forget['cos_sim'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []

    # Iterate through each row in the DataFrame
    for i, row in forget.iterrows():
        question = row['question']
        answer = row['answer']
        
        # Format prompt using a global template (assumed defined elsewhere)
        #prompt = template.format(instruction=question)
        prompt = question
        
        # Generate answer using the provided model and tokenizer
        gen_answer = generate_outputs(prompt, model, tokenizer, device=device)
        
        # Evaluate generated answer using ROUGE and cosine similarity metrics
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        prob = calculate_cond_prob(prompt, answer, tokenizer, model, device) #or gen_answer?

        # Update DataFrame and store metric scores
        forget.loc[i, 'gen_answer'] = gen_answer
        forget.loc[i, 'probs'] = prob.item()
        forget.loc[i, 'rouge_l'] = rougel
        forget.loc[i, 'cos_sim'] = cosine_sim

        probas.append(prob.item())
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)

    # Calculate the average scores for each metric and overall efficacy
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    
    forget_efficacy = 1.0 - np.mean(all_scores)
    print('forget_efficacy scores:',forget_efficacy)
    return forget, all_scores, forget_efficacy


def compute_model_utility_retain(retain, model, tokenizer, retriever_model, device):
    """
    Evaluate the forget efficacy by generating answers for each row in the provided DataFrame,
    computing evaluation metrics, and updating the DataFrame with generated answers.

    Parameters:
        forget (pd.DataFrame): DataFrame containing at least 'question' and 'answer' columns.
        model: The model used for generating answers.
        tokenizer: The tokenizer corresponding to the model.
        cfg: Configuration object that must include 'retriever_model' for cosine similarity evaluation.
        device: The device to run model computations on (e.g., "cpu" or "cuda").

    Returns:
        tuple: A tuple containing:
            - forget (pd.DataFrame): The updated DataFrame with a new 'gen_answer' column.
            - forget_efficacy (float): The computed forget efficacy score.
    """
    
    #retain = pd.read_csv(retain_path)
    retain['gen_answer'] = ''
    retain['probs'] = ''
    retain['rouge_l'] = ''
    retain['cos_sim'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []

    # Iterate through each row in the DataFrame
    for i, row in retain.iterrows():
        question = row['question']
        answer = row['answer']
        
        #prompt = template.format(instruction=question)
        prompt = question
        
        # Generate answer using the provided model and tokenizer
        gen_answer = generate_outputs(prompt, model, tokenizer, device=device)
        
        # Evaluate generated answer using ROUGE and cosine similarity metrics
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        prob = calculate_cond_prob(prompt, answer, tokenizer, model, device)  #or gen_answer?

        retain.loc[i, 'gen_answer'] = gen_answer
        retain.loc[i, 'probs'] = prob.item()
        retain.loc[i, 'rouge_l'] = rougel
        retain.loc[i, 'cos_sim'] = cosine_sim
        probas.append(prob.item())
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)

    # Calculate the average scores for each metric and overall efficacy
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    
    model_utility_retain = hmean(all_scores)
    #model_utility = np.mean(all_scores)
    print('Model Utility scores:',model_utility_retain)
    
    return retain, all_scores, model_utility_retain



def compute_model_utility_test(test, model, tokenizer, retriever_model, device):
    """
    Evaluate the forget efficacy by generating answers for each row in the provided DataFrame,
    computing evaluation metrics, and updating the DataFrame with generated answers.

    Parameters:
        forget (pd.DataFrame): DataFrame containing at least 'question' and 'answer' columns.
        model: The model used for generating answers.
        tokenizer: The tokenizer corresponding to the model.
        cfg: Configuration object that must include 'retriever_model' for cosine similarity evaluation.
        device: The device to run model computations on (e.g., "cpu" or "cuda").

    Returns:
        tuple: A tuple containing:
            - forget (pd.DataFrame): The updated DataFrame with a new 'gen_answer' column.
            - forget_efficacy (float): The computed forget efficacy score.
    """
    
    #test = pd.read_csv(test_path)
    test['gen_answer'] = ''
    test['probs'] = ''
    test['rouge_l'] = ''
    test['cos_sim'] = ''
    probas = []
    rouge1s = []
    rougels = []
    cos_sim = []

    # Iterate through each row in the DataFrame
    for i, row in test.iterrows():
        question = row['question']
        answer = row['answer']
        
        # Format prompt using a global template (assumed defined elsewhere)
        #prompt = template.format(instruction=question)
        prompt = question
        
        # Generate answer using the provided model and tokenizer
        gen_answer = generate_outputs(prompt, model, tokenizer, device=device)
        
        # Evaluate generated answer using ROUGE and cosine similarity metrics
        rouge1, rougel = eval_rouge_recall(gen_answer, answer)
        cosine_sim = eval_cosine_similarity(gen_answer, answer, retriever_model, device)
        prob = calculate_cond_prob(prompt, answer, tokenizer, model, device)  #or gen_answer?

        # Update DataFrame and store metric scores
        test.loc[i, 'gen_answer'] = gen_answer
        test.loc[i, 'probs'] = prob.item()
        test.loc[i, 'rouge_l'] = rougel
        test.loc[i, 'cos_sim'] = cosine_sim
        probas.append(prob.item())
        rouge1s.append(rouge1)
        rougels.append(rougel)
        cos_sim.append(cosine_sim)

    # Calculate the average scores for each metric and overall efficacy
    all_scores = np.array([np.mean(probas), np.mean(rougels), np.mean(cos_sim)])
    
    model_utility_test = hmean(all_scores)
    #model_utility = np.mean(all_scores)
    print('Model Utility scores:',model_utility_test)
    
    return test, all_scores, model_utility_test


def calculate_conditional_perplexity(question, model, tokenizer, device, max_new_tokens=100, sys_text=""):
    """
    Generate an answer to a question and calculate its conditional perplexity.
    
    This calculates the conditional perplexity of the answer given the question prompt.
    
    Args:
        question (str): The user question
        model: The language model
        tokenizer: The tokenizer for the model
        device: The device to run the model on
        max_new_tokens (int): Maximum tokens to generate
        sys_text (str): Optional system prompt
        
    Returns:
        float: The conditional perplexity value
        str: The generated answer
    """
    # Prepare the prompt with optional system text
    if sys_text:
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": question}
        ]
    else:
        messages = [{"role": "user", "content": question}]
    
    # Format input using chat template
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the formatted input
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].size(1)
    
    # Generate the answer
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True
        )
    
    # Get the full sequence (prompt + generated answer)
    full_sequence = generation_output.sequences[0]
    
    # Extract just the generated part (without the prompt)
    generated_ids = full_sequence[input_length:]
    
    # Decode the answer
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Calculate perplexity on only the generated part
    # Create new input that includes both the prompt and generated answer
    full_text = formatted_input + answer
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    
    # Create labels, masking prompt tokens (-100 is the ignore index)
    labels = full_inputs["input_ids"].clone()
    labels[:, :input_length] = -100
    
    # Forward pass with labels to compute loss
    with torch.no_grad():
        outputs = model(full_inputs["input_ids"], labels=labels)
    
    # Get the loss value (negative log likelihood)
    nll = outputs.loss.item()
    
    # Calculate perplexity: exp(average negative log likelihood)
    perplexity = torch.exp(torch.tensor(nll)).item()
    
    return perplexity, answer



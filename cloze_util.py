import pandas as pd
import re
from typing import List, Union
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu
from word2number import w2n


def process_cloze_questions(df, pipe, max_new_tokens=17):
    """
    Process a DataFrame of quiz questions, format each question with its options,
    and use a language model to generate answers.
    
    Args:
        dob_questions (pandas.DataFrame): DataFrame containing cloze questions
        pipe (function): The language model pipeline function to generate answers
        
    Returns:
        pandas.DataFrame: The input DataFrame with the 'ul_answers' column updated
    """
    df['cloze_answers'] = ''
    for i, row in df.iterrows():
        questions = row['question']
        instruct = f"Please fill in the blanks with the correct answers. Only provide the answer, do not write any explanation. Question:{questions}"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": instruct},
        ]
        
        outputs = pipe(messages, max_new_tokens=max_new_tokens)
        df.at[i, 'cloze_answers'] = outputs[0]['generated_text'][-1]['content']
    
    return df


def convert_number(text):
    """Attempt to convert number words to digits."""
    try:
        # word_to_num converts textual numbers to a numeric type
        return str(w2n.word_to_num(text))
    except Exception:
        return text
    

def normalize_text(text):
    """Lowercase, remove punctuation, extra spaces, and convert numbers."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert number words to digits if possible
    text = convert_number(text)
    return text


def exact_match(true_answer, pred_answer):
    """
    Compare normalized true and predicted answers.
    For multiple blanks, we split the answers by comma and compare each part.
    """
    true_parts = [normalize_text(part) for part in true_answer.split(',')]
    pred_parts = [normalize_text(part) for part in pred_answer.split(',')]
    # If number of blanks do not match, count as non-match.
    if len(true_parts) != len(pred_parts):
        return False
    return all(tp == pp for tp, pp in zip(true_parts, pred_parts))

def fuzzy_similarity(true_answer, pred_answer):
    """
    Compute a similarity score between normalized true and predicted answers
    using difflib's SequenceMatcher. For multi-blank answers, average the scores.
    """
    true_parts = [normalize_text(part) for part in true_answer.split(',')]
    pred_parts = [normalize_text(part) for part in pred_answer.split(',')]
    
    # If number of blanks differ, return 0 similarity.
    if len(true_parts) != len(pred_parts):
        return 0.0

    scores = [
        SequenceMatcher(None, tp, pp).ratio()
        for tp, pp in zip(true_parts, pred_parts)
    ]
    return sum(scores) / len(scores)

def compute_bleu(true_answer, pred_answer):
    """
    Compute a BLEU score comparing the tokenized true answer with the prediction.
    For multiple blanks, we average the BLEU scores.
    """
    true_parts = [normalize_text(part).split() for part in true_answer.split(',')]
    pred_parts = [normalize_text(part).split() for part in pred_answer.split(',')]
    
    if len(true_parts) != len(pred_parts):
        return 0.0
    
    bleu_scores = []
    for true_tokens, pred_tokens in zip(true_parts, pred_parts):
        # We use smoothing here because answers are short.
        try:
            bleu = sentence_bleu([true_tokens], pred_tokens)
        except Exception:
            bleu = 0.0
        bleu_scores.append(bleu)
    return sum(bleu_scores) / len(bleu_scores)


def compute_mean_scores(df):
    """
    Compute mean scores from a DataFrame/dictionary containing 'answer' and 'cloze_answers'.

    For each pair of true and predicted answers, it computes:
      - exact_match: a boolean value (True/False) indicating an exact match.
      - similarity_score: a score (e.g., between 0 and 1) using a fuzzy similarity function.
      - bleu_score: a BLEU score for the predicted answer.
    
    The function then returns:
      - exact_match_percentage: the percentage of exact matches (True values).
      - similarity_percentage: the percentage of similarity scores that are exactly 1.0.
      - average_bleu: the average BLEU score formatted in a simplified numeric form.

    Parameters:
        df (dict or DataFrame): Contains two keys/columns, "answer" and "cloze_answers".

    Returns:
        dict: A dictionary with keys 'exact_match_percentage', 'similarity_percentage', and 'average_bleu'.
    """
    exact_matches = []
    similarity_scores = []
    bleu_scores = []

    for true_ans, pred_ans in zip(df["answer"], df["cloze_answers"]):
        exact_matches.append(exact_match(true_ans, pred_ans))
        similarity_scores.append(fuzzy_similarity(true_ans, pred_ans))
        bleu_scores.append(compute_bleu(true_ans, pred_ans))

    # Compute percentage of exact matches (True values)
    exact_match_percentage = sum(exact_matches) / len(exact_matches) * 100

    # Compute percentage of similarity scores that are exactly 1.0
    similarity_percentage = sum(1 for score in similarity_scores if score == 1.0) / len(similarity_scores) * 100

    # Compute average BLEU score and simplify the number format
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    # Format the average to a fixed-point number with 6 decimal places
    average_bleu_simple = float(f"{average_bleu:.6f}")

    return {
        "exact_match_percentage": exact_match_percentage,
        "similarity_percentage": similarity_percentage,
        "average_bleu": average_bleu_simple
    }


### augment_cloze_questions i.e., appending answers to the cloze question

def augment_cloze_questions(questions: List[str], answers: List[Union[str, List[str]]]) -> pd.Series:
    """
    Augment cloze questions by inserting the original answer into the question.
    
    Args:
        questions (List[str]): List of cloze questions with blanks
        answers (List[Union[str, List[str]]]): Corresponding answers to fill the blanks
    
    Returns:
        pd.Series: Series of augmented questions
    """
    def parse_answers(answer: Union[str, List[str]]) -> List[str]:
        """
        Parse answers into a list, handling both single and multiple answers.
        
        Args:
            answer (Union[str, List[str]]): Answer or list of answers
        
        Returns:
            List[str]: List of answers
        """
        if isinstance(answer, str):
            # Split comma-separated answers if needed
            return [a.strip() for a in answer.split(',')]
        return answer

    def fill_cloze_question(question: str, question_answers: List[str]) -> str:
        """
        Replace the blanks in the cloze question with the answers.
        
        Args:
            question (str): Cloze question with blanks
            question_answers (List[str]): Answers to fill the blanks
        
        Returns:
            str: Question with the blanks filled
        """
        # Find all blanks in the question
        blanks = re.findall(r'___', question)
        
        # If answers are fewer than blanks, repeat the last answer
        if len(question_answers) < len(blanks):
            # Extend answers list by repeating the last answer
            extended_answers = question_answers + [question_answers[-1]] * (len(blanks) - len(question_answers))
        else:
            # Use only as many answers as there are blanks
            extended_answers = question_answers[:len(blanks)]
        
        # Replace blanks one by one
        filled_question = question
        for answer in extended_answers:
            filled_question = filled_question.replace('___', str(answer), 1)
        
        return filled_question
    
    # Validate input
    if len(questions) != len(answers):
        raise ValueError("Number of questions must match number of answers")
    
    # Create augmented questions
    augmented_questions = [
        fill_cloze_question(q, parse_answers(a)) 
        for q, a in zip(questions, answers)
    ]
    
    # Convert to Series
    return pd.Series(augmented_questions)




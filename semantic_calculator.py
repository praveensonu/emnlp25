import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm # Use standard tqdm for scripts
import logging
import argparse # Import argparse for command-line arguments

# --- Configuration (Input Files - can be overridden by args if needed) ---
DOMAIN_QA_PATH = "/home/praveen/theoden/emnlp_25/dataset/domain_qa.csv"
FORGET_DATA_URL = "hf://datasets/Shiyu-Lab/Wikipedia_Person_Unlearn/forget_20_1/train-00000-of-00001.parquet"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def compute_semantic_similarity(text1, text2, model, model_type='bge'):
    """Computes cosine similarity between two texts using the provided model."""
    try:
        # GTE uses standard encoding for documents, query prompt is for query<>doc
        # For doc<>doc comparison, standard encoding should be used for both.
        embedding1 = model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
        embedding2 = model.encode(text2, convert_to_tensor=True, show_progress_bar=False)

        # Compute cosine similarity
        similarity = util.cos_sim(embedding1, embedding2)
        # Return scalar float rounded
        return round(float(similarity[0][0]), 2)
    except Exception as e:
        logger.error(f"Error computing similarity for texts: '{str(text1)[:50]}...' and '{str(text2)[:50]}...'. Error: {e}", exc_info=True)
        # Handle potential errors, e.g., empty input strings
        if not isinstance(text1, str) or not isinstance(text2, str) or not text1 or not text2:
            logger.warning("Encountered non-string or empty text during similarity calculation.")
            return np.nan # Return NaN for errors or empty inputs
        return np.nan # Return NaN for other errors

# --- Main Script Logic ---
def main(args):
    """Main function to load data, preprocess, compute scores, and save results."""
    logger.info("--- Starting Semantic Score Calculation ---")
    logger.info(f"Selected model type: {args.model_type}")
    logger.info(f"Output file: {args.output_file}")
    # Set GPUs based on environment variable set before running script
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
         logger.info(f"Using CUDA devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
         logger.warning("CUDA_VISIBLE_DEVICES not set. Using default GPU assignment.")

    # --- Load Data ---
    try:
        logger.info(f"Loading domain QA data from: {DOMAIN_QA_PATH}")
        domain = pd.read_csv(DOMAIN_QA_PATH)
        logger.info(f"Loaded domain data with shape: {domain.shape}")

        logger.info(f"Loading forget data from Hugging Face: {FORGET_DATA_URL}")
        # Make sure 'datasets' and 'pyarrow' libraries are installed
        forget_20_1 = pd.read_parquet(FORGET_DATA_URL)
        logger.info(f"Loaded forget data with shape: {forget_20_1.shape}")
    except FileNotFoundError:
        logger.error(f"Error: Domain QA file not found at {DOMAIN_QA_PATH}")
        return # Exit if essential data is missing
    except ImportError as e:
        logger.error(f"ImportError loading data: {e}. Did you install 'pyarrow' and 'datasets'?")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return # Exit on other data loading errors

    # --- Merge DataFrames ---
    logger.info("Merging dataframes on 'title' column...")
    df = pd.merge(
        forget_20_1,
        domain,
        on="title",
        how="inner",
        suffixes=("_forget", "_domain") # Suffixes might change column names like 'wikipage'
    )
    logger.info(f"Merged dataframe shape before preprocessing: {df.shape}")
    if df.empty:
        logger.warning("Merged dataframe is empty. Check 'title' columns in input files.")
        logger.info("--- Script Finished (No data to process after merge) ---")
        return

    # --- Data Preprocessing ---
    logger.info("Starting data preprocessing...")
    # Define the columns expected after merge (adjust suffixes if needed)
    # Assuming merge resulted in 'wikipage_forget' and 'content_domain' based on suffixes
    # *Important*: Check the actual column names after the merge!
    wiki_col = 'wikipage' # Adjust if merge added suffix, e.g., 'wikipage_forget'
    content_col = 'content' # Adjust if merge added suffix, e.g., 'content_domain'
    required_cols = ['title', 'SimilarName', wiki_col, content_col]

    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns after merge: {missing_cols}. Available columns: {list(df.columns)}")
        logger.error("Please check the 'suffixes' in pd.merge and adjust 'wiki_col' and 'content_col' variables.")
        return

    logger.info(f"Dropping duplicates based on subset: {required_cols}")
    df = df.drop_duplicates(subset=required_cols, keep="first")
    logger.info(f"Shape after dropping duplicates: {df.shape}")

    logger.info(f"Selecting columns: {required_cols}")
    df = df[required_cols] # Select only the needed columns

    logger.info(f"Dropping rows with NaN in '{wiki_col}' or '{content_col}'")
    df = df.dropna(subset=[wiki_col, content_col])
    logger.info(f"Shape after dropping NaN values: {df.shape}")

    if df.empty:
        logger.warning("Dataframe is empty after preprocessing steps.")
        logger.info("--- Script Finished (No data left to process) ---")
        return

    # Convert text columns to string just in case
    df[wiki_col] = df[wiki_col].astype(str)
    df[content_col] = df[content_col].astype(str)
    logger.info(f"Ensured '{wiki_col}' and '{content_col}' are string type.")


    # --- Load Model ---
    model_name = None
    trust_remote_code = False
    max_seq_length = None # Default handled by model

    if args.model_type == 'bge':
        model_name = 'BAAI/bge-m3'
    elif args.model_type == 'gte':
        model_name = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
        trust_remote_code = True
        max_seq_length = 16700 # Specific to this GTE model example
    else:
        logger.error(f"Invalid model type specified: {args.model_type}. Choose 'bge' or 'gte'.")
        return

    logger.info(f"Loading Sentence Transformer model: {model_name}")
    logger.info(f"Trust remote code: {trust_remote_code}")
    if max_seq_length:
         logger.info(f"Setting max_seq_length: {max_seq_length}")

    try:
        # SentenceTransformer will automatically use CUDA if available and specified by CUDA_VISIBLE_DEVICES
        model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        if max_seq_length:
            model.max_seq_length = max_seq_length
        logger.info("Model loaded successfully.")
    except ImportError as e:
         logger.error(f"ImportError loading model '{model_name}': {e}.")
         logger.error("Ensure 'transformers' and potentially 'accelerate' are installed and up-to-date.")
         return
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}", exc_info=True)
        logger.error("Ensure the model name is correct and you have an internet connection.")
        return

    # --- Compute Semantic Scores ---
    logger.info("Computing semantic similarity scores...")
    # Register tqdm with pandas apply
    tqdm.pandas(desc=f"Calculating Scores ({args.model_type})")

    df["semantic_score_docs"] = df.progress_apply(
        lambda row: compute_semantic_similarity(
            row[wiki_col],
            row[content_col],
            model, # Pass the loaded model
            args.model_type # Pass model type (might be useful later, though not used in current func)
        ),
        axis=1
    )
    logger.info("Finished computing scores.")

    # --- Save Results ---
    logger.info(f"Saving results to: {args.output_file}")
    try:
        # Select columns for final output - include score and identifiers
        output_df = df[['title', 'SimilarName', wiki_col, content_col, 'semantic_score_docs']]
        output_df.to_csv(args.output_file, index=False)
        logger.info("Results saved successfully.")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}", exc_info=True)

    logger.info("--- Script Finished ---")

# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute semantic similarity between document pairs.")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['bge', 'gte'],
        default='bge', # Default to bge-m3
        help="Type of model to use ('bge' for BAAI/bge-m3, 'gte' for Alibaba-NLP/gte-Qwen2-1.5B-instruct)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='semantic_scores.csv',
        help="Path to save the output CSV file."
    )
    # Add arguments for input files if you want them configurable too
    # parser.add_argument("--domain_csv", type=str, default=DOMAIN_QA_PATH, help="Path to domain QA CSV file.")
    # parser.add_argument("--forget_parquet", type=str, default=FORGET_DATA_URL, help="Path or URL to forget data parquet file.")

    parsed_args = parser.parse_args()
    main(parsed_args)
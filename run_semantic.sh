#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="semantic_calculator.py"
OUTPUT_LOG="semantic_calculator.log"
OUTPUT_CSV="semantic_scores_gte.csv" # Define output CSV name

# --- CHOOSE MODEL ---
# Set this variable to 'bge' or 'gte'
MODEL_CHOICE="gte" # <--- EDIT THIS LINE TO CHOOSE THE MODEL ('bge' or 'gte')
# --------------------

# Optional: Specify the full path to your python executable if needed
# PYTHON_EXEC="/path/to/your/python_env/bin/python"
PYTHON_EXEC="python" # Assumes python is in PATH or you activate env below

# --- Set CUDA Devices (if not set globally) ---
# You can set it here if preferred over setting it in the python script,
# but setting in Python (as done now) is often cleaner.
export CUDA_VISIBLE_DEVICES="7"
# -----------------------------------------------

echo "--------------------------------------------------"
echo "Starting Semantic Calculation Script"
echo "Using Model Type: ${MODEL_CHOICE}"
echo "Output CSV: ${OUTPUT_CSV}"
echo "--------------------------------------------------"

# --- Optional: Activate Conda or Virtual Environment ---
# Uncomment and modify the following lines if you use an environment
# echo "Activating environment..."
# source /path/to/your/conda/etc/profile.d/conda.sh # Adjust path if necessary
# conda activate your_env_name
# OR for virtualenv:
# source /path/to/your/venv/bin/activate
# if [ $? -ne 0 ]; then
#   echo "ERROR: Failed to activate environment. Exiting."
#   exit 1
# fi
# echo "Environment activated."
# -------------------------------------------------------

echo "Running Python script: ${PYTHON_SCRIPT}"
echo "Output and errors will be logged to: ${OUTPUT_LOG}"
echo "Process will run in the background (nohup)."

# Run the python script using nohup, passing the model type and output file
# > ${OUTPUT_LOG} redirects stdout to the log file
# 2>&1 redirects stderr to the same place as stdout
# & runs the command in the background
nohup ${PYTHON_EXEC} -u ${PYTHON_SCRIPT} --model_type ${MODEL_CHOICE} --output_file ${OUTPUT_CSV} > ${OUTPUT_LOG} 2>&1 &

# Get the Process ID (PID) of the background job
PID=$!

echo "--------------------------------------------------"
echo "Script started in background with PID: ${PID}"
echo "You can disconnect from the server now."
echo "Monitor progress with: tail -f ${OUTPUT_LOG}"
echo "Check ${OUTPUT_LOG} for errors if the script fails."
echo "The final results will be saved to ${OUTPUT_CSV} upon completion."
echo "--------------------------------------------------"

exit 0
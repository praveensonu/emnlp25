#!/bin/bash

# run_dpo_training.sh - Script to run DPO training with different n_forget values on multiple GPUs
# Usage: ./run_dpo_training.sh

# Check if the script has execute permissions
if [ ! -x "$0" ]; then
    echo "Adding execute permission to this script..."
    chmod +x "$0"
fi

# Function to run training with specific parameters
run_training() {
    local n_forget=$2
    local gpu_id=$0
    local output_dir="${3:-experiments/n_forget_${n_forget}}"
    
    echo "Starting training with n_forget=$n_forget on GPU $gpu_id"
    
    # Create a temporary Python script with modified parameters
    temp_script="temp_dpo_script_${n_forget}.py"
    cp dpo_batch_trainer.py "$temp_script"
    
    # Modify the n_forget parameter in the script
    cat > "run_config_${n_forget}.py" << EOF
from config import Config

class ModifiedConfig(Config):
    def __init__(self):
        super().__init__()
        # Override specific parameters
        self.n_forget_in_batch = $n_forget
        self.total_batch_size = 8  # Keep the total batch size constant
        self.save_dir = "$output_dir"

# Replace the original config with our modified version
cfg = ModifiedConfig()
EOF
    
    # Run the training with specified GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    python "$temp_script" 2>&1 | tee "log_n_forget_${n_forget}.txt" &
    
    # Store the process ID
    echo $! > "pid_n_forget_${n_forget}.txt"
    
    echo "Training job for n_forget=$n_forget started with PID $(cat "pid_n_forget_${n_forget}.txt")"
}

# Define the experiments to run
# Format: n_forget_value:gpu_id:output_directory
declare -a experiments=(
    "4:0:experiments/n_forget_4"
    "5:1:experiments/n_forget_5" 
    "6:2:experiments/n_forget_6"
    "7:3:experiments/n_forget_7"
)

# Create Python wrapper to modify the original script
cat > modify_script.py << 'EOF'
#!/usr/bin/env python3
import sys
import re

def modify_script(input_file, output_file, n_forget):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Replace the n_forget_in_batch value
    content = re.sub(
        r'n_forget_in_batch = \d+', 
        f'n_forget_in_batch = {n_forget}', 
        content
    )
    
    # Adjust n_retain_in_batch accordingly
    content = re.sub(
        r'n_retain_in_batch = total_batch_size - n_forget_in_batch',
        f'n_retain_in_batch = total_batch_size - n_forget_in_batch  # Modified for n_forget={n_forget}',
        content
    )
    
    with open(output_file, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python modify_script.py input_file output_file n_forget")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    n_forget = int(sys.argv[3])
    modify_script(input_file, output_file, n_forget)
EOF

chmod +x modify_script.py

# Launch all experiments
for exp in "${experiments[@]}"; do
    IFS=':' read -r n_forget gpu output_dir <<< "$exp"
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Create temporary script with modified n_forget value
    python modify_script.py dpo_batch_trainer.py "dpo_batch_trainer_${n_forget}.py" "$n_forget"
    
    # Run the training
    run_training "$n_forget" "$gpu" "$output_dir"
    
    # Wait a bit to avoid conflicts
    sleep 2
done

# Function to monitor jobs
monitor_jobs() {
    echo "Monitoring jobs. Press Ctrl+C to exit monitoring (training will continue)"
    
    while true; do
        clear
        echo "=== Training Jobs Status ==="
        echo ""
        
        for exp in "${experiments[@]}"; do
            IFS=':' read -r n_forget gpu output_dir <<< "$exp"
            pid_file="pid_n_forget_${n_forget}.txt"
            
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                if ps -p "$pid" > /dev/null; then
                    echo "n_forget=$n_forget (PID $pid): RUNNING on GPU $gpu"
                    
                    # Show GPU usage for this specific GPU
                    gpu_info=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
                    echo "  GPU $gpu: $gpu_info%"
                else
                    echo "n_forget=$n_forget (PID $pid): FINISHED or FAILED"
                fi
            else
                echo "n_forget=$n_forget: Not started or PID file missing"
            fi
            echo ""
        done
        
        # Check if any jobs are still running
        running_jobs=0
        for exp in "${experiments[@]}"; do
            IFS=':' read -r n_forget _ _ <<< "$exp"
            pid_file="pid_n_forget_${n_forget}.txt"
            
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                if ps -p "$pid" > /dev/null; then
                    running_jobs=$((running_jobs + 1))
                fi
            fi
        done
        
        if [ $running_jobs -eq 0 ]; then
            echo "All jobs have completed!"
            break
        fi
        
        echo "Running jobs: $running_jobs/${#experiments[@]}"
        sleep 10
    done
}

# Start monitoring
monitor_jobs

echo "Script execution completed. Check log files for details."

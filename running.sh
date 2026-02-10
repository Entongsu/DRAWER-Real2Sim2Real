#!/bin/bash

# Check if all necessary arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <script> <checkpoint_file>"
    exit 1
fi

# Assign command-line arguments to variables
SCRIPT=$1                # Path to your training script (input from terminal)
CHECKPOINT_FILE=$2       # Specific checkpoint to check for (input from terminal)

# Memory and GPU thresholds
MEMORY_THRESHOLD=25      # Memory usage threshold in percentage
GPU_THRESHOLD=21000      # GPU memory threshold in MB (21 GB)

# Function to check GPU usage
check_gpu_usage() {
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
    if [ "$GPU_USED" -gt "$GPU_THRESHOLD" ]; then
        echo "GPU memory usage is too high: $GPU_USED MB"
        return 1  # GPU memory exceeds threshold
    else
        return 0  # GPU memory is within acceptable range
    fi
}

# Function to check memory usage
check_memory_usage() {
    
    # Get memory utilization using free
    MEMORY_USED=$(free | awk '/Mem/ { printf("%.2f"), $3/$2 * 100.0 }')
    
    echo "Memory usage: $MEMORY_USED%"
    # Check if memory usage exceeds the threshold
    if (( ${MEMORY_USED%.*} > MEMORY_THRESHOLD )); then
        echo "Warning: Memory usage is too high!"
        return 1
    else
        return 0  # Memory usage is within acceptable range
    fi
}

# Function to check if the specified checkpoint exists
check_checkpoint() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        return 0  # File exists, no need to restart
    else
        echo "Checkpoint $CHECKPOINT_FILE is missing."
        return 1  # File is missing, need to restart
    fi
}


cleanup_and_reset() {
    echo "Cleaning up resources..."
    
    # Restart GPU (if applicable)
    # restart_gpu  # Commented out
    
    # Kill processes using high memory
    echo "Killing high memory usage processes..."
    pkill -f "python"  # Adjust this to target relevant processes
    
    # Wait before retrying
    sleep 10
}

# Function to run the training script
run_script() {
    echo "Running script..."
    eval $SCRIPT
    
    # Capture the exit status of the script
    SCRIPT_EXIT_STATUS=$?
    
    if [ $SCRIPT_EXIT_STATUS -eq 0 ]; then
        echo "Training finished successfully. No restart needed."
        exit 0  # Exit the loop and script if training finished
    else
        echo "Training interrupted or failed. Cleaning up and restarting."
        cleanup_and_reset  # Call the cleanup function on failure or interruption
    fi
}

# Main loop
while true; do
    # Check for GPU and memory usage
    check_gpu_usage
    GPU_STATUS=$?
    check_memory_usage
    MEMORY_STATUS=$?
    
    if [ "$GPU_STATUS" -ne 0 ] || [ "$MEMORY_STATUS" -ne 0 ]; then
        echo "High resource usage detected. Restarting GPU/Memory and rerunning the script..."
        
        # Kill high memory usage processes (this could be your script or other processes)
        if [ "$MEMORY_STATUS" -ne 0 ]; then
            echo "Killing high memory usage processes..."
            pkill -f "python"  # Adjust the process to target relevant processes
        fi
        
        # Wait a few seconds before restarting the script
        sleep 10
    fi
    
    # Check if the specific checkpoint exists
    check_checkpoint
    CHECKPOINT_STATUS=$?
    
    if [ $CHECKPOINT_STATUS -ne 0 ]; then
        echo "Checkpoint is missing. Restarting the training..."
        run_script
    else
        echo "Checkpoint $CHECKPOINT_FILE is present. No need to restart."
        exit 0  # Exit the loop if the checkpoint exists
    fi
    
    # Sleep for a while before checking again (adjust as needed)
    sleep 300
done

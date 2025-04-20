#!/bin/bash

# Base directory (adjust this to your project root)
BASE_DIR=""  # Replace with your actual path
INFERENCE_SCRIPT="$BASE_DIR/safelawbench/inference/vllm_inference.py"


# Default values
NUM_GPUS=8
MASTER_PORT=2954
CONFIG_DIR="$BASE_DIR/hklexsafe/configs/inference/0"  # Directory containing YAML files

# Specify the configs you want to run
CONFIGS_TO_RUN=(
    # "Qwen2.5-72B-Instruct.yaml"
    # "Mistral-Large-Instruct-2411.yaml"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base_dir)
            BASE_DIR="$2"
            INFERENCE_SCRIPT="$BASE_DIR/hklexsafe/inference/vllm_inference.py"
            CONFIG_DIR="$BASE_DIR/hklexsafe/configs/inference"
            shift 2
            ;;
        --config_dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create log directory
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log_message() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $1" | tee -a "$LOG_FILE"
}

# Function to log errors
log_error() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] ERROR: $1" | tee -a "$LOG_FILE"
}

# Function to check if config file exists
check_config() {
    local config_path="$1"
    if [ ! -f "$config_path" ]; then
        log_error "Config file '$config_path' does not exist"
        return 1
    fi
    return 0
}

# Function to run inference for a single config
run_inference() {
    local config_file="$1"
    local config_path="$CONFIG_DIR/$config_file"
    
    log_message "================================================"
    log_message "Starting inference for config: $config_file"
    log_message "================================================"
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your GPU setup
    export MASTER_PORT=$MASTER_PORT
    
    # Add PYTHONPATH to include project root
    export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

    # Run the inference script
    python "$INFERENCE_SCRIPT" --config_file "$config_path" 2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        log_error "Inference failed for $config_file with exit code $exit_code"
        return $exit_code
    fi
    
    log_message "Completed inference for $config_file"
    log_message "------------------------------------------------"
}

# Check if inference script exists
if [ ! -f "$INFERENCE_SCRIPT" ]; then
    log_error "Inference script not found at: $INFERENCE_SCRIPT"
    exit 1
fi

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    log_error "Config directory '$CONFIG_DIR' does not exist"
    exit 1
fi

# Initialize Ray cluster
log_message "Initializing Ray cluster..."
ray start --head --port=$MASTER_PORT

# Track overall success
overall_success=true

# Check all configs exist before starting
for config in "${CONFIGS_TO_RUN[@]}"; do
    if ! check_config "$CONFIG_DIR/$config"; then
        overall_success=false
        break
    fi
done

# Run inference for each config if all configs exist
if [ "$overall_success" = true ]; then
    total_configs=${#CONFIGS_TO_RUN[@]}
    current_config=0
    
    for config in "${CONFIGS_TO_RUN[@]}"; do
        ((current_config++))
        log_message "Processing config $current_config of $total_configs: $config"
        if ! run_inference "$config"; then
            overall_success=false
            break
        fi
    done
fi

# Stop Ray cluster
log_message "Stopping Ray cluster..."
ray stop

# Final status
if [ "$overall_success" = true ]; then
    log_message "All inference runs completed successfully!"
    exit 0
else
    log_error "Some inference runs failed. Check the logs for details."
    exit 1
fi

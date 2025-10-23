# Activate the conda environment
conda activate criticsearch

# Set your Google Search API key
export SER_API_KEY=your_api_key

export CUDA_VISIBLE_DEVICES='0,1'
bash scripts/train_grpo_critic.sh NUM_GPUS_PER_NODE 2 MODEL_PATH models/Qwen/Qwen2.5-0.5B-Instruct DATA_PATH data/CriticSearch_dataset TOTAL_STEPS 203 IP localhost SEARCH_MODE wiki SIMULATION_LLM models/iic/Simulation_LLM_google_3B START_THRESHOLD 0 END_THRESHOLD 0.5 SEARCH_ENGINE wiki MAX_TURNS 5 TOPK 5
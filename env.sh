# conda create -n sglang310 python=3.10 -y
# conda activate sglang310
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

pip install wandb
pip install serpapi
pip3 install flash-attn --no-build-isolation

pip install "sglang[all]"

# CUDA_VISIBLE_DEVICES='3' python -m sglang.launch_server --model-path models/Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --tp 1 --dp 1 --port 8000  --disable-cuda-graph


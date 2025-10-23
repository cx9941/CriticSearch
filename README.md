<div align="center">
<p align="center">
  <img src="assets/model.jpg" width="70%" height="280%" />
</p>
</div>

<div align="center">
<h1>CriticSearch: Incentivize the Search Capability of LLMs with a Critictor
</h1>
</div>


<div align="center">
  <a href='https://alibaba-nlp.github.io/ZeroSearch/'><img src='https://img.shields.io/badge/Homepage-ZeroSearch-6c5ce7?logo=github&logoColor=white'></a>
  <a href='https://arxiv.org/pdf/2505.04588'><img src='https://img.shields.io/badge/Paper-arXiv-d63031?logo=arxiv&logoColor=white'></a>
  <a href='https://huggingface.co/collections/sunhaonlp/zerosearch-v2-6827f4ee6b6265069d443d4e'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-0984e3'></a>
  <a href='https://huggingface.co/datasets/sunhaonlp/ZeroSearch_dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-00b894'></a>
  <a href='https://x.com/_akhaliq/status/1920397374007984516'><img src='https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244'></a><br>
</div>



<p align="center">
  <i><b>Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou</b></i><br>
  <i><b>Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, Jingren Zhou</b></i><br>
  <i>Tongyi Lab <img src="./assets/tongyi.png" width="14px">, Alibaba Group</i>
</p>



# 🔥 News

- **[2025.06.08]** Released the [simulation LLMs](https://huggingface.co/collections/sunhaonlp/simulation-llm-wiki-v2-6857b06122425526d82a42d4) and [policy models](https://huggingface.co/collections/sunhaonlp/zerosearch-policy-wiki-v2-68442dce61d2e68f6623e500) compatible with Wikipedia Search.
- **[2025.05.17]** Released the [simulation LLMs](https://huggingface.co/collections/sunhaonlp/simulation-llm-google-v2-6827f4e45bca955ed2b2d0ba) and [policy models](https://huggingface.co/collections/sunhaonlp/zerosearch-policy-google-v2-6827f4ee6b6265069d443d4e) compatible with Google Search.
- **[2025.05.17]** Released the [simulation tuning dataset](https://huggingface.co/datasets/sunhaonlp/SimulationTuning_dataset).
- **[2025.05.17]** Added support for three RL algorithms: REINFORCE, GPRO, and PPO.
- **[2025.05.08]** Released the initial codebase and paper.


# 🤗 Resources

| Retriever | Simulation Tuning Dataset                                    | Simulation LLMs                                              | Policy Models                                                 |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Wikipedia | •[SimulationTuning\_wiki\_dataset](https://huggingface.co/datasets/sunhaonlp/SimulationTuning_wiki_dataset)     | •[Simulation\_LLM\_wiki\_3B\_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_wiki_3B_V2)<br>•[Simulation\_LLM\_wiki\_7B\_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_wiki_7B_V2)<br>•[Simulation\_LLM\_wiki\_14B\_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_wiki_14B_V2)             | •[ZeroSearch\_wiki\_V2\_Qwen2.5\_3B](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_3B)<br>•[ZeroSearch\_wiki\_V2\_Qwen2.5\_3B\_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_3B_Instruct)<br>•[ZeroSearch\_wiki\_V2\_Llama\_3.2\_3B](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Llama_3.2_3B)<br>•[ZeroSearch\_wiki\_V2\_Llama\_3.2\_3B\_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Llama_3.2_3B_Instruct)<br>•[ZeroSearch\_wiki\_V2\_Qwen2.5\_7B](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B)<br>•[ZeroSearch\_wiki\_V2\_Qwen2.5\_7B\_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B_Instruct)                         |
| Google    | •[SimulationTuning\_google\_dataset](https://huggingface.co/datasets/sunhaonlp/SimulationTuning_google_dataset) | •[Simulation\_LLM\_google\_3B\_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_google_3B_V2)<br>•[Simulation\_LLM\_google\_7B\_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_google_7B_V2)<br>•[Simulation\_LLM\_google\_14B\_V2](https://huggingface.co/sunhaonlp/Simulation_LLM_google_14B_V2) | •[ZeroSearch\_google\_V2\_Qwen2.5\_3B](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B)<br>•[ZeroSearch\_google\_V2\_Qwen2.5\_3B\_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_3B_Instruct)<br>•[ZeroSearch\_google\_V2\_Llama\_3.2\_3B](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Llama_3.2_3B)<br>•[ZeroSearch\_google\_V2\_Llama\_3.2\_3B\_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Llama_3.2_3B_Instruct)<br>•[ZeroSearch\_google\_V2\_Qwen2.5\_7B](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_7B)<br>•[ZeroSearch\_google\_V2\_Qwen2.5\_7B\_Instruct](https://huggingface.co/Alibaba-NLP/ZeroSearch_google_V2_Qwen2.5_7B_Instruct) |

# 📌 Introduction

- We propose ZeroSearch, a novel reinforcement learning framework that incentivizes the capability of LLMs to use a real search engine with simulated searches during training. 
- Through supervised fine-tuning, we transform the LLM into a retrieval module capable of generating both relevant and noisy documents in response to a query. We further introduce a curriculum rollout mechanism to progressively elicit the model’s reasoning ability by exposing it to increasingly challenging retrieval scenarios.
- We conduct extensive experiments on both in-domain and out-of-domain datasets. Results show that ZeroSearch outperforms real search engine-based models while incurring zero API cost. Moreover, it generalizes well across both base and instruction-tuned LLMs of various sizes and supports different reinforcement learning algorithms.

# 🛠 Dependencies

```bash
conda create -n zerosearch python=3.9
conda activate zerosearch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install wandb
pip install serpapi

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation

# sglang
# If you encounter package conflicts when trying to install sglang in the current environment, we recommend creating a new environment and installing sglang there.
pip install sglang[all]
```

```bash
# ==========================================================
# Step 0. 创建并激活新环境
# ==========================================================
conda create -n sglang310 python=3.10 -y
conda activate sglang310

# ==========================================================
# Step 1. 安装匹配 CUDA 12.4 的 PyTorch
# ==========================================================
# 说明：此命令会自动安装 torch / torchvision / torchaudio
#       并确保与 CUDA 12.4 兼容。
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install wandb
pip install serpapi
# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation

# sglang
# If you encounter package conflicts when trying to install sglang in the current environment, we recommend creating a new environment and installing sglang there.
pip install sglang[all]

CUDA_VISIBLE_DEVICES='3' python -m sglang.launch_server --model-path models/Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --tp 1 --dp 1 --port 8000  --disable-cuda-graph
```


# 📖 Quick Start
(1) Download the training dataset.

```bash
huggingface-cli download --repo-type dataset --resume-download sunhaonlp/ZeroSearch_dataset --local-dir ZeroSearch_dataset

# (Optional) Download the Simulation Tuning dataset, required only if you want to train your own simulation LLMs
huggingface-cli download --repo-type dataset --resume-download sunhaonlp/SimulationTuning_dataset --local-dir SimulationTuning_dataset
```

(2) Download the simulation LLMs.

```bash
# Simulation LLMs are available in different parameter sizes. Choose the one that best suits your needs.
# The 14B version is recommended for its stable and reliable simulation performance.
huggingface-cli download --resume-download sunhaonlp/Simulation_LLM_google_3B_V2 --local-dir Simulation_LLM_google_3B

huggingface-cli download --resume-download sunhaonlp/Simulation_LLM_google_7B_V2 --local-dir Simulation_LLM_google_7B

huggingface-cli download --resume-download sunhaonlp/Simulation_LLM_google_14B_V2 --local-dir Simulation_LLM_google_14B
```

(3) Launch a local simulation server.

```bash
# Prompt-based simulation
python -m sglang.launch_server --model-path Qwen2.5-14B-Instruct --host 0.0.0.0 --tp 2 --dp 2 --port 6001

# Fine-tuning-based simulation
python -m sglang.launch_server --model-path Simulation_LLM_google_14B --host 0.0.0.0 --tp 2 --dp 2 --port 6001
```

(4) Conduct RL training with Qwen2.5-3B-Instruct.

```bash
# Activate the conda environment
conda activate zerosearch

# Set your Google Search API key
export SER_API_KEY=your_api_key

# You can run REINFORCE, GRPO or PPO training using the scripts below.
# The START_THRESHOLD and END_THRESHOLD parameters define the initial and final difficulty levels of the training tasks. Adjusting these values can help optimize model performance.

## Prompt-based simulation
bash train_reinforce.sh NUM_GPUS_PER_NODE 4 MODEL_PATH Qwen2.5-3B-Instruct DATA_PATH ZeroSearch_dataset TOTAL_STEPS 203 IP localhost SEARCH_MODE simulate_prompt SIMULATION_LLM Qwen2.5-14B-Instruct START_THRESHOLD 0 END_THRESHOLD 0.5 SEARCH_ENGINE google MAX_TURNS 5 TOPK 5
bash train_grpo.sh NUM_GPUS_PER_NODE 4 MODEL_PATH Qwen2.5-3B-Instruct DATA_PATH ZeroSearch_dataset TOTAL_STEPS 203 IP localhost SEARCH_MODE simulate_prompt SIMULATION_LLM Qwen2.5-14B-Instruct START_THRESHOLD 0 END_THRESHOLD 0.5 SEARCH_ENGINE google MAX_TURNS 5 TOPK 5
bash train_ppo.sh NUM_GPUS_PER_NODE 4 MODEL_PATH Qwen2.5-3B-Instruct DATA_PATH ZeroSearch_dataset TOTAL_STEPS 203 IP localhost SEARCH_MODE simulate_prompt SIMULATION_LLM Qwen2.5-14B-Instruct START_THRESHOLD 0 END_THRESHOLD 0.5 SEARCH_ENGINE google MAX_TURNS 5 TOPK 5

## Fine-tuning-based simulation
bash train_reinforce.sh NUM_GPUS_PER_NODE 4 MODEL_PATH Qwen2.5-3B-Instruct DATA_PATH ZeroSearch_dataset TOTAL_STEPS 203 IP localhost SEARCH_MODE simulate_sft SIMULATION_LLM Simulation_LLM_google_14B START_THRESHOLD 0 END_THRESHOLD 0.5 SEARCH_ENGINE google MAX_TURNS 5 TOPK 5
bash train_grpo.sh NUM_GPUS_PER_NODE 4 MODEL_PATH Qwen2.5-3B-Instruct DATA_PATH ZeroSearch_dataset TOTAL_STEPS 203 IP localhost SEARCH_MODE simulate_sft SIMULATION_LLM Simulation_LLM_google_14B START_THRESHOLD 0 END_THRESHOLD 0.5 SEARCH_ENGINE google MAX_TURNS 5 TOPK 5
bash train_ppo.sh NUM_GPUS_PER_NODE 4 MODEL_PATH Qwen2.5-3B-Instruct DATA_PATH ZeroSearch_dataset TOTAL_STEPS 203 IP localhost SEARCH_MODE simulate_sft SIMULATION_LLM Simulation_LLM_google_14B START_THRESHOLD 0 END_THRESHOLD 0.5 SEARCH_ENGINE google MAX_TURNS 5 TOPK 5
```

# 💡 Performance

### 📊 Main Results

<div align="center">
    <img src="assets/results.jpg" width="80%" height="auto" />
</div>

### 📊 Compare ZeroSearch with Real Search Engine 

<div align="center">
    <img src="assets/compare_real_search.jpg" width="80%" height="auto" />
</div>

### 📊 Choice of Simulation LLMs

<div align="center">
    <img src="assets/compare_simulation_llm.jpg" width="80%" height="auto" />
</div>

### 📊 Case Study

<div align="center">
    <img src="assets/case_study.jpg" width="80%" height="auto" />
</div>


# 🙏 Acknowledgements

This work is implemented based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [veRL](https://github.com/volcengine/verl), and [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main). We sincerely thank the authors of these projects for their valuable contributions to the open-source community.

## 👍 Awesome work inspired by ZeroSearch

- [SSRL](https://github.com/TsinghuaC3I/SSRL): SSRL: Self-Search Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/TsinghuaC3I/SSRL)](https://github.com/TsinghuaC3I/SSRL)


# 📧 Contact

If you have any questions, feel free to reach out to me via email: [sunhao@stu.pku.edu.cn](mailto:sunhao@stu.pku.edu.cn)

## 🚩Citation

If this work is helpful, please kindly cite as:

```bigquery
@article{sun2025zerosearch,
  title={ZeroSearch: Incentivize the Search Capability of LLMs without Searching},
  author={Sun, Hao and Qiao, Zile and Guo, Jiayan and Fan, Xuanbo and Hou, Yingyan and Jiang, Yong and Xie, Pengjun and Huang, Fei and Zhang, Yan},
  journal={arXiv preprint arXiv:2505.04588},
  year={2025}
}
```

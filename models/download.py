#模型下载
from modelscope import snapshot_download
# model_dir = snapshot_download('iic/ZeroSearch_google_V2_Qwen2.5_3B', cache_dir='.')
# model_dir = snapshot_download('iic/ZeroSearch_google_V2_Qwen2.5_3B_Instruct', cache_dir='.')
# model_dir = snapshot_download('iic/ZeroSearch_google_V2_Qwen2.5_7B', cache_dir='.')
# model_dir = snapshot_download('iic/ZeroSearch_google_V2_Qwen2.5_7B_Instruct', cache_dir='.')
model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='.')
model_dir = snapshot_download('Qwen/Qwen2.5-3B-Instruct', cache_dir='.')
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='.')

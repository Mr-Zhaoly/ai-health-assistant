# In[1]:

#模型下载
from modelscope import snapshot_download
#model_dir = snapshot_download('iic/gte_Qwen2-7B-instruct', cache_dir='/root/autodl-tmp/models')
model_dir = snapshot_download('iic/gte_Qwen2-1.5B-instruct', cache_dir='D:/LLM/gte-qwen2')
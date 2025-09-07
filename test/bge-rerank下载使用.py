# In[1]:
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-reranker-large', cache_dir='D:/LLM/bge-reranker')

# In[2]:
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('D:/LLM/bge-reranker/BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('D:/LLM/bge-reranker/BAAI/bge-reranker-large')
model.eval()

pairs = [['what is panda?', 'The giant panda is a bear species endemic to China.']]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
scores = model(**inputs).logits.view(-1).float()
print(scores)  # 输出相关性分数
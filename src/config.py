import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # DashScope配置
    DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
    DEEPSEEK_MODEL = "deepseek-v3"
    TEXT_EMBEDDING = "text-embedding-v4"

    # 文件路径
    PDF_PATH = "D:/code/ai-health-assistant/data/中国居民膳食指南.pdf"
    IMAGES_PATH = "D:/code/ai-health-assistant/data/images/"
    PROCESSED_DIR = "D://code//ai-health-assistant//data//processed//"
    KNOWLEDGE_BASE_DIR = "D://code//ai-health-assistant//knowledge_base//"
    NUTRITION_DICT_PATH = "D:/code/ai-health-assistant/data/nutrition_dict.txt"

    # 大模型路径
    BGE_RERANKER_PATH = "D:/LLM/bge-reranker/BAAI/bge-reranker-large"

    # 文本处理参数
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

config = Config()
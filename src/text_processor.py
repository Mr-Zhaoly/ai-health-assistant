import re
import jieba
import os
from typing import List


class TextProcessor:
    def __init__(self, dict_path="data/nutrition_dict.txt"):
        # 检查词典文件是否存在
        if os.path.exists(dict_path):
            jieba.load_userdict(dict_path)
            print(f"已加载自定义词典: {dict_path}")
        else:
            # 添加一些默认的营养学术语
            default_dict = [
                "营养素", "膳食纤维", "蛋白质", "碳水化合物", "脂肪",
                "维生素", "矿物质", "微量元素", "能量代谢", "营养密度",
                "血糖指数", "膳食指南", "食物多样", "均衡膳食", "营养标签",
                "膳食宝塔", "中国居民", "健康体重", "营养状况", "食物成分"
            ]
            for term in default_dict:
                jieba.add_word(term)
            print(f"使用内置默认营养学术语词典")

    def clean_text(self, text: str) -> str:
        """清洗中文文本"""
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fa5，。；：？！、（）【】《》“”‘’\s\w]', ' ', text)
        # 合并多余空白
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def segment_text(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """将长文本分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # 确保不在句子中间截断
            if end < len(text):
                # 尝试在句号处截断
                sentence_end = text.rfind('。', start, end)
                if sentence_end > start + chunk_size // 2:  # 确保不会截取太短的块
                    end = sentence_end + 1
            chunks.append(text[start:end])
            start = end - overlap  # 设置重叠部分
        return chunks
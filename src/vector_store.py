from typing import List

import numpy as np
import json
import os
from pathlib import Path


class VectorStore:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.embeddings = None
        self.texts = []
        self.metadata = []

        # 确保目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def load(self):
        """加载已有的向量存储"""
        embeddings_path = self.storage_path / "embeddings.npy"
        texts_path = self.storage_path / "texts.json"
        metadata_path = self.storage_path / "metadata.json"

        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        if texts_path.exists():
            with open(texts_path, 'r', encoding='utf-8') as f:
                self.texts = json.load(f)
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

    def save(self):
        """保存向量存储"""
        np.save(self.storage_path / "embeddings.npy", self.embeddings)
        with open(self.storage_path / "texts.json", 'w', encoding='utf-8') as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)
        with open(self.storage_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_embeddings(self, new_embeddings: np.ndarray, new_texts: List[str], new_metadata: List[dict] = None):
        """添加新的嵌入向量"""
        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = new_embeddings.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.texts.extend(new_texts)
        if new_metadata:
            self.metadata.extend(new_metadata)
        else:
            self.metadata.extend([{}] * len(new_texts))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """搜索最相关的文本块"""
        if self.embeddings is None:
            return []

        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 获取最相似的top_k个结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx])
            })

        return results
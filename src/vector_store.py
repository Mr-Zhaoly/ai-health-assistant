from typing import List, Any

import numpy as np
import json
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_community.llms import Tongyi
from src.config import Config


class VectorStore:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.config = Config
        self.faiss_store = None
        self.texts = []
        self.metadata = []
        self.embeddings_model = DashScopeEmbeddings(
            model=self.config.TEXT_EMBEDDING,
            dashscope_api_key=self.config.DASHSCOPE_API_KEY,
        )
        self.llm = Tongyi(model_name=self.config.DEEPSEEK_MODEL, dashscope_api_key=self.config.DASHSCOPE_API_KEY)

        # 确保目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def load(self):
        """加载已有的向量存储"""
        index_path = self.storage_path / "faiss/index.faiss"
        texts_path = self.storage_path / "texts.json"
        metadata_path = self.storage_path / "metadata.json"

        if index_path.exists():
            self.faiss_store = FAISS.load_local(
                str(self.storage_path / "faiss"),
                self.embeddings_model,
                allow_dangerous_deserialization=True
            )
        if texts_path.exists():
            with open(texts_path, 'r', encoding='utf-8') as f:
                self.texts = json.load(f)
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

    def save(self):
        """保存向量存储到FAISS"""
        if self.faiss_store is not None:
            self.faiss_store.save_local(str(self.storage_path / "faiss"))
        with open(self.storage_path / "texts.json", 'w', encoding='utf-8') as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)
        with open(self.storage_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_embeddings(self, new_texts: List[str], new_metadata: List[dict] = None):
        """添加新的嵌入向量到FAISS"""
        if new_metadata is None:
            new_metadata = [{}] * len(new_texts)

        # 创建Document对象
        documents = [
            Document(
                page_content=text,
                metadata=meta
            )
            for text, meta in zip(new_texts, new_metadata)
        ]

        if self.faiss_store is None:
            # 创建新的FAISS存储
            self.faiss_store = FAISS.from_documents(documents, self.embeddings_model)
        else:
            # 添加到现有存储
            self.faiss_store.add_documents(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """使用FAISS搜索最相关的文本块"""
        if self.faiss_store is None:
            return []

        # FAISS搜索
        results = self.faiss_store.similarity_search_by_vector(query_embedding, k=top_k)

        # 格式化结果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity": 0.0  # FAISS不直接返回相似度分数
            })

        return formatted_results
from langchain_community.llms.tongyi import Tongyi

from .config import Config
from .dashscope_client import DashScopeClient
from .vector_store import VectorStore
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
from langchain.retrievers import MultiQueryRetriever



class QueryEngine:
    def __init__(self, dashscope_client: DashScopeClient, vector_store: VectorStore):
        self.client = dashscope_client
        self.vector_store = vector_store
        self.config = Config
        # 初始化 BGE Reranker（已经提前下载好模型）
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.config.BGE_RERANKER_PATH)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.config.BGE_RERANKER_PATH).eval()

    def query(self, question: str, top_k: int = 3, rerank_top_n: int = 10) -> tuple[str, list[dict]]:
        """处理用户查询"""
        # 获取问题的向量表示
        question_embedding = self.client.get_embeddings([question])[0]

        # 初步检索更多候选上下文 (例如前10个)
        candidate_chunks = self.vector_store.search(question_embedding, top_k=rerank_top_n)

        # 使用 BGE Reranker 重新排序
        reranked_chunks = self._rerank(question, candidate_chunks)

        # 取重排序后分数最高的 top_k 个
        relevant_chunks = reranked_chunks[:top_k]

        # 组合上下文
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

        # 生成回答
        response = self.client.generate_response(question, context)

        return response, relevant_chunks

    def _rerank(self, query: str, chunks: list[dict], threshold: float = None) -> list[dict]:
        """使用 BGE Reranker 对检索结果进行重排序"""
        # 构造输入对
        sentence_pairs = [[query, chunk['text']] for chunk in chunks]

        # 编码并推理
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                sentence_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            scores = self.reranker_model(**inputs).logits.view(-1).float()

        # 添加分数到 chunks 并排序
        for i, chunk in enumerate(chunks):
            chunk['score'] = float(scores[i])

        # 过滤低分结果（可选）
        if threshold is not None:
            chunks = [chunk for chunk in chunks if chunk['score'] >= threshold]

        # 按分数降序排列
        return sorted(chunks, key=lambda x: x['score'], reverse=True)
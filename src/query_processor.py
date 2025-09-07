from .dashscope_client import DashScopeClient
from .vector_store import VectorStore


class QueryEngine:
    def __init__(self, dashscope_client: DashScopeClient, vector_store: VectorStore):
        self.client = dashscope_client
        self.vector_store = vector_store

    def query(self, question: str, top_k: int = 3) -> tuple[str, list[dict]]:
        """处理用户查询"""
        # 获取问题的向量表示
        question_embedding = self.client.get_embeddings([question])[0]

        # 在知识库中搜索相关上下文
        relevant_chunks = self.vector_store.search(question_embedding, top_k=top_k)

        # 组合上下文
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

        # 生成回答
        response = self.client.generate_response(question, context)

        return response, relevant_chunks
from .config import Config
from .dashscope_client import DashScopeClient
from .vector_store import VectorStore
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback
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

    def create_multi_query_retriever(self, k = 4):
        """
        创建MultiQueryRetriever

        参数:
            vectorstore: 向量数据库
            llm: 大语言模型，用于查询改写

        返回:
            retriever: MultiQueryRetriever对象
        """
        # 创建基础检索器
        base_retriever = self.vector_store.faiss_store.as_retriever(search_kwargs={"k": k})

        # 创建MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.vector_store.llm
        )
        return retriever

    def process_query_with_multi_retriever(self, query: str, retriever):
        """
        使用MultiQueryRetriever处理查询

        参数:
            query: 用户查询
            retriever: MultiQueryRetriever对象
            llm: 大语言模型

        返回:
            response: 回答
            unique_pages: 相关文档的页码集合
        """
        # 执行查询，获取相关文档
        docs = retriever.invoke(query)
        print(f"找到 {len(docs)} 个相关文档")

        # 加载问答链
        chain = load_qa_chain(self.vector_store.llm, chain_type="stuff")

        # 准备输入数据
        input_data = {"input_documents": docs, "question": query}

        # 使用回调函数跟踪API调用成本
        with get_openai_callback() as cost:
            # 执行问答链
            response = chain.invoke(input=input_data)
            print(f"查询已处理。成本: {cost}")

        # 记录源数据
        sources = []

        # 获取每个文档块的来源页码
        for doc in docs:
            text_content = getattr(doc, "page_content", "")
            sources.append({"text": text_content, "metadata": doc.metadata, "similarity": 0.0})

        return response["output_text"], sources
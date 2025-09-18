import streamlit as st
from src.config import Config
from src.pdf_processor import PDFProcessor
from src.text_processor import TextProcessor
from src.dashscope_client import DashScopeClient
from src.vector_store import VectorStore
from src.query_processor import QueryEngine
from src.query_rewriter_processor import QueryRewriter
from PyPDF2 import PdfReader
import os


class HealthAssistantApp:
    def __init__(self):
        self.config = Config()
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor()
        self.dashscope_client = DashScopeClient(self.config.DASHSCOPE_API_KEY)
        self.vector_store = VectorStore(self.config.KNOWLEDGE_BASE_DIR)
        self.query_engine = None
        self.query_rewriter = None

        # 加载或初始化知识库
        self.initialize_knowledge_base()

    def initialize_knowledge_base(self):
        """初始化或加载知识库"""
        self.vector_store.load()

        # 如果知识库为空，处理PDF并创建知识库
        if not self.vector_store.faiss_store:
            print("正在初始化知识库，这可能需要一些时间...")
            self.process_pdf_and_build_kb()

        self.query_engine = QueryEngine(self.dashscope_client, self.vector_store)
        self.query_rewriter = QueryRewriter(self.dashscope_client)

    def process_pdf_and_build_kb(self):
        """处理PDF并构建知识库"""
        # # 读取PDF文件
        # pdf_reader = PdfReader(self.config.PDF_PATH)
        # # pages_text, char_page_mapping = self.pdf_processor.extract_text_with_page_numbers(pdf_reader)
        # 处理images目录中的独立图片文件
        print("  - 正在处理独立图片文件...")
        img_dir = self.config.IMAGES_PATH
        pages_text = self.pdf_processor.images_to_text(img_dir)

        # 清理和分块文本
        all_chunks = []
        for page_num, text in enumerate(pages_text):
            cleaned_text = self.text_processor.clean_text(text)
            chunks = self.text_processor.chunk_text(
                cleaned_text,
                self.config.CHUNK_SIZE,
                self.config.CHUNK_OVERLAP
            )
            for chunk in chunks:
                all_chunks.append(chunk)

        # 添加到向量存储
        metadata = [{"source": "中国居民膳食指南（2022）", "page": i // 10 + 1} for i in range(len(all_chunks))]
        self.vector_store.add_embeddings(all_chunks, metadata)
        self.vector_store.save()

    def run(self):
        """运行应用"""
        print("AI健康助手 - 基于中国居民膳食指南（2022）")

        print("我可以回答关于中国膳食指南和营养健康的问题")

        history = []

        while True:
            # 提示用户输入
            user_input = input("请输入你的问题（输入 quit/exit 退出）: ").strip()
            # 检查是否要退出
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break  # 退出 while 循环
            new_query = user_input
            if history:
                # 比较型Query改写
                new_query = self.query_rewriter.rewrite_comparative_query(user_input, history)
                # 上下文依赖型Query改写
                # new_query = self.query_rewriter.rewrite_context_dependent_query(user_input, history)

            # （模拟）回答逻辑 —— 这里可以替换成你自己的逻辑，比如调用大模型 API
            print("用户:", new_query)
            response, sources = self.query_engine.query(new_query, top_k=3, rerank_top_n=10)
            history.append("用户:" + user_input)
            history.append("健康助手管家:" + response)
            # 输出回答
            print("健康助手管家:", response)
            print("-" * 40)  # 分隔线，美观一点

        # question = input("请输入你的问题: ")

        # if question:
        #     print("正在思考...")
        #     # 正常查询整合重排序
        #     response, sources = self.query_engine.query(question, top_k=3, rerank_top_n=10)
        #     # 使用MultiQueryRetriever多语义查询
        #     # response, sources = self.query_engine.process_query_with_multi_retriever(question, self.query_engine.create_multi_query_retriever(4))
        #
        #     print("回答:")
        #     print(response)
        #     # 上下文依赖型Query改写
        #     print("正在对问题进行上下文依赖型Query改写...")
        #     new_query = self.query_rewriter.rewrite_context_dependent_query(question, [])
        #     print("改写后的问题:", new_query)
        #
        #     print("参考来源:")
        #     for i, source in enumerate(sources):
        #         print(f"来源 {i + 1} (相似度: {source['similarity']:.3f})")
        #         print(source["text"])
        #         if "page" in source["metadata"]:
        #             print(f"来自第 {source['metadata']['page']} 页")


if __name__ == "__main__":
    app = HealthAssistantApp()
    app.run()
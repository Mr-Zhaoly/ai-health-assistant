import streamlit as st
from src.config import Config
from src.pdf_processor import PDFProcessor
from src.text_processor import TextProcessor
from src.dashscope_client import DashScopeClient
from src.vector_store import VectorStore
from src.query_processor import QueryEngine
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

        # 加载或初始化知识库
        self.initialize_knowledge_base()

    def initialize_knowledge_base(self):
        """初始化或加载知识库"""
        self.vector_store.load()

        # 如果知识库为空，处理PDF并创建知识库
        if not self.vector_store.texts:
            st.info("正在初始化知识库，这可能需要一些时间...")
            self.process_pdf_and_build_kb()

        self.query_engine = QueryEngine(self.dashscope_client, self.vector_store)

    def process_pdf_and_build_kb(self):
        """处理PDF并构建知识库"""
        # # 读取PDF文件
        # pdf_reader = PdfReader(self.config.PDF_PATH)
        # # pages_text, char_page_mapping = self.pdf_processor.extract_text_with_page_numbers(pdf_reader)
        # 处理images目录中的独立图片文件
        print("  - 正在处理独立图片文件...")
        pages_text = []
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

        # 获取嵌入向量
        embeddings = self.dashscope_client.get_embeddings(all_chunks)

        # 添加到向量存储
        metadata = [{"source": "中国居民膳食指南（2022）", "page": i // 10 + 1} for i in range(len(all_chunks))]
        self.vector_store.add_embeddings(embeddings, all_chunks, metadata)
        self.vector_store.save()

    def run(self):
        """运行应用"""
        print("AI健康助手 - 基于中国居民膳食指南（2022）")

        print("我可以回答关于中国膳食指南和营养健康的问题")

        question = input("请输入你的问题: ")

        if question:
            print("正在思考...")
            response, sources = self.query_engine.query(question)

            print("回答:")
            print(response)

            print("参考来源:")
            for i, source in enumerate(sources):
                print(f"来源 {i + 1} (相似度: {source['similarity']:.3f})")
                print(source["text"])
                if "page" in source["metadata"]:
                    print(f"来自第 {source['metadata']['page']} 页")


if __name__ == "__main__":
    app = HealthAssistantApp()
    app.run()
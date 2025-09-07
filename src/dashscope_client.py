import dashscope
from dashscope import TextEmbedding, Generation
from typing import List
import numpy as np
from src.config import Config


class DashScopeClient:
    def __init__(self, api_key: str):
        dashscope.api_key = api_key
        self.config = Config()

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本向量"""
        responses = []
        for text in texts:
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v4,
                input=text,
                dimensions=1024
            )
            print(f"    - resp: {resp}")
            if resp.status_code == 200:
                responses.append(np.array(resp.output['embeddings'][0]['embedding']))
            else:
                raise Exception(f"Embedding error: {resp.code} - {resp.message}")
        return np.array(responses)

    def generate_response(self, prompt: str, context: str = "") -> str:
        """使用DeepSeek-V3生成回答"""
        full_prompt = f"基于以下知识：{context}\n\n请回答：{prompt}" if context else prompt

        response = Generation.call(
            model=self.config.DEEPSEEK_MODEL,
            prompt=full_prompt,
            max_tokens=1500,
            temperature=0.1  # 低温度确保回答更准确
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"抱歉，生成回答时出错: {response.code} - {response.message}"
from src.dashscope_client import DashScopeClient

class QueryRewriter:
    def __init__(self, dashscope_client: DashScopeClient):
        self.client = dashscope_client

    def rewrite_context_dependent_query(self, current_query, conversation_history) -> str:
        """上下文依赖型Query改写"""
        instruction = """
            你是一个智能的查询优化助手。请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖于上下文。
            如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。
            如果不依赖，直接返回原问题。
            """

        prompt = f"""
            ### 指令 ###
            {instruction}
        
            ### 对话历史 ###
            {conversation_history}
        
            ### 当前问题 ###
            {current_query}
        
            ### 改写后的问题 ###
            """
        return self.client.get_completion(prompt)
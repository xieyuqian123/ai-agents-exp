import os
from typing import Union, Generator, Optional
from openai import OpenAI

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    支持流式 (Streaming) 和非流式响应。
    """
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        
        环境变量映射:
        - api_key -> SILICONFLOW_API_KEY
        - base_url -> SILICONFLOW_BASE_URL (默认: https://api.siliconflow.cn/v1)
        - model -> MODEL_ID (默认: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
        """
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = model or os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")

        if not self.api_key:
            raise ValueError("API Key 未提供。请在构造函数中传入或设置 SILICONFLOW_API_KEY 环境变量。")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, system_prompt: str, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        调用LLM API来生成回应。
        
        Args:
            prompt: 用户的输入提示。
            system_prompt: 系统提示词。
            stream: 是否开启流式传输。默认为 False。
            
        Returns:
            如果 stream=False，返回完整的响应字符串。
            如果 stream=True，返回一个生成器，逐块产生响应内容。
        """
        print(f"正在调用大语言模型 (Stream={stream})...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream
            )

            if stream:
                return self._handle_stream(response)
            else:
                answer = response.choices[0].message.content
                print("大语言模型响应成功。")
                return answer
                
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"

    def _handle_stream(self, response) -> Generator[str, None, None]:
        """处理流式响应的辅助方法"""
        full_content = []
        try:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content.append(content)
                    yield content
            print("\n大语言模型流式响应结束。")
        except Exception as e:
            print(f"流式处理过程中出错: {e}")
            yield f"[Error: {e}]"

import os
import re
from dotenv import load_dotenv

# 使用绝对导入，从 travel_agent 包中导入我们需要的模块和变量
from travel_agent.llm_client import OpenAICompatibleClient
from travel_agent.available_tools import available_tools
from travel_agent.prompt import AGENT_SYSTEM_PROMPT

def main():
    """
    旅行规划智能体的主函数。
    """
    # 加载环境变量
    load_dotenv()

    # 从环境变量获取 API Keys 和模型配置
    API_KEY = os.getenv("SILICONFLOW_API_KEY")
    BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # 检查关键的 API Keys 是否已设置
    if not API_KEY or not TAVILY_API_KEY:
        raise ValueError("请确保在.env文件中设置了 SILICONFLOW_API_KEY 和 TAVILY_API_KEY")

    # 将 Tavily API Key 设置到环境变量中，以便 get_attraction 工具函数可以访问
    os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

    # 初始化 LLM 客户端
    # 参数现在是可选的，如果留空会自动从环境变量读取
    # 这里我们演示显式传入（保持原样），或者您可以简化为 llm = OpenAICompatibleClient()
    llm = OpenAICompatibleClient(model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL)

    # 定义用户的初始问题
    user_prompt = "我下周想去北京玩，请帮我推荐一些适合的景点"
    print(f"用户问题: {user_prompt}")

    current_prompt = user_prompt
    max_turns = 5  # 设置最大对话轮次，防止无限循环

    for i in range(max_turns):
        print(f"\n--- 第 {i+1} 轮 ---")
        
        # 1. 调用大语言模型生成思考和行动
        response_text = llm.generate(current_prompt, AGENT_SYSTEM_PROMPT)
        print(f"LLM响应: {response_text}")

        # 2. 检查是否需要终止循环
        if "finish(" in response_text:
            final_answer_match = re.search(r'finish\(answer="(.*)"\)', response_text, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1)
                print(f"\n✅ 最终答案: {final_answer}")
                break
        
        # 3. 解析并执行工具调用
        action_match = re.search(r'Action: (.*)', response_text, re.DOTALL)
        if action_match:
            action_str = action_match.group(1).strip()
            
            # 解析工具名称和参数
            tool_name_match = re.match(r'(\w+)\(', action_str)
            if not tool_name_match:
                print("❌ 错误: 无法解析工具名称。")
                current_prompt = "错误: 我无法解析你上一个响应中的工具名称。"
                continue

            tool_name = tool_name_match.group(1)
            
            # 提取参数字符串
            arg_str_match = re.search(r'\((.*)\)', action_str)
            if not arg_str_match:
                print("❌ 错误: 无法解析工具参数。")
                current_prompt = "错误: 我无法解析你上一个响应中的工具参数。"
                continue
            
            arg_str = arg_str_match.group(1)
            
            try:
                # 解析参数
                args = dict(re.findall(r'(\w+)="([^"]*)"', arg_str))
            except Exception:
                print(f"❌ 错误: 解析参数 '{arg_str}' 失败。")
                current_prompt = f"错误: 我无法解析你上一个响应中的参数 '{arg_str}'。"
                continue

            # 4. 执行工具
            if tool_name in available_tools:
                tool_function = available_tools[tool_name]
                try:
                    tool_result = tool_function(**args)
                    print(f"工具 '{tool_name}' 已执行，结果: {tool_result}")
                    # 将工具结果作为下一轮的输入
                    current_prompt = f"这是上次工具调用的结果: {tool_result}"
                except Exception as e:
                    print(f"❌ 错误: 执行工具 '{tool_name}' 时出错: {e}")
                    current_prompt = f"错误: 执行工具 '{tool_name}' 时出错: {e}"
            else:
                print(f"❌ 错误: 尝试调用不存在的工具 '{tool_name}'")
                current_prompt = f"错误: 你尝试调用的工具 '{tool_name}' 不存在。"
        else:
            print("⚠️ 警告: 未找到有效的 'Action:'，智能体可能已偏离轨道。正在使用原始响应重试。")
            current_prompt = response_text # 将不规范的输出直接作为下一轮的输入，给模型一个修正的机会

    else:
        print("\n⚠️ 已达到最大对话轮次，程序终止。")

if __name__ == "__main__":
    main()

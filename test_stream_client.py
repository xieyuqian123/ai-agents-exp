import os
import sys
from dotenv import load_dotenv

# Ensure we can import travel_agent
sys.path.append(os.getcwd())

from travel_agent.llm_client import OpenAICompatibleClient

def test_stream():
    load_dotenv()
    
    API_KEY = os.getenv("SILICONFLOW_API_KEY")
    # If using local env, ensure these are set. If not, this test might fail if keys are missing.
    # But user has been running the agent so keys should be in .env
    
    if not API_KEY:
        print("Skipping test: SILICONFLOW_API_KEY not found in env.")
        return

    # BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    # MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    
    client = OpenAICompatibleClient()
    
    print("Testing Streaming Response:")
    print("-" * 20)
    
    try:
        # Request a stream
        stream = client.generate("Simply say 'Streaming works!'", "You are a test bot.", stream=True)
        
        full_response = ""
        for chunk in stream:
            print(chunk, end="", flush=True)
            full_response += chunk
            
        print("\n" + "-" * 20)
        print("Test Complete.")
        
    except Exception as e:
        print(f"\nTest Failed: {e}")

if __name__ == "__main__":
    test_stream()

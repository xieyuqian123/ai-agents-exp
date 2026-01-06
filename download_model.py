import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    # 模型名称
    model_id = "Qwen/Qwen1.5-0.5B-Chat"
    
    # 本地保存路径
    save_directory = os.path.join(os.getcwd(), "models", "Qwen1.5-0.5B-Chat")
    
    print(f"准备下载模型: {model_id}")
    print(f"保存路径: {save_directory}")
    
    # 创建保存目录
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    try:
        # 下载并加载分词器
        print("正在下载分词器 (Tokenizer)...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(save_directory)
        print("分词器下载并保存完成。")
        
        # 下载并加载模型
        print("正在下载模型 (Model)... 这可能需要一些时间。")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.save_pretrained(save_directory)
        print("模型下载并保存完成。")
        
        print(f"\n✅ 成功！模型已保存至: {save_directory}")
        
    except Exception as e:
        print(f"\n❌ 下载过程中发生错误: {e}")

if __name__ == "__main__":
    download_model()

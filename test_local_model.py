import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model():
    # 1. 指向我们刚刚下载好的【本地路径】
    # 使用绝对路径或相对路径均可，这里使用相对路径
    local_model_path = "./models/Qwen1.5-0.5B-Chat" 
    
    # 检查路径是否存在
    if not os.path.exists(local_model_path):
        print(f"❌ 本地路径不存在: {local_model_path}")
        print("请先运行 python download_model.py 下载模型。")
        # 如果本地没有，回退到云端 ID (可选)
        model_id = "Qwen/Qwen1.5-0.5B-Chat"
        print(f"⚠️ 将尝试从 Hugging Face 在线加载: {model_id}")
    else:
        print(f"✅ 发现本地模型: {local_model_path}")
        model_id = local_model_path

    # 2. 设置设备 (Mac 上通常是 cpu，如果是 M芯片可以用 mps，但这里先用通用逻辑)
    # 注意：mps (Metal Performance Shaders) 是 Mac 的 GPU 加速，但 pytorch 支持情况视版本而定
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" 
    else:
        device = "cpu"
        
    print(f"🚀 使用设备: {device}")

    try:
        # 3. 加载分词器 (从本地)
        print("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 4. 加载模型 (从本地) 并移动到设备
        print("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        print("🎉 模型加载完成！准备生成测试...")

        # 5. 使用对话模板进行测试
        messages = [
            {"role": "system", "content": "你是一个有用的助理。"},
            {"role": "user", "content": "你好，请写一首现代诗。"}
        ]
        
        # 使用 apply_chat_template 自动格式化输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"\n构建的 Prompt:\n{text}")
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # 生成回复
        # ---------------------------------------------------------
        # 采样参数配置区
        # ---------------------------------------------------------
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,       # 最大生成长度
            
            do_sample=True,           # 【关键】开启采样模式，否则下面的 top_k/top_p 不生效
            
            temperature=0.1,          # 温度：越低越保守(0.1)，越高越发散(1.0+)
            top_k=50,                 # Top-K：每一步只考虑概率最高的 50 个词
            top_p=0.9,                # Top-P (核采样)：只考虑累积概率达到 90% 的词
            
            repetition_penalty=1.1,   # 重复惩罚：>1.0 表示惩罚重复内容，减少复读机现象
            
            pad_token_id=tokenizer.eos_token_id # 防止警告
        )
        # ---------------------------------------------------------
        
        # 只解码新生成的 token
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\n🤖 模型回复:\n{response}")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    test_model()

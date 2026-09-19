#!/usr/bin/env python3
"""
OpenAI API Key 连通性验证脚本（Windows 适用）
"""
import os
import sys

try:
    import openai
except ImportError:
    print("错误：未安装 openai 库。请执行：pip install openai")
    sys.exit(1)

# ---------- 配置 ----------
API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    API_KEY = input("请输入您的 OpenAI API Key: ").strip()
    if not API_KEY:
        print("错误：未提供 API Key。")
        sys.exit(1)

# ---------- 创建客户端 ----------
client = openai.OpenAI(
    api_key=API_KEY,
    timeout=30.0,
)

def test_connection():
    print("正在测试 OpenAI API 连通性...")
    try:
        models = client.models.list()
        model_count = len(models.data)
        print(f"✅ 连接成功！您的 API Key 有效。")
        print(f"   共获取到 {model_count} 个可用模型。")
        if model_count > 0:
            sample = [m.id for m in models.data[:5]]
            print(f"   示例模型：{', '.join(sample)}")
        return True
    except openai.AuthenticationError as e:
        print("❌ 认证失败：API Key 无效或已过期。")
        print(f"   详细信息：{e}")
        return False
    except openai.RateLimitError as e:
        print("❌ 速率限制：请求过于频繁，请稍后再试。")
        print(f"   详细信息：{e}")
        return False
    except openai.APIConnectionError as e:
        print("❌ 网络连接失败：无法访问 OpenAI API。")
        print("   请检查您的网络代理设置或防火墙。")
        print(f"   详细信息：{e}")
        return False
    except openai.APIStatusError as e:
        print(f"❌ API 返回错误状态码 {e.status_code}。")
        print(f"   详细信息：{e}")
        return False
    except openai.APIError as e:
        print(f"❌ OpenAI API 错误：{e}")
        return False
    except Exception as e:
        print(f"❌ 发生未预期的错误：{e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

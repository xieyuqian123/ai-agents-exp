import argparse
import os
import sys

from dotenv import load_dotenv

from common.available_tools import ToolExecutor
from common.search import serpapi_search_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="当前最新的苹果手机是什么")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--gl", default="cn")
    parser.add_argument("--hl", default="zh-cn")
    parser.add_argument("--dotenv", default=os.path.join(os.getcwd(), ".env"))
    args = parser.parse_args()

    dotenv_path = args.dotenv
    loaded = load_dotenv(dotenv_path=dotenv_path, override=True)
    print("dotenv_path:", dotenv_path)
    print("dotenv_exists:", os.path.exists(dotenv_path))
    print("dotenv_loaded:", bool(loaded))

    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_KEY")
    has_key = bool(serpapi_api_key or serpapi_key)
    print("SERPAPI_API_KEY_present:", bool(serpapi_api_key), "len:", len(serpapi_api_key or ""))
    print("SERPAPI_KEY_present:", bool(serpapi_key), "len:", len(serpapi_key or ""))
    if not has_key:
        print("错误: 未配置 SERPAPI_API_KEY（或 SERPAPI_KEY）。")
        print("请在 .env 中配置后再运行，例如：SERPAPI_API_KEY=xxxx")
        return 1

    executor = ToolExecutor({"search_web": serpapi_search_text})

    print("=== Tools ===")
    print(list(executor.list()))

    print("\n=== Simulated tool call ===")
    print(
        f'Action: search_web(query="{args.query}", gl="{args.gl}", hl="{args.hl}", limit={args.limit})'
    )

    try:
        result = executor.execute(
            "search_web",
            query=args.query,
            gl=args.gl,
            hl=args.hl,
            limit=args.limit,
        )
    except Exception as e:
        print("\n=== Tool error ===")
        print(type(e).__name__ + ":", str(e))
        return 2

    print("\n=== Tool result ===")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

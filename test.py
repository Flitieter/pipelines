#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

BASE_URL = "http://127.0.0.1:10086"   # 你的后端地址
MODEL = "alayajet"           # 替换成你 pipeline 的 id

def ask_stream(user_text: str):
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_text}],
        "stream": True,
    }
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                msg = json.loads(data)
                delta = msg["choices"][0]["delta"]
                piece = delta.get("content", "")
                if piece:
                    print(piece, end="", flush=True)
            except Exception:
                print(data, end="", flush=True)
        print()  # 最后换行

def ask_nonstream(user_text: str):
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_text}],
        "stream": False,
    }
    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print(content)

def main():
    print("=== Chat CLI (输入内容，回车发送；输入 'exit' 退出) ===")
    while True:
        try:
            user_text = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break

        if not user_text.strip():
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("再见!")
            break

        # 流式调用
        print("Assistant: ", end="", flush=True)
        ask_stream(user_text)
        # 如果你想要一次性返回，可以换成 ask_nonstream(user_text)

if __name__ == "__main__":
    main()

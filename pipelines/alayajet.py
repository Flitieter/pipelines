from typing import Union, Generator, Iterator, List
import requests

EXTERNAL_URL = "http://127.0.0.1:9875/chat"  # 你的外部 GET 流式接口

class Pipeline:
    def __init__(self):
        self.pipeline_id = "alayajet"

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        def gen() -> Generator[str, None, None]:
            # 根据你的接口需要自行添加 params/headers
            input_messages = messages.copy()
            input_messages.append({"role": "user", "content": user_message})
            payload = {
                "model": "infllm-vdb:latest",
                "messages": input_messages,
                "stream": True,
            }
            # 超时 (连接超时, 读取超时) 可按需调整
            with requests.post(EXTERNAL_URL, json=payload, stream=True, timeout=(5, 300)) as r:
                r.raise_for_status()

                # 1) 如果对方是“逐行”输出（推荐用于文本/类SSE但你不想自己加 data: 前缀）
                for line in r.iter_lines(decode_unicode=True):
                    if line is None or line == "":
                        continue
                    # 直接 yield 纯字符串，不做任何加工
                    print(line)
                    yield line

                # 若你的外部接口不是按行，而是原始字节块，可以改用 iter_content：
                # for chunk in r.iter_content(chunk_size=None):
                #     if not chunk:
                #         continue
                #     yield chunk.decode("utf-8", errors="ignore")

        # 返回真正的 generator 对象
        return gen()
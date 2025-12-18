from PIL import Image
from torchvision import transforms
import base64
import io
import requests
import numpy as np
import time
import anthropic
import httpx


class Claude3Agent:
    def __init__(self):
        import os
        self.prompt = "prompts.txt"
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.max_tokens = 50
        # self.temperature = self.cfg["temperature"]
        self.to_pil = transforms.ToPILImage()
        self.errors = {}
        self.responses = {}
        self.current_round = 0
        # self.resize = transforms.Resize((self.cfg["img_size"], self.cfg["img_size"]))
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def reset(self):
        self.errors = {}
        self.responses = {}
        self.current_round = 0

    def _request(self, obs, questions, debug_path=None):
        # context_messages = []
        pil_image = Image.fromarray(obs)
        image_bytes = io.BytesIO()

        pil_image.save(image_bytes, format="png")
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        chat_output = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=self.max_tokens,
            system=open(self.prompt).read(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": questions,
                        },
                    ],
                }
            ],
        )
        res = chat_output.content[0].text
        self.responses[self.current_round] = res
        return res, False

    # def _request_gpt4v(self, chat_input):
    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {self.api_key}",
    #     }
    #     response = requests.post(
    #         "https://api.openai.com/v1/chat/completions",
    #         headers=headers,
    #         json=chat_input,
    #     )
    #     json_res = response.json()
    #     print(f">>>>>> the original output from gpt4v is: {json_res} >>>>>>>>>")
    #     if "choices" in json_res:
    #         res = json_res["choices"][0]["message"]["content"]
    #     elif "error" in json_res:
    #         self.errors[self.current_round] = json_res
    #         res = "gpt4v API error"
    #         if json_res['error']['code'] == 'rate_limit_exceeded':
    #             time.sleep(10)
    #             return res, True
    #         else:
    #             raise RuntimeError

    #     # the prompt come with "Answer: " prefix
    #     self.responses[self.current_round] = res
    #     # return " ".join(res.split(" ")[1:])
    #     return res, False

    def ask(
        self,
        questions,
        obs,
        debug_path=None,
    ):
        if obs is None:
            return None
        self.current_round += 1
        retry = True
        while retry:
            ans, retry = self._request(obs, questions, debug_path=debug_path)
        ans = ans.lower().split(";")
        # if 'no' in ans:
        #     return False
        # return True
        return ans

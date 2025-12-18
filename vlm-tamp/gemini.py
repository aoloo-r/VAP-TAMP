from PIL import Image
from torchvision import transforms
import base64
import io
import requests
import numpy as np
import time
import httpx
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.generative_models import Image as gmImage


class GeminiAgent:
    def __init__(self):
        self.prompt = "prompts.txt"
        self.max_tokens = 50
        # self.temperature = self.cfg["temperature"]
        self.to_pil = transforms.ToPILImage()
        self.errors = {}
        self.responses = {}
        self.current_round = 0
        vertexai.init(project="plasma-centaur-420115")
        self.model = GenerativeModel("gemini-1.0-pro-vision")

    def reset(self):
        self.errors = {}
        self.responses = {}
        self.current_round = 0

    def _request(self, obs, questions):
        pil_image = Image.fromarray(obs)
        image_bytes = io.BytesIO()

        pil_image.save(image_bytes, format="png")
        # base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        image = gmImage.from_bytes(image_bytes.getvalue())
        text_input = open(self.prompt).read()
        text_input += "\nNow, answer the following questions: "
        text_input += questions
        response = self.model.generate_content([image, text_input])
        print(response.text)
        return response.text.strip(), False

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
            ans, retry = self._request(obs, questions)
        ans = ans.lower().split(";")
        return ans


GeminiAgent()

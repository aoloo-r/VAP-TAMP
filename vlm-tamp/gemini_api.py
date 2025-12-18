#!/usr/bin/env python3
"""
Gemini API wrapper compatible with DKPrompt's VLM agent interface.
Uses direct Gemini API instead of Vertex AI.
"""

import google.generativeai as genai
from PIL import Image
import numpy as np
from typing import List, Union
import io


class GeminiAPIAgent:
    """VLM agent using Gemini API directly (not Vertex AI)."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini API agent.

        Args:
            api_key: Gemini API key
            model_name: Model to use (gemini-1.5-flash, gemini-1.5-pro, etc.)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.errors = {}
        self.responses = {}
        self.current_round = 0

    def reset(self):
        """Reset agent state."""
        self.errors = {}
        self.responses = {}
        self.current_round = 0

    def ask(self, questions: Union[str, List[str]], obs: np.ndarray) -> List[str]:
        """
        Ask VLM questions about an observation.

        Args:
            questions: Single question string or semicolon-separated questions
            obs: RGB image as numpy array (H, W, 3)

        Returns:
            List of answers (one per question)
        """
        # Parse questions
        if isinstance(questions, str):
            if ";" in questions:
                question_list = [q.strip() for q in questions.split(";")]
            else:
                question_list = [questions]
        else:
            question_list = questions

        # Convert observation to PIL Image
        if isinstance(obs, np.ndarray):
            pil_image = Image.fromarray(obs.astype(np.uint8))
        else:
            pil_image = obs

        # Query each question
        answers = []
        for question in question_list:
            try:
                # Add instruction for binary yes/no answers
                prompt = f"{question}\n\nPlease answer with 'yes', 'no', or 'uncertain' if you cannot determine from this view."

                response = self.model.generate_content([prompt, pil_image])
                answer = response.text.strip().lower()

                # Store response
                self.responses[self.current_round] = answer
                answers.append(answer)

            except Exception as e:
                error_msg = f"Error querying VLM: {e}"
                self.errors[self.current_round] = error_msg
                answers.append("uncertain")
                print(f"⚠️  {error_msg}")

        self.current_round += 1
        return answers

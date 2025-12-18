# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import base64
import io
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)


class MultiCropGeminiClient:
    """Gemini API client for multi-crop VLM planning with higher image context limits."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.prompt = self.cfg["prompt"]
        self.api_key = self.cfg["api_key"]
        self.max_tokens = self.cfg.get("max_tokens", 1000)
        self.temperature = self.cfg.get("temperature", 0.2)
        self.model = self.cfg.get("model", "gemini-2.0-flash-exp")  # Default to Gemini 2.0 Flash
        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize((self.cfg["img_size"], self.cfg["img_size"]))

    def reset(self):
        self.errors = {}
        self.responses = {}
        self.current_round = 0
        self.goal = None

    def _image_to_base64(self, pil_image):
        """Convert PIL image to base64 string."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _prepare_samples(self, obs, goal, debug_path=None):
        """Prepare samples for Gemini API."""
        self.goal = goal
        
        # Build the content list for Gemini
        contents = []
        
        # Add text prompt
        contents.append({"text": self.prompt})
        
        # Add images
        for img_id, object_image in enumerate(obs.object_images):
            idx = object_image.crop_id
            pil_image = self.resize(Image.fromarray(np.array(object_image.image, dtype=np.uint8)))
            
            # Display for debugging
            plt.subplot(1, len(obs.object_images), img_id + 1)
            plt.imshow(pil_image)
            plt.axis("off")
            plt.title(f"img_{idx}")
            
            # Convert to base64 for Gemini
            base64_image = self._image_to_base64(pil_image)
            
            # Add image content (Gemini format)
            contents.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64_image
                }
            })
            
            # Add image label
            contents.append({
                "text": f"\nThis is img_{idx}.\n"
            })

        # Build scene graph text
        scene_graph_text = "\n2. Scene descriptions: "
        if obs.scene_graph:
            # Create mapping from instance_id to display name (crop_id or "other_N")
            instance_to_name = {}
            other_counter = 0
            
            # First, map all selected crops
            for obj_crop in obs.object_images:
                instance_to_name[obj_crop.instance_id] = f"img_{obj_crop.crop_id}"
            
            # Then, map unselected instances that appear in scene graph
            for rel in obs.scene_graph:
                for instance_id in [rel[0], rel[1]]:
                    if instance_id not in instance_to_name:
                        instance_to_name[instance_id] = f"other_{other_counter}"
                        other_counter += 1
            
            # Build scene graph text with proper names
            for rel in obs.scene_graph:
                name_a = instance_to_name.get(rel[0], "unknown")
                name_b = instance_to_name.get(rel[1], "unknown")
                scene_graph_text += f"{name_a} is {rel[2]} {name_b}; "
        
        contents.append({"text": scene_graph_text + "\n"})
        
        # DEBUG: Print what scene descriptions are being sent to VLM
        print("\n📝 SCENE DESCRIPTIONS SENT TO VLM:")
        print("-" * 40)
        print(scene_graph_text)
        print("-" * 40)

        plt.suptitle(f"Prompts that are automatically generated for task: {self.goal}")
        plt.show()

        # Add final query and answer prompt
        contents.append({"text": f"3. Query: {self.goal}\n"})
        contents.append({"text": "4. Answer: "})
        
        # Format for Gemini API
        request_data = {
            "contents": [{
                "parts": contents
            }],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        }
        
        return request_data

    def act_on_observations(self, obs, goal=None, debug_path=None):
        """Main entry point for getting VLM response."""
        if not obs:
            raise RuntimeError("no object-centric visual observations!")
        self.current_round += 1
        request_data = self._prepare_samples(obs, goal, debug_path=debug_path)
        return self._request(request_data)

    def _request(self, request_data):
        """Make request to Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        
        try:
            response = requests.post(url, headers=headers, json=request_data)
            response.raise_for_status()
            
            json_res = response.json()
            
            if "candidates" in json_res and len(json_res["candidates"]) > 0:
                candidate = json_res["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    # Extract text from the first part
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            # Fallback if response structure is unexpected
            print("Warning: Unexpected Gemini response structure")
            print("Response:", json_res)
            return "explore"  # Default fallback
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            return "explore"  # Default fallback
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "explore"  # Default fallback
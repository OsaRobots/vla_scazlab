"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy
import torch.nn.functional as F
json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray, "entropy": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            # action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            action, entropy, probs, act_range = self.predict_with_probs(**inputs, unnorm_key=unnorm_key, do_sample=False)

            response = {"action": action, "entropy": entropy, "probs": probs, "range":act_range}
            if double_encode:
                return JSONResponse(json_numpy.dumps(response))
            else:
                return JSONResponse(response)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return JSONResponse({"error": "An error occurred"}, status_code=500)
        
    def predict_with_probs(self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # Run VLA inference
        gen_dict = self.vla.generate(input_ids, max_new_tokens=self.vla.get_action_dim(unnorm_key), return_dict_in_generate=True, output_logits=True, **kwargs)
        logits = gen_dict.logits
        entropy, probs = self.calculate_entropy(logits)
        generated_ids = gen_dict.sequences
        
        
        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.vla.get_action_dim(unnorm_key) :].cpu().numpy()
        #ulas
        # print(predicted_action_token_ids)
        discretized_actions = self.vla.vocab_size - predicted_action_token_ids
        # print(discretized_actions)
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.vla.bin_centers.shape[0] - 1)
        # print(discretized_actions)
        normalized_actions = self.vla.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.vla.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        act_range = [0.5 * (self.vla.bins + 1) * (action_high[i] - action_low[i]) + action_low[i] for i in range(6)]

        return actions, entropy, probs, act_range

    def calculate_entropy(self,logits_list):
        # Calculate entropy for each logits tensor
        entropy_list = []
        prob_list = []
        for logits in logits_list:
            # Apply softmax to get probabilities
            action_logits = logits[:,31744:32000]
            probabilities = F.softmax(action_logits, dim=-1)
            # print(probabilities.shape)
            
            
            # Calculate entropy
            entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
            
            # Store entropy for each token
            entropy_list.append(entropy.cpu().numpy()[0])

            # Store probability distribution for each token
            p = probabilities.cpu().numpy()[0]
            # print(256-np.argmax(p))
            prob_list.append(p)

        return entropy_list, prob_list

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "/home/uk/VLAs/fine-tuned/openvla-7b+vla_dataset+b16+lr-0.0005+lora-r32+dropout-0.0" #"openvla/openvla-7b"               # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg.openvla_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
import os
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
import numpy as np
import gradio as gr
from dataclasses import dataclass
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse
# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str,
                                                             Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:

    def __init__(self,
                 openvla_path: Union[str, Path],
                 attn_implementation: Optional[str] = "flash_attention_2"):
        self.openvla_path = openvla_path
        self.attn_implementation = attn_implementation
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path,
                                                       trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

    def predict_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict an action based on an input JSON payload.
        Payload format:
            {
                "image": np.ndarray,  # Image as a NumPy array
                "instruction": str    # Instruction string
            }
        """
        try:
            # Parse JSON payload
            image = payload["image"]
            instruction = payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)
            image = np.array(image).astype(np.uint8)

            if not isinstance(image, np.ndarray):
                return {"error": "The 'image' field must be a NumPy array."}
            if not isinstance(instruction, str):
                return {"error": "The 'instruction' field must be a string."}

            # Convert the NumPy array to a PIL Image

            image = Image.fromarray(image).convert("RGB")

            # Generate the prompt and run inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(prompt, image).to(self.device,
                                                      dtype=torch.bfloat16)
            action = self.vla.predict_action(**inputs,
                                             unnorm_key=unnorm_key,
                                             do_sample=False)

            return {"action": action}
        except Exception as e:
            logging.error(traceback.format_exc())
            return {"error": str(e)}


@dataclass
class DeployConfig:
    openvla_path: Union[
        str, Path] = "openvla/openvla-7b"  # HF Hub Path or local model path


def deploy(cfg: DeployConfig):
    server = OpenVLAServer(cfg.openvla_path)

    def predict_action(payload: Dict[str, Any]) -> Dict[str, Any]:
        return server.predict_action(payload)

    # Define Gradio app with JSON input and output
    app = gr.Interface(
        fn=predict_action,
        inputs=gr.JSON(
            label=
            "Input JSON (e.g., {'image': <NumPy array>, 'instruction': 'Your instruction'})"
        ),
        outputs=gr.JSON(
            label="Output JSON (e.g., {'action': 'Predicted action'})"),
        title="OpenVLA JSON Interface",
        description=
        "Provide an input JSON with 'image' (NumPy array) and 'instruction' to get a predicted action.",
    )

    # Launch the Gradio app
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Set the openvla_path for your script.")

    # Add the argument for openvla_path
    parser.add_argument(
        "--openvla_path",
        type=str,
        default=
        "openvla/openvla-7b",  # Set to True if this argument is mandatory
        help="Path to the OpenVLA directory or executable.")

    args = parser.parse_args()
    cfg = DeployConfig()
    cfg.openvla_path = args.openvla_path

    deploy(cfg)

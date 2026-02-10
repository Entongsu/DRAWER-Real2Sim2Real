import os
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Union

import torch
from dataclasses import dataclass
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import gradio as gr
from typing import Any, Dict, Optional, Union
import numpy as np
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
        """
        A simple server for OpenVLA models; exposes a Gradio interface for action prediction.
        """
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

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(
                    Path(self.openvla_path) / "dataset_statistics.json",
                    "r") as f:
                self.vla.norm_stats = json.load(f)

    def predict_action(self, image: Image.Image, instruction: str) -> str:
        """
        Predicts an action based on an input image and instruction.
        """
        try:
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            image = np.array(image)
            inputs = self.processor(prompt, image.convert("RGB")).to(
                self.device, dtype=torch.bfloat16)
            action = self.vla.predict_action(**inputs, do_sample=False)
            return json.dumps(action)  # Return as a JSON string
        except Exception as e:
            logging.error(traceback.format_exc())
            return f"Error: {str(e)}"


@dataclass
class DeployConfig:
    openvla_path: Union[
        str, Path] = "openvla/openvla-7b"  # HF Hub Path or local model path


def deploy(cfg: DeployConfig):
    server = OpenVLAServer(cfg.openvla_path)

    # Define Gradio interface
    def inference(image, instruction):
        return server.predict_action(image, instruction)

    interface = gr.Interface(
        fn=inference,
        inputs=[
            gr.inputs.Image(type="pil", label="Input Image"),
            gr.inputs.Textbox(lines=2,
                              placeholder="Enter instruction here",
                              label="Instruction"),
        ],
        outputs=gr.outputs.Textbox(label="Predicted Action"),
        title="OpenVLA Model",
        description=
        "Provide an image and instruction to get a predicted action.",
    )

    # Launch the Gradio interface with a public link
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    cfg = DeployConfig()
    deploy(cfg)

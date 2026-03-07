import os
import math
import tempfile
import numpy as np
import torch
from PIL import Image, ImageOps

from workflow.looper_workflow import WorkflowEngine
from workflow.api_workflow_builder import BUILDER_MAP, SAVE_PREFIX
from utils.json_spec import LoopSettings
from utils.comfyui_client import ComfyUIClient
from utils.util import resize_image_match_area

# Model-specific constants
MODEL_CONFIGS = {
    "sdxl": {"area": 1024**2, "latent_reduction": 8},
    "flux1d": {"area": 1024**2, "latent_reduction": 8},
    "sd3.5": {"area": 1024**2, "latent_reduction": 8},
}

# Default settings per model type (mirrors the old engine DEFAULT_SETTING_DICTs)
MODEL_DEFAULTS = {
    "sdxl": {
        "checkpoint": "sdXL_v10VAEFix.safetensors",
        "cfg": 8.0,
        "denoise_steps": 20,
        "neg_prompt": "text, watermark, logo",
    },
    "flux1d": {
        "clip": ["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors"],
        "denoise_steps": 20,
    },
    "sd3.5": {
        "checkpoint": "sd3.5_large.safetensors",
        "clip": ["t5xxl_fp8_e4m3fn.safetensors", "clip_g.safetensors"],
        "cfg": 3.5,
        "denoise_steps": 20,
        "neg_prompt": "text, watermark, logo",
    },
}


class APIWorkflowEngine(WorkflowEngine):
    """Workflow engine that submits jobs to a ComfyUI server via HTTP API."""

    def __init__(self, model_type: str, client: ComfyUIClient):
        self.model_type = model_type
        self.client = client
        self.builder_fn = BUILDER_MAP[model_type]
        self.config = MODEL_CONFIGS[model_type]

        self.NAME = model_type
        self.DEFAULT_SETTING_DICT = MODEL_DEFAULTS.get(model_type, {})

    def resize_images_for_model(self, input_path: str, output_paths: list[str]):
        area = self.config["area"]
        modulo = self.config["latent_reduction"]
        for output_path in output_paths:
            resize_image_match_area(input_path, output_path, area, modulo)

    def setup(self):
        self.client.check_server()

    def compute_iteration(self, image_tensor: torch.Tensor, loopsettings: LoopSettings) -> torch.Tensor:
        # 1. Save tensor to temp PNG
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "looper_upload.png")
        i = 255.0 * image_tensor[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(temp_path, compress_level=0)

        # 2. Upload to ComfyUI server
        server_filename = self.client.upload_image(temp_path)

        # 3. Build workflow JSON
        workflow = self.builder_fn(server_filename, loopsettings, self.DEFAULT_SETTING_DICT)

        # 4. Execute workflow
        history = self.client.execute_workflow(workflow)

        # 5. Find output image in history
        output_images = self.client.get_output_images(history)
        if not output_images:
            raise RuntimeError("ComfyUI returned no output images")

        # 6. Download output image
        out_info = output_images[0]
        result_path = os.path.join(temp_dir, "looper_result.png")
        self.client.download_image(
            filename=out_info["filename"],
            subfolder=out_info.get("subfolder", ""),
            image_type=out_info.get("type", "output"),
            save_path=result_path,
        )

        # 7. Load as tensor and return (matching ComfyUI's [B, H, W, C] float32 format)
        result_img = Image.open(result_path)
        result_img = ImageOps.exif_transpose(result_img)
        result_img = result_img.convert("RGB")
        result_array = np.array(result_img).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_array).unsqueeze(0)

        # Clean up temp files
        for p in [temp_path, result_path]:
            if os.path.exists(p):
                os.remove(p)

        return result_tensor

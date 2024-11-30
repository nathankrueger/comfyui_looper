import os
import torch
import tqdm
from typing import IO
from PIL.PngImagePlugin import PngInfo

from image_processing.animator import make_animation
from utils.json_spec import SettingsManager, default_seed, LoopSettings
from image_processing.transforms import load_image_with_transforms
from utils.util import (
    import_custom_nodes,
    add_comfyui_directory_to_sys_path,
    save_tensor_to_images,
    get_loop_img_filename,
)

# ComfyUI imports
add_comfyui_directory_to_sys_path()
import_custom_nodes()

class WorkflowEngine:
    """
    Base class that can be overridden for different models
    """
    
    NAME = None

    @classmethod
    def get_name(cls):
        return cls.NAME
    
    def resize_images_for_model(self, input_path: str, output_paths: list[str]):
        """
        Used to appropriately resize images for a given model
        """
        
        # override me
        pass

    def setup(self):
        """
        Used for any 1-time setup
        """

        # override me
        return None

    def compute_iteration(self, image_tensor: torch.Tensor, loopsettings: LoopSettings) -> torch.Tensor:
        """
        Computes the next image based on the loopsettings, and the current image
        """

        # override me
        return None

def looper_main(
    engine: WorkflowEngine,
    loop_img_path: str,
    output_folder: str,
    json_file: str,
    animation_file: str | None,
    animation_type: str,
    animation_params: dict[str, str],
    log_file: IO[str]
):
    sm = SettingsManager(json_file)
    sm.validate()

    with torch.inference_mode():
        engine.setup()
        prev_seed = None
        total_iter = sm.get_total_iterations()
        for iter in tqdm.tqdm(range(total_iter)):
            print()
            
            # load settings from JSON
            loopsettings = sm.get_elaborated_loopsettings_for_iter(iter)
            transforms = loopsettings.transforms
            seed = loopsettings.seed

            # if a new seed is explicitly set, use it, otherwise always get a new one
            if seed == prev_seed:
                seed = default_seed()
                loopsettings.seed = seed
            sm.update_seed(iter, seed)
            prev_seed = seed

            # load in image & resize it
            image_tensor = load_image_with_transforms(
                image_path=loop_img_path,
                transforms=transforms,
                iter=iter,
                offset=loopsettings.offset,
                total_iter=total_iter
            )

            # generate the image in a workflow specific manner
            vae_decode_result = engine.compute_iteration(image_tensor, loopsettings)

            # save the images -- loop filename, and requested output
            loopsettings_json = loopsettings.to_json(indent=4)
            output_image_filename = os.path.join(output_folder, get_loop_img_filename(iter+1))
            pnginfo = PngInfo()
            pnginfo.add_text(key='looper_settings', value=loopsettings_json, zip=False)
            save_tensor_to_images(
                output_filenames=[loop_img_path, output_image_filename],
                image=vae_decode_result[0],
                png_info=pnginfo
            )

            # add entry to the logfile
            log_file.write(f"{output_image_filename}:" + loopsettings_json + os.linesep)

    # save animation
    if animation_file is not None:
        make_animation(
            type=animation_type,
            input_folder=output_folder,
            output_animation=os.path.join(output_folder, animation_file),
            params=animation_params
        )

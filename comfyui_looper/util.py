import sys
import os
import shutil
import math
from datetime import datetime
from typing import Sequence, Mapping, Any, Union
from PIL import Image, ImageOps
import numpy as np
import torch

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()

def load_image(image_path: str):
    i = Image.open(image_path)

    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    return (image, mask.unsqueeze(0))

def save_tensor_to_images(image, output_filenames: list[str]):
    first_image_path = None
    for output_filename in output_filenames:
        output_folder = os.path.dirname(output_filename)
        os.makedirs(output_folder, exist_ok=True)
        if first_image_path is None:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(output_filename, pnginfo=None, compress_level=0)
            first_image_path = output_filename
        else:
            shutil.copy(first_image_path, output_filename)

def get_loop_img_filename(idx: int) -> str:
    return f"loop_img_{idx:06}.png"

def get_log_filename(log_basename: str) -> str:
    dt = datetime.now()
    dt_str = dt.strftime("%Y_%m_%d__%H_%M_%S")

    log_ext = log_basename.split('.')[-1] if '.' in log_basename else ""
    log_basename_no_ext = '.'.join(log_basename.split('.')[:-1])

    return f"{log_basename_no_ext}_{dt_str}.{log_ext}"

def all_subclasses(cls) -> set:
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])

def resize_image(input_path: str, output_path: str, max_dim: int):
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)
    width, height = img.size
    
    if width > height:
        ratio = max_dim / width
        img = img.resize((max_dim, int(height * ratio)))
    else:
        ratio = max_dim / height
        img = img.resize((int(width * ratio), max_dim))

    img.save(output_path, pnginfo=None, compress_level=0)

def resize_image_match_area(input_path: str, output_path: str, area: int, modulo: int | None):
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)
    width, height = img.size

    w2h_ratio = float(width) / float(height)
    new_h = int(math.sqrt(float(area) / w2h_ratio))
    new_w = int(float(area) / float(new_h))

    if modulo is not None:
        new_h = new_h - (new_h % modulo)
        new_w = new_w - (new_w % modulo)
    
    img = img.resize((new_w, new_h))
    img.save(output_path, pnginfo=None, compress_level=0)

def get_torch_device_vram_used_gb() -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_used = torch.cuda.memory_allocated(device) / (1024 ** 3)
    return vram_used

import math
import ast
import operator as op

# adapted from solution: https://stackoverflow.com/questions/43836866/safely-evaluate-simple-string-equation
class MathParser:
    """ Basic parser with local variable and math functions 
    
    Args:
       vars (mapping): mapping object where obj[name] -> numerical value 
       math (bool, optional): if True (default) all math function are added in the same name space
       
    Example:
       
       data = {'r': 3.4, 'theta': 3.141592653589793}
       parser = MathParser(data)
       assert parser.parse('r*cos(theta)') == -3.4
       data['theta'] =0.0
       assert parser.parse('r*cos(theta)') == 3.4
    """
        
    _operators2method = {
        ast.Add: op.add, 
        ast.Sub: op.sub, 
        ast.BitXor: op.xor, 
        ast.Or:  op.or_, 
        ast.And: op.and_, 
        ast.Mod:  op.mod,
        ast.Mult: op.mul,
        ast.Div:  op.truediv,
        ast.Pow:  op.pow,
        ast.FloorDiv: op.floordiv,              
        ast.USub: op.neg, 
        ast.UAdd: lambda a:a  
    }
    
    def __init__(self, vars, math=True):
        self._vars = vars
        if not math:
            self._alt_name = self._no_alt_name
        
    def _Name(self, name):
        try:
            return  self._vars[name]
        except KeyError:
            return self._alt_name(name)

    @staticmethod
    def _alt_name(name):
        if name.startswith("_"):
            raise NameError(f"{name!r}") 
        try:
            return  getattr(math, name)
        except AttributeError:
            raise NameError(f"{name!r}") 

    @staticmethod
    def _no_alt_name(name):
        raise NameError(f"{name!r}") 

    def eval_(self, node):
        if isinstance(node, ast.Expression):
            return self.eval_(node.body)
        if isinstance(node, ast.Num): # <number>
            return node.n
        if isinstance(node, ast.Name):
            return self._Name(node.id) 
        if isinstance(node, ast.BinOp):            
            method = self._operators2method[type(node.op)]                      
            return method( self.eval_(node.left), self.eval_(node.right) )            
        if isinstance(node, ast.UnaryOp):             
            method = self._operators2method[type(node.op)]  
            return method( self.eval_(node.operand) )
        if isinstance(node, ast.Attribute):
            return getattr(self.eval_(node.value), node.attr)

        if isinstance(node, ast.Call):            
            return self.eval_(node.func)( 
                      *(self.eval_(a) for a in node.args),
                      **{k.arg:self.eval_(k.value) for k in node.keywords}
                     )
        else:
            raise TypeError(node)

    def parse(self, expr):
        return self.eval_(ast.parse(expr, mode='eval'))
    
    def __call__(self, expr):
        return self.parse(expr)
    
def parse_params(params_list: list[str]) -> dict[str, str]:
    params = {}
    if params_list is not None:
        for param in params_list:
            param_key = param.split(':')[0]
            param_val = param.split(':')[1] if ':' in param else None
            params[param_key] = param_val

    return params
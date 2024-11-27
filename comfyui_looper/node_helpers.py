from nodes import (
    LoraLoader,
    CheckpointLoaderSimple,
    NODE_CLASS_MAPPINGS,
)

class LoraManager:
    def __init__(self):
        self.prev_hash = None
        self.loraloader = LoraLoader()
        self.prev_model = None
        self.prev_clip = None

    def reload_if_needed(self, lora_list: list[tuple[str, float]], model, clip) -> tuple:
        """
        Check the state of the input lora_dict against the current
        hash, if changes are detected, load in new weights, otherwise
        do nothing.
        """

        current_hash = hash(str(lora_list))
        if current_hash != self.prev_hash:
            curr_model = model
            curr_clip = clip
    
            for lora_name, lora_strength in lora_list:
                curr_model, curr_clip = self.loraloader.load_lora(
                    lora_name=lora_name,
                    strength_model=lora_strength,
                    strength_clip=lora_strength,
                    model=curr_model,
                    clip=curr_clip,
                )

            self.prev_hash = current_hash
            self.prev_model = curr_model
            self.prev_clip = curr_clip

        return self.prev_model, self.prev_clip
    
class CheckpointManager:
    def __init__(self):
        self.prev_ckpt: str = None
        self.ckptloader = CheckpointLoaderSimple()
        self.prev_ckpt_model = None
        self.prev_ckpt_clip = None
        self.prev_ckpt_vae = None

    def reload_if_needed(self, ckpt: str) -> tuple:
        if ckpt != self.prev_ckpt:
            ckpt_model, ckpt_clip, ckpt_vae = self.ckptloader.load_checkpoint(
                ckpt_name="sdXL_v10VAEFix.safetensors"
            )

            self.prev_ckpt = ckpt
            self.prev_ckpt_model = ckpt_model
            self.prev_ckpt_clip = ckpt_clip
            self.prev_ckpt_vae = ckpt_vae
        
        return self.prev_ckpt_model, self.prev_ckpt_clip, self.prev_ckpt_vae


class SDXLClipEncodeWrapper:
    def __init__(self):
        self.node = NODE_CLASS_MAPPINGS["CLIPTextEncodeSDXL"]()

    def encode(self, w: int, h: int, pos_text: str, neg_text: str, clip):
        positive_conditioning, = self.node.encode(
            width=w,
            height=h,
            crop_w=0,
            crop_h=0,
            target_width=w,
            target_height=h,
            text_g=pos_text,
            text_l=pos_text,
            clip=clip
        )
        negative_conditioning, = self.node.encode(
            width=w,
            height=h,
            crop_w=0,
            crop_h=0,
            target_width=w,
            target_height=h,
            text_g=neg_text,
            text_l=neg_text,
            clip=clip
        )

        return positive_conditioning, negative_conditioning
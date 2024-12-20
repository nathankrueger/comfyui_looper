from nodes import (
    LoraLoader,
    UNETLoader,
    DualCLIPLoader,
    VAELoader,
    CheckpointLoaderSimple,
    CLIPTextEncode,
    ControlNetLoader,
    NODE_CLASS_MAPPINGS,
)

from utils.json_spec import ConDelta, LoraFilter

class LoraManager:
    def __init__(self):
        self.prev_hash = None
        self.loraloader = LoraLoader()
        self.prev_model = None
        self.prev_clip = None

    def reload_if_needed(self, lora_list: list[LoraFilter], model, clip) -> tuple:
        """
        Check the state of the input lora_dict against the current
        hash, if changes are detected, load in new weights, otherwise
        do nothing.
        """

        current_hash = hash(tuple(lora_list))
        if current_hash != self.prev_hash:
            curr_model = model
            curr_clip = clip
    
            for lorafilter in lora_list:
                curr_model, curr_clip = self.loraloader.load_lora(
                    lora_name=lorafilter.lora_path,
                    strength_model=lorafilter.lora_strength,
                    strength_clip=lorafilter.lora_strength,
                    model=curr_model,
                    clip=curr_clip,
                )

            self.prev_hash = current_hash
            self.prev_model = curr_model
            self.prev_clip = curr_clip

        return self.prev_model, self.prev_clip
    
class SDXLCheckpointManager:
    def __init__(self):
        self.prev_ckpt_filename: str = None
        self.ckptloader = CheckpointLoaderSimple()
        self.prev_ckpt_model = None
        self.prev_ckpt_clip = None
        self.prev_ckpt_vae = None

    def reload_if_needed(self, ckpt: str) -> tuple:
        if ckpt != self.prev_ckpt_filename:
            self.prev_ckpt_model, self.prev_ckpt_clip, self.prev_ckpt_vae = self.ckptloader.load_checkpoint(ckpt_name=ckpt)
            self.prev_ckpt_filename = ckpt
        
        return self.prev_ckpt_model, self.prev_ckpt_clip, self.prev_ckpt_vae

class Flux1DModelManager:
    def __init__(self):
        self.unet_loader = UNETLoader()
        self.dual_clip_loader = DualCLIPLoader()
        self.vae_loader = VAELoader()

        # input strings
        self.prev_unet_filename: str = None
        self.prev_vae_filename: str = None
        self.prev_clip1_filename: str = None
        self.prev_clip2_filename: str = None

        # output model objects
        self.prev_unet = None
        self.prev_clip = None
        self.prev_vae = None

    def reload_if_needed(self, unet: str, vae: str, clip1: str, clip2: str) -> tuple:
        # unet reload
        if unet != self.prev_unet_filename:
            self.prev_unet, = self.unet_loader.load_unet(
                unet_name=unet,
                weight_dtype="default"
            )
            self.prev_unet_filename = unet
        
        # vae reload
        if vae != self.prev_vae_filename:
            self.prev_vae, = self.vae_loader.load_vae(vae)
            self.prev_vae_filename = vae

        # clip reload
        if clip1 != self.prev_clip1_filename or clip2 != self.prev_clip2_filename:
            self.prev_clip, = self.dual_clip_loader.load_clip(
                clip_name1=clip1,
                clip_name2=clip2,
                type="flux"
            )
            self.prev_clip1_filename = clip1
            self.prev_clip2_filename = clip2
        
        return self.prev_unet, self.prev_clip, self.prev_vae

class SD3p5ModelManager:
    def __init__(self):
        self.dual_clip_loader = DualCLIPLoader()
        self.ckptloader = CheckpointLoaderSimple()

        # input strings
        self.prev_model_checkpoint_filename: str = None
        self.prev_clip1_filename: str = None
        self.prev_clip2_filename: str = None

        # output model objects
        self.prev_model = None
        self.prev_clip = None
        self.prev_vae = None

    def reload_if_needed(self, ckpt: str, clip1: str, clip2: str) -> tuple:
        # unet reload
        if ckpt != self.prev_model_checkpoint_filename:
            self.prev_model, _, self.prev_vae = self.ckptloader.load_checkpoint(ckpt_name=ckpt)
            self.prev_model_checkpoint_filename = ckpt
        
        # clip reload
        if clip1 != self.prev_clip1_filename or clip2 != self.prev_clip2_filename:
            self.prev_clip, = self.dual_clip_loader.load_clip(
                clip_name1=clip1,
                clip_name2=clip2,
                type="sd3"
            )
            self.prev_clip1_filename = clip1
            self.prev_clip2_filename = clip2
        
        return self.prev_model, self.prev_clip, self.prev_vae


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
    
class ClipEncodeWrapper:
    def __init__(self):
        self.node = CLIPTextEncode()

    def encode(self, pos_text: str, neg_text: str, clip):
        positive_conditioning, = self.node.encode(clip=clip, text=pos_text)
        negative_conditioning, = self.node.encode(clip=clip, text=neg_text)
        return positive_conditioning, negative_conditioning
    
class ControlNetManager:
    def __init__(self, controlnet_model_file: str):
        self.node = ControlNetLoader()
        self.controlnet_model = None
        self.controlnet_model_file = controlnet_model_file

    def reload_if_needed(self, canny_param):
        if canny_param is not None:
            if self.controlnet_model is None:
                self.controlnet_model = self.node.load_controlnet(self.controlnet_model_file)[0]
        else:
            self.controlnet_model = None

        return self.controlnet_model

class ConDeltaNodeWrapper:
    def __init__(self):
        self.cond_subtract_node = NODE_CLASS_MAPPINGS["ConditioningSubtract"]()
        self.cond_apply_node = NODE_CLASS_MAPPINGS["ConditioningAddConDelta"]()
        self.delta_text_node_a = CLIPTextEncode()
        self.delta_text_node_b = CLIPTextEncode()

class ConDeltaManager:
    def __init__(self):
        self.clip_enconder = ClipEncodeWrapper()
        self.cd_node_wrappers: list[ConDeltaNodeWrapper] = []
        self.prev_num_con_deltas: int = -1

    def encode(self, clip, pos_text: str, neg_text: str, con_deltas: list[ConDelta]):
        pos_cond, neg_cond = self.clip_enconder.encode(pos_text=pos_text, neg_text=neg_text, clip=clip)
        
        if len(con_deltas) != self.prev_num_con_deltas:
            self.prev_num_con_deltas = len(con_deltas)
            self.cd_node_wrappers.clear()

            for cd in con_deltas:
                cdnw = ConDeltaNodeWrapper()
                cond_a, = cdnw.delta_text_node_a.encode(clip=clip, text=cd.pos)
                cond_b, = cdnw.delta_text_node_b.encode(clip=clip, text=cd.neg)
                delta_con, = cdnw.cond_subtract_node.subtract(
                    conditioning_a=cond_a,
                    conditioning_b=cond_b
                )
                pos_cond, = cdnw.cond_apply_node.addDelta(
                    conditioning_delta_strength=cd.strength,
                    conditioning_base=pos_cond,
                    conditioning_delta=delta_con,
                )
                self.cd_node_wrappers.append(cdnw)
        else:
            for cdnw, cd in zip(self.cd_node_wrappers, con_deltas):
                cond_a, = cdnw.delta_text_node_a.encode(clip=clip, text=cd.pos)
                cond_b, = cdnw.delta_text_node_b.encode(clip=clip, text=cd.neg)
                delta_con, = cdnw.cond_subtract_node.subtract(
                    conditioning_a=cond_a,
                    conditioning_b=cond_b
                )
                pos_cond, = cdnw.cond_apply_node.addDelta(
                    conditioning_delta_strength=cd.strength,
                    conditioning_base=pos_cond,
                    conditioning_delta=delta_con,
                )

        return pos_cond, neg_cond

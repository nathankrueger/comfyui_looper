from nodes import LoraLoader

class LoraManager:
    def __init__(self):
        self.prev_hash = None
        self.loraloader = LoraLoader()
        self.prev_model = None
        self.prev_clip = None

    def reload_if_needed(self, lora_list: list[tuple[str, float]], model, clip):
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
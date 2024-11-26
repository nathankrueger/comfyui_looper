from comfyui_looper.json_spec import *

def get_loop_settings():
    return LoopSettings(
        loop_iterations=20,
        checkpoint='foo.safetensors',
        prompt='a beautfiul photo, one with commas',
        denoise_steps=30,
        denoise_amt=0.5,
        loras=[('lora_a.safetensors', 1.0), ('lora_b.safetensors', .212354)],
        transforms=[{'name':'zoom_in', 'zoom_amt':0.985}]
    )

def test_serdes_loopsettings():
    test = get_loop_settings()
    test_json = test.to_json(indent=4)
    test_clone = LoopSettings.schema().loads(test_json)
    assert test == test_clone

def test_serdes_workflow():
    ls1 = get_loop_settings()
    ls2 = get_loop_settings()
    ls2.checkpoint="bar"
    ls2.prompt="abc"

    test_workflow = Workflow(all_settings=[ls1, ls2], version=1)
    test_json = test_workflow.to_json(indent=4)
    test_clone = Workflow.schema().loads(test_json)
    assert test_workflow == test_clone
    # with open('test.json', 'w', encoding='utf-8') as f:
    #     f.write(test_json)

def test_settings_manager():
    pass
import tempfile
import pytest

# workaround for running in debugger -- it picks up pytest.ini this way
if __name__ == '__main__':
    pytest.main(['-s'])

from comfyui_looper.utils.json_spec import *

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
    w = Workflow(
        [
            # 0
            LoopSettings(
                seed=123,
                loop_iterations=1,
                canny=[1.0,1.0,1.0]
            ),
            # 1
            LoopSettings(
                transforms=[{'name':'zoom_in','zoom_amt':0.5}],
                loras=[('foo', 0.5)],
            ),
            # 2
            LoopSettings(
                seed=456,
                canny=[]
            ),
            # 3
            LoopSettings(),
            # 4
            LoopSettings(
                transforms=[],
                loras=[]
            ),
        ] + [LoopSettings()] * 5
    )

    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="utf-8")
        temp_file.write(w.to_json(indent=4))
        temp_file.flush()
        sm = SettingsManager(temp_file.name)

        # canny
        assert w.all_settings[0].canny == [1.0,1.0,1.0]
        assert sm.get_setting_for_iter('canny', 0) == [1.0,1.0,1.0]
        assert w.all_settings[1].canny == [None]
        assert sm.get_setting_for_iter('canny', 1) == [1.0,1.0,1.0]
        assert w.all_settings[2].canny == []
        assert sm.get_setting_for_iter('canny', 2) == []
        assert w.all_settings[3].canny == [None]
        assert sm.get_setting_for_iter('canny', 3) == []
        assert w.all_settings[4].canny == [None]
        assert sm.get_setting_for_iter('canny', 4) == []
        for i in range(5,10):
            assert w.all_settings[i].canny == [None]
            assert sm.get_setting_for_iter('canny', i) == []

        # seed
        assert sm.get_setting_for_iter('seed', 0) != sm.get_setting_for_iter('seed', 1)
        assert sm.get_setting_for_iter('seed', 2) == 456

        # transforms
        assert w.all_settings[0].transforms == [None]
        assert sm.get_setting_for_iter('transforms', 0) == []
        assert w.all_settings[1].transforms == [{'name':'zoom_in','zoom_amt':0.5}]
        assert sm.get_setting_for_iter('transforms', 1) == [{'name':'zoom_in','zoom_amt':0.5}]
        assert w.all_settings[2].transforms == [None]
        assert sm.get_setting_for_iter('transforms', 2) == [{'name':'zoom_in','zoom_amt':0.5}]
        assert w.all_settings[3].transforms == [None]
        assert sm.get_setting_for_iter('transforms', 3) == [{'name':'zoom_in','zoom_amt':0.5}]
        assert w.all_settings[4].transforms == []
        assert sm.get_setting_for_iter('transforms', 4) == []
        for i in range(5,10):
            assert w.all_settings[i].transforms == [None]
            assert sm.get_setting_for_iter('transforms', i) == []

        # loras
        assert w.all_settings[0].loras == [None]
        assert sm.get_setting_for_iter('loras', 0) == []
        assert w.all_settings[1].loras == [('foo', 0.5)]
        assert sm.get_setting_for_iter('loras', 1) == [('foo', 0.5)]
        assert w.all_settings[2].loras == [None]
        assert sm.get_setting_for_iter('loras', 2) == [('foo', 0.5)]
        assert w.all_settings[3].loras == [None]
        assert sm.get_setting_for_iter('loras', 3) == [('foo', 0.5)]
        assert w.all_settings[4].loras == []
        assert sm.get_setting_for_iter('loras', 4) == []
        for i in range(5,10):
            assert w.all_settings[i].loras == [None]
            assert sm.get_setting_for_iter('loras', i) == []

    finally:
        try:
            temp_file.close()
            os.unlink(temp_file.name)
        except:
            pass
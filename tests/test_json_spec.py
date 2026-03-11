import tempfile
import warnings
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
        neg_prompt='ugly',
        denoise_steps=30,
        denoise_amt=0.5,
        con_deltas=[
            ConDelta(pos="bright", neg="dark", strength=5.0),
            ConDelta(pos="friendly", neg="mean", strength=-1.0),
        ],
        loras=[LoraFilter('lora_a.safetensors', 1.0), LoraFilter('lora_b.safetensors', .212354)],
        transforms=[{'name':'zoom_in', 'zoom_amt':0.985}]
    )

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_serdes_loopsettings():
    test = get_loop_settings()
    test_json = test.to_json(indent=4)
    test_clone = LoopSettings.schema().loads(test_json)
    assert test == test_clone

@pytest.mark.filterwarnings("ignore::UserWarning")
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
                canny=Canny(0.0, 0.5, 0.75),
                con_deltas=[
                    ConDelta(pos="bright", neg="dark", strength=5.0),
                    ConDelta(pos="friendly", neg="mean", strength=-1.0),
                ],
            ),
            # 1
            LoopSettings(
                transforms=[{'name':'zoom_in','zoom_amt':0.5}],
                con_deltas=[
                    ConDelta(pos="rich", neg="poor", strength=2.5)
                ],
                loras=[LoraFilter('foo', 0.5)],
            ),
            # 2
            LoopSettings(
                seed=456,
                canny=None,
            ),
            # 3
            LoopSettings(
                con_deltas=[],
            ),
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
        sm = SettingsManager(temp_file.name, animation_params={})

        # con_delta
        assert w.all_settings[0].con_deltas == [ConDelta(pos="bright", neg="dark", strength=5.0), ConDelta(pos="friendly", neg="mean", strength=-1.0)]
        assert sm.get_setting_for_iter('con_deltas', 0) == [ConDelta(pos="bright", neg="dark", strength=5.0), ConDelta(pos="friendly", neg="mean", strength=-1.0)]
        assert w.all_settings[1].con_deltas == [ConDelta(pos="rich", neg="poor", strength=2.5)]
        assert sm.get_setting_for_iter('con_deltas', 1) == [ConDelta(pos="rich", neg="poor", strength=2.5)]

        # actual value from JSON
        assert w.all_settings[2].con_deltas == EMPTY_LIST
        # infered value we want
        assert sm.get_setting_for_iter('con_deltas', 2) == [ConDelta(pos="rich", neg="poor", strength=2.5)]

        assert w.all_settings[3].con_deltas == []
        assert sm.get_setting_for_iter('con_deltas', 3) == []

        for i in range(4,10):
            assert w.all_settings[i].con_deltas == EMPTY_LIST
            assert sm.get_setting_for_iter('con_deltas', i) == []

        # canny
        assert w.all_settings[0].canny == Canny(0.0, 0.5, 0.75)
        assert sm.get_setting_for_iter('canny', 0) == Canny(0.0, 0.5, 0.75)
        assert w.all_settings[1].canny == EMPTY_OBJECT
        assert sm.get_setting_for_iter('canny', 1) == Canny(0.0, 0.5, 0.75)
        assert w.all_settings[2].canny == None
        assert sm.get_setting_for_iter('canny', 2) == None
        assert w.all_settings[3].canny == EMPTY_OBJECT
        assert sm.get_setting_for_iter('canny', 3) == None
        assert w.all_settings[4].canny == EMPTY_OBJECT
        assert sm.get_setting_for_iter('canny', 4) == None
        for i in range(5,10):
            assert w.all_settings[i].canny == EMPTY_OBJECT
            assert sm.get_setting_for_iter('canny', i) == None

        # seed
        assert sm.get_setting_for_iter('seed', 0) != sm.get_setting_for_iter('seed', 1)
        assert sm.get_setting_for_iter('seed', 2) == 456

        # transforms
        assert w.all_settings[0].transforms == EMPTY_LIST
        assert sm.get_setting_for_iter('transforms', 0) == []
        assert w.all_settings[1].transforms == [{'name':'zoom_in','zoom_amt':0.5}]
        assert sm.get_setting_for_iter('transforms', 1) == [{'name':'zoom_in','zoom_amt':0.5}]
        assert w.all_settings[2].transforms == EMPTY_LIST
        assert sm.get_setting_for_iter('transforms', 2) == [{'name':'zoom_in','zoom_amt':0.5}]
        assert w.all_settings[3].transforms == EMPTY_LIST
        assert sm.get_setting_for_iter('transforms', 3) == [{'name':'zoom_in','zoom_amt':0.5}]
        assert w.all_settings[4].transforms == []
        assert sm.get_setting_for_iter('transforms', 4) == []
        for i in range(5,10):
            assert w.all_settings[i].transforms == EMPTY_LIST
            assert sm.get_setting_for_iter('transforms', i) == []

        # loras
        assert w.all_settings[0].loras == EMPTY_LIST
        assert sm.get_setting_for_iter('loras', 0) == []
        assert w.all_settings[1].loras == [LoraFilter('foo', 0.5)]
        assert sm.get_setting_for_iter('loras', 1) == [LoraFilter('foo', 0.5)]
        assert w.all_settings[2].loras == EMPTY_LIST
        assert sm.get_setting_for_iter('loras', 2) == [LoraFilter('foo', 0.5)]
        assert w.all_settings[3].loras == EMPTY_LIST
        assert sm.get_setting_for_iter('loras', 3) == [LoraFilter('foo', 0.5)]
        assert w.all_settings[4].loras == []
        assert sm.get_setting_for_iter('loras', 4) == []
        for i in range(5,10):
            assert w.all_settings[i].loras == EMPTY_LIST
            assert sm.get_setting_for_iter('loras', i) == []

    finally:
        try:
            temp_file.close()
            os.unlink(temp_file.name)
        except:
            pass


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_serdes_workflow_with_overrides():
    ls = get_loop_settings()
    w = Workflow(
        all_settings=[ls],
        version=1,
        frame_overrides={1: {'cfg': 20.0}},
        formula_overrides={0: {'denoise_amt': 0.8}},
    )
    # __post_init__ normalizes keys to int and values to typed
    assert w.frame_overrides == {1: {'cfg': 20.0}}
    test_json = w.to_json(indent=4)
    clone = Workflow.schema().loads(test_json)
    assert clone.frame_overrides == {1: {'cfg': 20.0}}
    assert clone.formula_overrides == {0: {'denoise_amt': 0.8}}
    assert clone.version == 1
    assert len(clone.all_settings) == 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_workflow_without_overrides_fields():
    ls = get_loop_settings()
    w = Workflow(all_settings=[ls], version=1)
    test_json = w.to_json(indent=4)
    clone = Workflow.schema().loads(test_json)
    assert clone.frame_overrides is None
    assert clone.formula_overrides is None


def test_workflow_loads_from_existing_json():
    """Existing JSON files (without override fields) still parse correctly."""
    sm = SettingsManager('data/tests/test_no_lora.json', animation_params={})
    assert sm.workflow.frame_overrides is None
    assert sm.workflow.formula_overrides is None
    assert len(sm.workflow.all_settings) == 2


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestOverrideSerdesRoundtrip:
    """Verify serialize_override_value -> deserialize_override_value produces identical values."""

    def test_canny_roundtrip(self):
        original = Canny(low_thresh=0.1, high_thresh=0.5, strength=0.8)
        serialized = serialize_override_value(original)
        deserialized = deserialize_override_value('canny', serialized)
        assert isinstance(deserialized, Canny)
        assert deserialized == original

    def test_canny_none_roundtrip(self):
        serialized = serialize_override_value(None)
        deserialized = deserialize_override_value('canny', serialized)
        assert deserialized is None

    def test_loras_roundtrip(self):
        original = [LoraFilter('a.safetensors', 0.5), LoraFilter('b.safetensors', 1.0)]
        serialized = serialize_override_value(original)
        deserialized = deserialize_override_value('loras', serialized)
        assert len(deserialized) == 2
        assert all(isinstance(l, LoraFilter) for l in deserialized)
        assert deserialized == original

    def test_con_deltas_roundtrip(self):
        original = [ConDelta(pos='bright', neg='dark', strength=5.0)]
        serialized = serialize_override_value(original)
        deserialized = deserialize_override_value('con_deltas', serialized)
        assert len(deserialized) == 1
        assert isinstance(deserialized[0], ConDelta)
        assert deserialized == original

    def test_transforms_roundtrip(self):
        original = [{'name': 'zoom_in', 'zoom_amt': 0.5}]
        serialized = serialize_override_value(original)
        deserialized = deserialize_override_value('transforms', serialized)
        assert deserialized == original

    def test_float_roundtrip(self):
        for key in ('denoise_amt', 'cfg'):
            serialized = serialize_override_value(0.75)
            deserialized = deserialize_override_value(key, serialized)
            assert deserialized == 0.75
            assert isinstance(deserialized, float)

    def test_int_roundtrip(self):
        for key in ('denoise_steps', 'seed'):
            serialized = serialize_override_value(42)
            deserialized = deserialize_override_value(key, serialized)
            assert deserialized == 42
            assert isinstance(deserialized, int)

    def test_string_roundtrip(self):
        for key in ('prompt', 'neg_prompt', 'checkpoint'):
            serialized = serialize_override_value('hello world')
            deserialized = deserialize_override_value(key, serialized)
            assert deserialized == 'hello world'

    def test_expression_string_roundtrip(self):
        """Expression strings in numeric fields survive the round-trip."""
        for key in ('denoise_amt', 'cfg', 'denoise_steps'):
            expr = '0.3 + 0.01*n'
            serialized = serialize_override_value(expr)
            deserialized = deserialize_override_value(key, serialized)
            assert deserialized == expr

    def test_repeated_roundtrip(self):
        """Multiple serialize/deserialize cycles produce stable results."""
        original = Canny(low_thresh=0.2, high_thresh=0.7, strength=0.9)
        value = original
        for _ in range(5):
            serialized = serialize_override_value(value)
            value = deserialize_override_value('canny', serialized)
        assert isinstance(value, Canny)
        assert value == original
import json
import os
import threading
from unittest.mock import MagicMock
from interactive.loop_state import (
    LoopState, LoopStatus,
    _remove_json_field, _inject_json_field, _write_workflow_with_overrides,
)
from utils.json_spec import LoopSettings, Canny, LoraFilter, ConDelta, SettingsManager


class TestLoopStateBasics:
    def test_initial_state(self):
        state = LoopState(total_iterations=50, output_folder='/tmp/test')
        assert state.get_status() == LoopStatus.RUNNING
        assert state.get_current_iteration() == 0
        assert state.get_total_iterations() == 50
        assert state.get_latest_image_index() == 0
        assert state.get_error() is None
        assert state.get_output_folder() == '/tmp/test'

    def test_iteration_tracking(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        state.set_current_iteration(5)
        state.set_latest_image_index(6)
        assert state.get_current_iteration() == 5
        assert state.get_latest_image_index() == 6

    def test_status_transitions(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        assert state.get_status() == LoopStatus.RUNNING

        state.set_status(LoopStatus.PAUSED)
        assert state.get_status() == LoopStatus.PAUSED

        state.set_status(LoopStatus.STOPPED)
        assert state.get_status() == LoopStatus.STOPPED

    def test_error_sets_stopped(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        state.set_error("GPU OOM")
        assert state.get_status() == LoopStatus.STOPPED
        assert state.get_error() == "GPU OOM"


class TestLoopStateSettings:
    def test_store_and_retrieve(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        state.store_settings(0, '{"seed": 123}')
        state.store_settings(1, '{"seed": 456}')
        assert state.get_settings(0) == '{"seed": 123}'
        assert state.get_settings(1) == '{"seed": 456}'
        assert state.get_settings(2) is None

    def test_clear_settings_from(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        for i in range(5):
            state.store_settings(i, f'{{"iter": {i}}}')

        state.clear_settings_from(3)
        assert state.get_settings(0) is not None
        assert state.get_settings(1) is not None
        assert state.get_settings(2) is not None
        assert state.get_settings(3) is None
        assert state.get_settings(4) is None

    def test_clear_settings_from_zero(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        state.store_settings(0, '{"a": 1}')
        state.store_settings(1, '{"a": 2}')
        state.clear_settings_from(0)
        assert state.get_settings(0) is None
        assert state.get_settings(1) is None


class TestLoopStatePauseResume:
    def test_pause_resume(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        state.pause()
        assert state.get_status() == LoopStatus.PAUSED

        state.resume()
        assert state.get_status() == LoopStatus.RUNNING

    def test_pause_blocks_thread(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        state.pause()
        unblocked = threading.Event()

        def worker():
            state.wait_if_paused()
            unblocked.set()

        t = threading.Thread(target=worker)
        t.start()

        # Worker should be blocked
        assert not unblocked.wait(timeout=0.2)

        # Resume should unblock
        state.resume()
        assert unblocked.wait(timeout=1.0)
        t.join()

    def test_wait_if_paused_does_not_block_when_running(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        done = threading.Event()

        def worker():
            state.wait_if_paused()
            done.set()

        t = threading.Thread(target=worker)
        t.start()
        assert done.wait(timeout=1.0)
        t.join()


class TestLoopStateRestart:
    def test_restart_request_lifecycle(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        assert state.get_and_clear_restart_request() is None

        state.request_restart(5)
        assert state.get_and_clear_restart_request() == 5

        # Should be cleared after get
        assert state.get_and_clear_restart_request() is None

    def test_restart_overwrites_previous_request(self):
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        state.request_restart(5)
        state.request_restart(3)
        assert state.get_and_clear_restart_request() == 3

    def test_resume_does_not_clear_restart_request(self):
        """Critical: resume() must NOT clear the restart request.
        This prevents a race condition when restarting from a paused state."""
        state = LoopState(total_iterations=10, output_folder='/tmp/test')
        state.pause()
        state.request_restart(5)
        state.resume()
        assert state.get_and_clear_restart_request() == 5


def _make_mock_sm(total_iterations=5, frame_overrides=None, formula_overrides=None):
    """Create a mock SettingsManager that returns simple elaborated settings.

    Override dicts should use int keys and typed values (matching Workflow.__post_init__).
    """
    sm = MagicMock()
    sm.get_total_iterations.return_value = total_iterations
    sm.workflow.frame_overrides = frame_overrides
    sm.workflow.formula_overrides = formula_overrides
    # Provide a real list of LoopSettings for all_settings so len() works
    # and formula overrides can setattr on them
    section_ls = LoopSettings(loop_iterations=total_iterations)
    section_ls.offset = 0
    sm.workflow.all_settings = [section_ls]

    def make_ls(i):
        ls = LoopSettings(loop_iterations=1)
        ls.offset = 0
        ls.prompt = f"prompt_{i}"
        ls.neg_prompt = "neg"
        ls.denoise_amt = 0.5
        ls.denoise_steps = 20
        ls.cfg = 7.0
        ls.seed = 100 + i
        ls.transforms = []
        ls.loras = []
        ls.canny = None
        ls.con_deltas = []
        ls.clip = []
        ls.checkpoint = "test.safetensors"
        return ls

    sm.get_elaborated_loopsettings_for_iter.side_effect = make_ls
    return sm


class TestPreElaboratedSettings:
    def test_init_pre_elaborated(self):
        state = LoopState(total_iterations=5, output_folder='/tmp/test')
        sm = _make_mock_sm(5)
        state.init_pre_elaborated(sm, 'test.json', {})
        for i in range(5):
            ls = state.get_pre_elaborated(i)
            assert ls.prompt == f"prompt_{i}"

    def test_get_pre_elaborated_returns_deepcopy(self):
        state = LoopState(total_iterations=3, output_folder='/tmp/test')
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, 'test.json', {})
        ls1 = state.get_pre_elaborated(0)
        ls2 = state.get_pre_elaborated(0)
        ls1.prompt = "modified"
        assert ls2.prompt == "prompt_0"

    def test_apply_frame_override_persists(self):
        state = LoopState(total_iterations=3, output_folder='/tmp/test')
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, 'test.json', {})
        state.apply_frame_override(1, {'denoise_amt': 0.99})
        ls = state.get_pre_elaborated(1)
        assert ls.denoise_amt == 0.99
        # Other frames unaffected
        ls0 = state.get_pre_elaborated(0)
        assert ls0.denoise_amt == 0.5

    def test_override_survives_multiple_gets(self):
        """Override should persist across multiple gets (not one-shot)."""
        state = LoopState(total_iterations=3, output_folder='/tmp/test')
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, 'test.json', {})
        state.apply_frame_override(1, {'cfg': 15.0})
        assert state.get_pre_elaborated(1).cfg == 15.0
        assert state.get_pre_elaborated(1).cfg == 15.0

    def test_get_all_overridden_frames(self):
        state = LoopState(total_iterations=5, output_folder='/tmp/test')
        sm = _make_mock_sm(5)
        state.init_pre_elaborated(sm, 'test.json', {})
        state.apply_frame_override(1, {'denoise_amt': 0.9})
        state.apply_frame_override(3, {'cfg': 10.0, 'prompt': 'new'})
        result = state.get_all_overridden_frames()
        assert 1 in result
        assert 'denoise_amt' in result[1]
        assert 3 in result
        assert set(result[3]) == {'cfg', 'prompt'}

    def test_is_field_overridden(self):
        state = LoopState(total_iterations=3, output_folder='/tmp/test')
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, 'test.json', {})
        state.apply_frame_override(1, {'seed': 999})
        assert state.is_field_overridden(1, 'seed') is True
        assert state.is_field_overridden(1, 'cfg') is False
        assert state.is_field_overridden(0, 'seed') is False

    def test_get_settings_reflects_frame_override(self):
        """After applying a frame override, get_settings should return the overridden value."""
        state = LoopState(total_iterations=3, output_folder='/tmp/test')
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, 'test.json', {})
        # Simulate iteration running and caching settings (as _run_iteration does)
        state.store_settings(1, '{"cfg": 7.0, "prompt": "original"}')
        assert '"cfg": 7.0' in state.get_settings(1)
        # Apply override — should invalidate the stale cache
        state.apply_frame_override(1, {'cfg': 20.0})
        settings_json = state.get_settings(1)
        assert '20.0' in settings_json
        assert '7.0' not in settings_json

    def test_reset_all_overrides(self):
        state = LoopState(total_iterations=3, output_folder='/tmp/test')
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, 'test.json', {})
        state.apply_frame_override(1, {'denoise_amt': 0.99})
        # Reset with a fresh SM
        sm2 = _make_mock_sm(3)
        state.reset_all_overrides(sm2)
        ls = state.get_pre_elaborated(1)
        assert ls.denoise_amt == 0.5
        assert state.get_all_overridden_frames() == {}

    def test_re_elaborate_preserves_frame_overrides(self):
        state = LoopState(total_iterations=5, output_folder='/tmp/test')
        sm = _make_mock_sm(5)
        state.init_pre_elaborated(sm, 'test.json', {})
        state.apply_frame_override(3, {'cfg': 20.0})
        # Re-elaborate from iter 2 onward
        sm2 = _make_mock_sm(5)
        state.re_elaborate_from(2, sm2)
        # Frame 3 override should still be present
        ls3 = state.get_pre_elaborated(3)
        assert ls3.cfg == 20.0
        # Frame 2 should be fresh from sm2
        ls2 = state.get_pre_elaborated(2)
        assert ls2.cfg == 7.0

    def test_re_elaborate_refreshes_view_cache(self):
        state = LoopState(total_iterations=5, output_folder='/tmp/test')
        sm = _make_mock_sm(5)
        state.init_pre_elaborated(sm, 'test.json', {})
        state.store_settings(2, '{"old": true}')
        state.store_settings(3, '{"old": true}')
        state.re_elaborate_from(2, sm)
        # Old cache is cleared, but get_settings falls back to pre-elaborated
        settings_2 = state.get_settings(2)
        assert settings_2 is not None
        assert '"old"' not in settings_2
        assert '"prompt_2"' in settings_2
        # Frame before re-elaborate point retains its original cache
        state.store_settings(1, '{"cached": true}')
        assert state.get_settings(1) == '{"cached": true}'

    def test_stores_json_file_and_params(self):
        state = LoopState(total_iterations=3, output_folder='/tmp/test')
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, '/path/to/workflow.json', {'frame_delay': '100'})
        assert state.get_json_file() == '/path/to/workflow.json'
        assert state.get_animation_params() == {'frame_delay': '100'}


# --- Text manipulation tests ---

CLEAN_JSON = """{
    "all_settings": [
        {
            "loop_iterations": 3,
            "checkpoint": "test.safetensors",
            "prompt": "test prompt",
            "denoise_steps": 20,
            "denoise_amt": 0.5,
            "cfg": 7.0
        }
    ],
    "version": 1
}"""

COMMENTED_JSON = """{
    // This is a workflow comment
    "all_settings": [
        {
            "loop_iterations": 3,
            // inner comment
            "prompt": "test prompt",
            "denoise_steps": 20,
            "denoise_amt": 0.5,
            "cfg": 7.0
        }
    ],
    // comment between fields
    "version": 1
}"""

JSON_WITH_FRAME_OVERRIDES = """{
    // Top comment
    "all_settings": [
        {
            "loop_iterations": 3,
            "prompt": "test prompt",
            "denoise_steps": 20,
            "denoise_amt": 0.5,
            "cfg": 7.0
        }
    ],
    "version": 1,
    "frame_overrides": {
        // comment inside overrides
        "1": {
            "cfg": 20.0
        }
    }
}"""

JSON_WITH_BOTH_OVERRIDES = """{
    // Top comment
    "all_settings": [
        {
            "loop_iterations": 3,
            "prompt": "test prompt",
            "denoise_steps": 20,
            "denoise_amt": 0.5,
            "cfg": 7.0
        }
    ],
    "version": 1,
    "formula_overrides": {
        "0": {
            "denoise_amt": 0.8
        }
    },
    "frame_overrides": {
        "1": {
            "cfg": 20.0
        }
    }
}"""


def _strip_comments(text):
    """Strip // comment lines to get parseable JSON."""
    return '\n'.join(line for line in text.splitlines() if not line.strip().startswith('//'))


class TestRemoveJsonField:
    def test_no_op_when_field_absent(self):
        result = _remove_json_field(CLEAN_JSON, 'frame_overrides')
        assert result == CLEAN_JSON

    def test_removes_last_field(self):
        result = _remove_json_field(JSON_WITH_FRAME_OVERRIDES, 'frame_overrides')
        parsed = json.loads(_strip_comments(result))
        assert 'frame_overrides' not in parsed
        assert parsed['version'] == 1
        assert len(parsed['all_settings']) == 1

    def test_removes_non_last_field(self):
        result = _remove_json_field(JSON_WITH_BOTH_OVERRIDES, 'formula_overrides')
        parsed = json.loads(_strip_comments(result))
        assert 'formula_overrides' not in parsed
        assert 'frame_overrides' in parsed
        assert parsed['frame_overrides']['1']['cfg'] == 20.0

    def test_preserves_comments_outside_removed_field(self):
        result = _remove_json_field(JSON_WITH_FRAME_OVERRIDES, 'frame_overrides')
        assert '// Top comment' in result

    def test_removes_comments_inside_field(self):
        result = _remove_json_field(JSON_WITH_FRAME_OVERRIDES, 'frame_overrides')
        assert '// comment inside overrides' not in result

    def test_handles_braces_in_string_values(self):
        json_with_braces = """{
    "all_settings": [],
    "version": 1,
    "frame_overrides": {
        "0": {
            "prompt": "a {fantasy} castle with {magic}"
        }
    }
}"""
        result = _remove_json_field(json_with_braces, 'frame_overrides')
        parsed = json.loads(_strip_comments(result))
        assert 'frame_overrides' not in parsed
        assert parsed['version'] == 1


class TestInjectJsonField:
    def test_inject_into_clean_json(self):
        overrides = {"1": {"cfg": 20.0}}
        result = _inject_json_field(CLEAN_JSON, 'frame_overrides', overrides)
        parsed = json.loads(result)
        assert parsed['frame_overrides'] == overrides
        assert parsed['version'] == 1
        assert len(parsed['all_settings']) == 1

    def test_preserves_comments(self):
        overrides = {"1": {"cfg": 20.0}}
        result = _inject_json_field(COMMENTED_JSON, 'frame_overrides', overrides)
        assert '// This is a workflow comment' in result
        assert '// inner comment' in result
        assert '// comment between fields' in result
        # Still valid JSON after stripping comments
        parsed = json.loads(_strip_comments(result))
        assert parsed['frame_overrides'] == overrides

    def test_inject_two_fields_sequentially(self):
        formula = {"0": {"denoise_amt": 0.9}}
        frame = {"2": {"cfg": 15.0}}
        text = _inject_json_field(CLEAN_JSON, 'formula_overrides', formula)
        text = _inject_json_field(text, 'frame_overrides', frame)
        parsed = json.loads(text)
        assert parsed['formula_overrides'] == formula
        assert parsed['frame_overrides'] == frame
        assert parsed['version'] == 1


class TestRemoveAndReinject:
    def test_replace_existing_overrides(self):
        """Remove old overrides and inject new ones, preserving comments."""
        text = JSON_WITH_BOTH_OVERRIDES
        text = _remove_json_field(text, 'frame_overrides')
        text = _remove_json_field(text, 'formula_overrides')
        new_frame = {"2": {"denoise_amt": 0.1}}
        text = _inject_json_field(text, 'frame_overrides', new_frame)
        parsed = json.loads(_strip_comments(text))
        assert parsed['frame_overrides'] == new_frame
        assert 'formula_overrides' not in parsed
        # Top comment preserved
        assert '// Top comment' in text

    def test_roundtrip_both_fields(self):
        """Remove both fields, re-inject new ones."""
        text = JSON_WITH_BOTH_OVERRIDES
        text = _remove_json_field(text, 'frame_overrides')
        text = _remove_json_field(text, 'formula_overrides')
        new_formula = {"0": {"cfg": 99.0}}
        new_frame = {"1": {"prompt": "new"}}
        text = _inject_json_field(text, 'formula_overrides', new_formula)
        text = _inject_json_field(text, 'frame_overrides', new_frame)
        parsed = json.loads(_strip_comments(text))
        assert parsed['formula_overrides'] == new_formula
        assert parsed['frame_overrides'] == new_frame
        assert parsed['version'] == 1
        assert '// Top comment' in text


# --- LoopState integration tests ---

def _write_test_json(path, text=CLEAN_JSON):
    """Write test JSON to a file and return the path."""
    with open(path, 'w') as f:
        f.write(text)
    return str(path)


class TestOverridePersistence:
    def test_persist_frame_overrides_creates_file(self, tmp_path):
        json_path = _write_test_json(tmp_path / 'workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        state.apply_frame_override(1, {'cfg': 20.0})
        state.persist_overrides()
        output_file = os.path.join(output_dir, 'workflow.json')
        assert os.path.exists(output_file)
        with open(output_file) as f:
            parsed = json.loads(f.read())
        assert parsed['frame_overrides']['1']['cfg'] == 20.0

    def test_persist_formula_overrides(self, tmp_path):
        json_path = _write_test_json(tmp_path / 'workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        state.record_formula_override(0, {'denoise_amt': 0.9})
        state.persist_overrides()
        output_file = os.path.join(output_dir, 'workflow.json')
        with open(output_file) as f:
            parsed = json.loads(f.read())
        assert parsed['formula_overrides']['0']['denoise_amt'] == 0.9
        assert 'frame_overrides' not in parsed

    def test_persist_both_override_types(self, tmp_path):
        json_path = _write_test_json(tmp_path / 'workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        state.apply_frame_override(1, {'cfg': 20.0})
        state.record_formula_override(0, {'denoise_amt': 0.8})
        state.persist_overrides()
        output_file = os.path.join(output_dir, 'workflow.json')
        with open(output_file) as f:
            parsed = json.loads(f.read())
        assert parsed['frame_overrides']['1']['cfg'] == 20.0
        assert parsed['formula_overrides']['0']['denoise_amt'] == 0.8

    def test_persist_preserves_comments(self, tmp_path):
        json_path = _write_test_json(tmp_path / 'workflow.json', COMMENTED_JSON)
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        state.apply_frame_override(0, {'cfg': 10.0})
        state.persist_overrides()
        output_file = os.path.join(output_dir, 'workflow.json')
        with open(output_file) as f:
            text = f.read()
        assert '// This is a workflow comment' in text
        assert '// inner comment' in text
        assert '// comment between fields' in text

    def test_persist_reflects_values(self, tmp_path):
        json_path = _write_test_json(tmp_path / 'workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        state.apply_frame_override(2, {'cfg': 20.0, 'prompt': 'new prompt'})
        state.persist_overrides()
        output_file = os.path.join(output_dir, 'workflow.json')
        with open(output_file) as f:
            parsed = json.loads(f.read())
        assert parsed['frame_overrides']['2']['cfg'] == 20.0
        assert parsed['frame_overrides']['2']['prompt'] == 'new prompt'

    def test_persist_replaces_existing_overrides(self, tmp_path):
        """Input JSON already has override fields; new persist should replace them."""
        json_path = _write_test_json(tmp_path / 'workflow.json', JSON_WITH_BOTH_OVERRIDES)
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        # Apply new overrides (different from what's in the file)
        state.apply_frame_override(0, {'denoise_amt': 0.1})
        state.record_formula_override(0, {'cfg': 99.0})
        state.persist_overrides()
        output_file = os.path.join(output_dir, 'workflow.json')
        with open(output_file) as f:
            text = f.read()
        parsed = json.loads(_strip_comments(text))
        # New overrides present
        assert parsed['frame_overrides']['0']['denoise_amt'] == 0.1
        assert parsed['formula_overrides']['0']['cfg'] == 99.0
        # Old overrides gone
        assert '1' not in parsed.get('frame_overrides', {})
        # Comments outside overrides preserved
        assert '// Top comment' in text

    def test_delete_overrides_file(self, tmp_path):
        json_path = _write_test_json(tmp_path / 'workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        state.apply_frame_override(0, {'cfg': 10.0})
        state.persist_overrides()
        output_file = os.path.join(output_dir, 'workflow.json')
        assert os.path.exists(output_file)
        state.delete_overrides_file()
        assert not os.path.exists(output_file)

    def test_persist_empty_deletes(self, tmp_path):
        json_path = _write_test_json(tmp_path / 'workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        # Create a file first
        state.apply_frame_override(0, {'cfg': 10.0})
        state.persist_overrides()
        output_file = os.path.join(output_dir, 'workflow.json')
        assert os.path.exists(output_file)
        # Reset clears overrides, then persist should delete
        sm2 = _make_mock_sm(3)
        state.reset_all_overrides(sm2)
        state.persist_overrides()
        assert not os.path.exists(output_file)

    def test_persist_serializes_complex_types(self, tmp_path):
        """Canny and LoraFilter objects should be serialized to dicts."""
        json_path = _write_test_json(tmp_path / 'workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        state.apply_frame_override(0, {
            'canny': Canny(low_thresh=0.1, high_thresh=0.5, strength=0.8),
            'loras': [LoraFilter(lora_path='test.safetensors', lora_strength=0.5)],
        })
        state.persist_overrides()
        output_file = os.path.join(output_dir, 'workflow.json')
        with open(output_file) as f:
            parsed = json.loads(f.read())
        canny = parsed['frame_overrides']['0']['canny']
        assert canny['low_thresh'] == 0.1
        assert canny['strength'] == 0.8
        loras = parsed['frame_overrides']['0']['loras']
        assert loras[0]['lora_path'] == 'test.safetensors'

    def test_output_filename_matches_input(self, tmp_path):
        json_path = _write_test_json(tmp_path / 'my_custom_workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        state.apply_frame_override(0, {'cfg': 10.0})
        state.persist_overrides()
        assert os.path.exists(os.path.join(output_dir, 'my_custom_workflow.json'))


class TestLoadOverridesFromWorkflow:
    def test_init_loads_frame_overrides(self, tmp_path):
        """JSON with frame_overrides should apply them to pre-elaborated settings."""
        sm = _make_mock_sm(3, frame_overrides={1: {'cfg': 99.0}})
        state = LoopState(total_iterations=3, output_folder=str(tmp_path))
        state.init_pre_elaborated(sm, 'test.json', {})
        ls = state.get_pre_elaborated(1)
        assert ls.cfg == 99.0
        # Other frames unaffected
        assert state.get_pre_elaborated(0).cfg == 7.0
        # Tracked as overridden
        assert state.is_field_overridden(1, 'cfg')
        assert not state.is_field_overridden(0, 'cfg')

    def test_init_loads_formula_overrides(self, tmp_path):
        """JSON with formula_overrides should apply them to SM sections before elaboration."""
        json_text = """{
    "all_settings": [
        {
            "loop_iterations": 3,
            "checkpoint": "sdXL_v10VAEFix.safetensors",
            "prompt": "original",
            "denoise_steps": 20,
            "denoise_amt": 0.5,
            "cfg": 7.0
        }
    ],
    "version": 1,
    "formula_overrides": {
        "0": {
            "cfg": 99.0
        }
    }
}"""
        json_path = _write_test_json(tmp_path / 'workflow.json', json_text)
        sm = SettingsManager(json_path, {})
        state = LoopState(total_iterations=3, output_folder=str(tmp_path))
        state.init_pre_elaborated(sm, json_path, {})
        # All frames in section 0 should have the formula override applied
        for i in range(3):
            ls = state.get_pre_elaborated(i)
            assert ls.cfg == 99.0, f"Frame {i} cfg should be 99.0"
        # Formula overrides should be tracked
        sections = state.get_all_overridden_sections()
        assert 0 in sections
        assert 'cfg' in sections[0]

    def test_init_loads_both_override_types(self, tmp_path):
        """Formula applied first (to SM), then frame applied after elaboration."""
        json_text = """{
    "all_settings": [
        {
            "loop_iterations": 3,
            "checkpoint": "sdXL_v10VAEFix.safetensors",
            "prompt": "original",
            "denoise_steps": 20,
            "denoise_amt": 0.5,
            "cfg": 7.0
        }
    ],
    "version": 1,
    "formula_overrides": {
        "0": {
            "cfg": 50.0
        }
    },
    "frame_overrides": {
        "1": {
            "cfg": 99.0
        }
    }
}"""
        json_path = _write_test_json(tmp_path / 'workflow.json', json_text)
        sm = SettingsManager(json_path, {})
        state = LoopState(total_iterations=3, output_folder=str(tmp_path))
        state.init_pre_elaborated(sm, json_path, {})
        # Frame 0 and 2: formula override (50.0)
        assert state.get_pre_elaborated(0).cfg == 50.0
        assert state.get_pre_elaborated(2).cfg == 50.0
        # Frame 1: frame override wins over formula (99.0)
        assert state.get_pre_elaborated(1).cfg == 99.0

    def test_formula_override_tracked_in_state(self):
        sm = _make_mock_sm(3, formula_overrides={0: {'denoise_amt': 0.9}})
        state = LoopState(total_iterations=3, output_folder='/tmp/test')
        state.init_pre_elaborated(sm, 'test.json', {})
        sections = state.get_all_overridden_sections()
        assert 0 in sections
        assert 'denoise_amt' in sections[0]

    def test_reset_clears_formula_overrides(self):
        sm = _make_mock_sm(3, formula_overrides={0: {'denoise_amt': 0.9}})
        state = LoopState(total_iterations=3, output_folder='/tmp/test')
        state.init_pre_elaborated(sm, 'test.json', {})
        assert state.get_all_overridden_sections() != {}
        sm2 = _make_mock_sm(3)
        state.reset_all_overrides(sm2)
        assert state.get_all_overridden_sections() == {}
        assert state.get_all_overridden_frames() == {}


class TestOverrideSerdesIntegration:
    """Full persist -> load -> verify round-trips through actual JSON files."""

    def test_persist_load_roundtrip_complex_types(self, tmp_path):
        """Persist Canny, LoraFilter, ConDelta overrides, load them back, verify typed objects."""
        json_path = _write_test_json(tmp_path / 'workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)

        # Phase 1: apply complex overrides and persist
        state = LoopState(total_iterations=3, output_folder=output_dir)
        sm = _make_mock_sm(3)
        state.init_pre_elaborated(sm, json_path, {})
        state.apply_frame_override(0, {
            'canny': Canny(low_thresh=0.1, high_thresh=0.5, strength=0.8),
            'loras': [LoraFilter('test.safetensors', 0.75)],
            'con_deltas': [ConDelta(pos='bright', neg='dark', strength=3.0)],
        })
        state.persist_overrides()

        # Phase 2: load the persisted file and verify typed objects come back
        output_file = os.path.join(output_dir, 'workflow.json')
        sm2 = SettingsManager(output_file, {})
        state2 = LoopState(total_iterations=3, output_folder=output_dir)
        state2.init_pre_elaborated(sm2, output_file, {})

        ls = state2.get_pre_elaborated(0)
        assert isinstance(ls.canny, Canny)
        assert ls.canny == Canny(low_thresh=0.1, high_thresh=0.5, strength=0.8)
        assert len(ls.loras) == 1
        assert isinstance(ls.loras[0], LoraFilter)
        assert ls.loras[0] == LoraFilter('test.safetensors', 0.75)
        assert len(ls.con_deltas) == 1
        assert isinstance(ls.con_deltas[0], ConDelta)
        assert ls.con_deltas[0] == ConDelta(pos='bright', neg='dark', strength=3.0)

    def test_persist_load_roundtrip_formula_overrides(self, tmp_path):
        """Persist formula overrides with complex types, load back, verify applied to all frames."""
        json_text = """{
    "all_settings": [
        {
            "loop_iterations": 3,
            "checkpoint": "test.safetensors",
            "prompt": "test",
            "denoise_steps": 20,
            "denoise_amt": 0.5,
            "cfg": 7.0
        }
    ],
    "version": 1
}"""
        json_path = _write_test_json(tmp_path / 'workflow.json', json_text)
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)

        # Phase 1: apply formula override with complex type and persist
        sm = SettingsManager(json_path, {})
        state = LoopState(total_iterations=3, output_folder=output_dir)
        state.init_pre_elaborated(sm, json_path, {})
        canny_override = Canny(low_thresh=0.2, high_thresh=0.6, strength=0.9)
        setattr(sm.workflow.all_settings[0], 'canny', canny_override)
        state.record_formula_override(0, {'canny': canny_override})
        state.re_elaborate_from(0, sm)
        state.persist_overrides()

        # Phase 2: load and verify formula override applied to all frames
        output_file = os.path.join(output_dir, 'workflow.json')
        sm2 = SettingsManager(output_file, {})
        state2 = LoopState(total_iterations=3, output_folder=output_dir)
        state2.init_pre_elaborated(sm2, output_file, {})
        for i in range(3):
            ls = state2.get_pre_elaborated(i)
            assert isinstance(ls.canny, Canny), f"Frame {i}: canny should be Canny"
            assert ls.canny == canny_override, f"Frame {i}: canny value mismatch"
        assert 0 in state2.get_all_overridden_sections()

    def test_repeated_persist_load_cycle(self, tmp_path):
        """Override -> persist -> load -> override again -> persist -> load -> verify."""
        json_path = _write_test_json(tmp_path / 'workflow.json')
        output_dir = str(tmp_path / 'output')
        os.makedirs(output_dir)
        output_file = os.path.join(output_dir, 'workflow.json')

        # Cycle 1: override cfg on frame 1
        sm = _make_mock_sm(3)
        state = LoopState(total_iterations=3, output_folder=output_dir)
        state.init_pre_elaborated(sm, json_path, {})
        state.apply_frame_override(1, {'cfg': 20.0})
        state.persist_overrides()

        # Cycle 1 verify: load and check
        sm2 = SettingsManager(output_file, {})
        state2 = LoopState(total_iterations=3, output_folder=output_dir)
        state2.init_pre_elaborated(sm2, output_file, {})
        assert state2.get_pre_elaborated(1).cfg == 20.0

        # Cycle 2: add a canny override on frame 0 (from loaded state)
        state2.apply_frame_override(0, {
            'canny': Canny(low_thresh=0.3, high_thresh=0.7, strength=1.0),
        })
        state2.persist_overrides()

        # Cycle 2 verify: load and check both overrides survived
        sm3 = SettingsManager(output_file, {})
        state3 = LoopState(total_iterations=3, output_folder=output_dir)
        state3.init_pre_elaborated(sm3, output_file, {})
        assert state3.get_pre_elaborated(1).cfg == 20.0
        assert isinstance(state3.get_pre_elaborated(0).canny, Canny)
        assert state3.get_pre_elaborated(0).canny == Canny(low_thresh=0.3, high_thresh=0.7, strength=1.0)

        # Cycle 3: override frame 1 cfg again with a different value
        state3.apply_frame_override(1, {'cfg': 42.0})
        state3.persist_overrides()

        sm4 = SettingsManager(output_file, {})
        state4 = LoopState(total_iterations=3, output_folder=output_dir)
        state4.init_pre_elaborated(sm4, output_file, {})
        assert state4.get_pre_elaborated(1).cfg == 42.0
        assert state4.get_pre_elaborated(0).canny == Canny(low_thresh=0.3, high_thresh=0.7, strength=1.0)

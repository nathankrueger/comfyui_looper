import json
import os
import pytest
from unittest.mock import MagicMock
from PIL import Image

from interactive.loop_state import LoopState, LoopStatus
from interactive.app_state import AppState
from interactive.flask_app import create_app
from utils.image_store import FilesystemImageStore
from utils.json_spec import LoopSettings


class _TestAppState:
    """Minimal AppState-like wrapper for tests that wraps a pre-existing LoopState."""

    def __init__(self, loop_state: LoopState, image_store=None, json_name: str = 'test.json'):
        self._loop_state = loop_state
        self._image_store = image_store
        self._json_name = json_name

    def is_loop_running(self):
        return self._loop_state is not None

    def get_loop_state(self):
        return self._loop_state

    def get_current_json_name(self):
        return self._json_name

    def get_current_json_file(self):
        return self._json_name

    def get_current_output_folder(self):
        return self._loop_state.get_output_folder() if self._loop_state else None

    def get_image_store(self):
        return self._image_store

    def start_loop(self, json_file, output_folder=None):
        raise NotImplementedError("Not used in tests")

    def stop_loop(self):
        self._loop_state = None


@pytest.fixture
def app_and_state(tmp_path):
    state = LoopState(total_iterations=10, output_folder=str(tmp_path))
    image_store = FilesystemImageStore(str(tmp_path))
    app_state = _TestAppState(state, image_store=image_store)
    app = create_app(app_state)
    app.config['TESTING'] = True
    return app, state, tmp_path


class TestStatusEndpoint:
    def test_returns_initial_state(self, app_and_state):
        app, state, _ = app_and_state
        with app.test_client() as client:
            resp = client.get('/api/status')
            data = resp.get_json()
            assert resp.status_code == 200
            assert data['status'] == 'running'
            assert data['total_iterations'] == 10
            assert data['latest_image_index'] == 0
            assert data['current_iteration'] == 0
            assert data['error'] is None
            assert data['json_name'] == 'test.json'

    def test_reflects_state_changes(self, app_and_state):
        app, state, _ = app_and_state
        state.set_current_iteration(5)
        state.set_latest_image_index(6)
        state.pause()

        with app.test_client() as client:
            data = client.get('/api/status').get_json()
            assert data['status'] == 'paused'
            assert data['current_iteration'] == 5
            assert data['latest_image_index'] == 6

    def test_shows_error(self, app_and_state):
        app, state, _ = app_and_state
        state.set_error("something broke")

        with app.test_client() as client:
            data = client.get('/api/status').get_json()
            assert data['status'] == 'stopped'
            assert data['error'] == 'something broke'


class TestImageEndpoint:
    def test_image_not_found(self, app_and_state):
        app, _, _ = app_and_state
        with app.test_client() as client:
            resp = client.get('/api/image/999')
            assert resp.status_code == 404

    def test_image_found(self, app_and_state):
        app, state, tmp_path = app_and_state
        img = Image.new('RGB', (10, 10), color='red')
        img.save(str(tmp_path / 'loop_img_000001.png'))
        state.set_latest_image_index(1)

        with app.test_client() as client:
            resp = client.get('/api/image/1')
            assert resp.status_code == 200
            assert resp.content_type == 'image/png'

    def test_starting_image(self, app_and_state):
        app, _, tmp_path = app_and_state
        img = Image.new('RGB', (10, 10), color='blue')
        img.save(str(tmp_path / 'loop_img_000000.png'))

        with app.test_client() as client:
            resp = client.get('/api/image/0')
            assert resp.status_code == 200


class TestSettingsEndpoint:
    def test_starting_image_settings(self, app_and_state):
        app, _, _ = app_and_state
        with app.test_client() as client:
            resp = client.get('/api/settings/0')
            data = resp.get_json()
            assert resp.status_code == 200
            assert 'Starting image' in data['settings']

    def test_stored_settings(self, app_and_state):
        app, state, _ = app_and_state
        state.store_settings(0, '{"seed": 42, "prompt": "test"}')

        with app.test_client() as client:
            resp = client.get('/api/settings/1')  # file index 1 = iteration 0
            data = resp.get_json()
            assert resp.status_code == 200
            assert data['settings'] == '{"seed": 42, "prompt": "test"}'

    def test_missing_settings(self, app_and_state):
        app, _, _ = app_and_state
        with app.test_client() as client:
            resp = client.get('/api/settings/99')
            assert resp.status_code == 404


class TestPauseResumeEndpoints:
    def test_pause(self, app_and_state):
        app, state, _ = app_and_state
        with app.test_client() as client:
            resp = client.post('/api/pause')
            assert resp.status_code == 200
            assert state.get_status() == LoopStatus.PAUSED

    def test_resume(self, app_and_state):
        app, state, _ = app_and_state
        state.pause()
        with app.test_client() as client:
            resp = client.post('/api/resume')
            assert resp.status_code == 200
            assert state.get_status() == LoopStatus.RUNNING


class TestRestartEndpoint:
    def test_restart_valid(self, app_and_state):
        app, state, _ = app_and_state
        with app.test_client() as client:
            resp = client.post('/api/restart',
                data=json.dumps({'from_image_index': 5}),
                content_type='application/json')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['from_image_index'] == 5

    def test_restart_invalid_index_zero(self, app_and_state):
        app, _, _ = app_and_state
        with app.test_client() as client:
            resp = client.post('/api/restart',
                data=json.dumps({'from_image_index': 0}),
                content_type='application/json')
            assert resp.status_code == 400

    def test_restart_missing_index(self, app_and_state):
        app, _, _ = app_and_state
        with app.test_client() as client:
            resp = client.post('/api/restart',
                data=json.dumps({}),
                content_type='application/json')
            assert resp.status_code == 400

    def test_restart_from_paused_resumes_loop(self, app_and_state):
        app, state, _ = app_and_state
        state.pause()
        with app.test_client() as client:
            client.post('/api/restart',
                data=json.dumps({'from_image_index': 3}),
                content_type='application/json')
            # State should be RUNNING (resume was called to unblock the loop)
            assert state.get_status() == LoopStatus.RUNNING
            # But the restart request should still be set
            assert state.get_and_clear_restart_request() == 3


def _make_mock_sm(total_iterations=10):
    """Create a mock SettingsManager for override tests."""
    sm = MagicMock()
    sm.get_total_iterations.return_value = total_iterations
    sm.workflow.frame_overrides = None
    sm.workflow.formula_overrides = None
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
    sm.get_loopsettings_for_iter.return_value = (0, section_ls)
    return sm


@pytest.fixture
def app_with_overrides(tmp_path):
    """Fixture that provides an app with pre-elaborated state for override tests."""
    # Create a real JSON file so persist_overrides() can read it
    json_path = str(tmp_path / 'workflow.json')
    with open(json_path, 'w') as f:
        json.dump({"all_settings": [{"loop_iterations": 10}], "version": 1}, f, indent=4)
    output_dir = str(tmp_path / 'output')
    os.makedirs(output_dir)
    state = LoopState(total_iterations=10, output_folder=output_dir)
    sm = _make_mock_sm(10)
    state.init_pre_elaborated(sm, json_path, {})
    state.set_settings_manager(sm)
    image_store = FilesystemImageStore(output_dir)
    app_state = _TestAppState(state, image_store=image_store)
    app = create_app(app_state)
    app.config['TESTING'] = True
    return app, state


class TestOverridePauseCheck:
    def test_frame_override_rejected_when_running(self, app_with_overrides):
        app, state = app_with_overrides
        # State is RUNNING by default
        with app.test_client() as client:
            resp = client.post('/api/override/frame',
                data=json.dumps({'iteration': 1, 'overrides': {'cfg': 10.0}}),
                content_type='application/json')
            assert resp.status_code == 409
            data = resp.get_json()
            assert 'Pause' in data['error']

    def test_frame_override_accepted_when_paused(self, app_with_overrides):
        app, state = app_with_overrides
        state.pause()
        with app.test_client() as client:
            resp = client.post('/api/override/frame',
                data=json.dumps({'iteration': 1, 'overrides': {'cfg': 10.0}}),
                content_type='application/json')
            assert resp.status_code == 200

    def test_formula_override_rejected_when_running(self, app_with_overrides):
        app, state = app_with_overrides
        with app.test_client() as client:
            resp = client.post('/api/override/formula',
                data=json.dumps({'iteration': 1, 'overrides': {'cfg': '7.0'}}),
                content_type='application/json')
            assert resp.status_code == 409
            data = resp.get_json()
            assert 'Pause' in data['error']

    def test_formula_override_accepted_when_paused(self, app_with_overrides):
        app, state = app_with_overrides
        state.pause()
        with app.test_client() as client:
            resp = client.post('/api/override/formula',
                data=json.dumps({'iteration': 1, 'overrides': {'cfg': '7.0'}}),
                content_type='application/json')
            assert resp.status_code == 200

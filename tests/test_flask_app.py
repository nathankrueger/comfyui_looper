import json
import os
import pytest
from PIL import Image

from interactive.loop_state import LoopState, LoopStatus
from interactive.flask_app import create_app


@pytest.fixture
def app_and_state(tmp_path):
    state = LoopState(total_iterations=10, output_folder=str(tmp_path))
    app = create_app(state)
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

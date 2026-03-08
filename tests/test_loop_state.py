import threading
from interactive.loop_state import LoopState, LoopStatus


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

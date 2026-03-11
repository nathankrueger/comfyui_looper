"""Tests for interactive loop restart + frame override interaction."""

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from interactive.loop_state import LoopState, LoopStatus
from interactive.interactive_loop import _handle_restart
from utils.image_store import FilesystemImageStore
from utils.json_spec import LoopSettings
from utils.util import get_loop_img_filename


def _make_test_image():
    return Image.new('RGB', (64, 64), color='red')


def _create_image_store_with_images(tmp_path, count):
    """Create a FilesystemImageStore with `count` numbered images (1-based)."""
    store = FilesystemImageStore(str(tmp_path))
    for i in range(1, count + 1):
        filename = get_loop_img_filename(i)
        _make_test_image().save(os.path.join(tmp_path, filename))
    return store


def _make_loopsettings(**overrides):
    """Create a minimal LoopSettings suitable for testing."""
    ls = LoopSettings(loop_iterations=10)
    ls.offset = 0
    ls.prompt = "test prompt"
    ls.neg_prompt = "test neg"
    ls.denoise_amt = 0.5
    ls.denoise_steps = 20
    ls.cfg = 7.0
    ls.seed = 42
    ls.transforms = []
    ls.loras = []
    ls.canny = None
    ls.con_deltas = []
    ls.clip = []
    ls.checkpoint = "test.safetensors"
    for k, v in overrides.items():
        setattr(ls, k, v)
    return ls


def _make_mocks(loopsettings):
    """Create engine and settings manager mocks that work with _run_iteration."""
    engine = MagicMock()
    engine.compute_iteration.return_value = _make_test_image()
    engine.get_default_for_setting.return_value = None

    sm = MagicMock()
    sm.get_elaborated_loopsettings_for_iter.return_value = loopsettings
    sm.get_wavefile.return_value = None
    sm.get_total_iterations.return_value = 10

    return engine, sm


def _init_state_with_pre_elaborated(state, sm, loopsettings):
    """Initialize pre-elaborated settings on state using a mock SM."""
    sm.get_elaborated_loopsettings_for_iter.side_effect = lambda i: _make_loopsettings(
        **{k: getattr(loopsettings, k) for k in ['prompt', 'neg_prompt', 'denoise_amt',
           'denoise_steps', 'cfg', 'seed', 'transforms', 'loras', 'canny', 'con_deltas',
           'clip', 'checkpoint']}
    )
    state.init_pre_elaborated(sm, 'test.json', {})
    # Reset side_effect so _handle_restart's call to _run_iteration works
    sm.get_elaborated_loopsettings_for_iter.side_effect = None
    sm.get_elaborated_loopsettings_for_iter.return_value = loopsettings


class TestHandleRestartWithStickyOverrides:
    """Sticky frame overrides persist through restarts."""

    def test_sticky_overrides_applied_during_restart(self, tmp_path):
        """Sticky frame overrides should still be applied when restarting to that frame."""
        state = LoopState(total_iterations=10, output_folder=str(tmp_path))
        image_store = _create_image_store_with_images(tmp_path, 5)
        loop_img_path = os.path.join(tmp_path, 'looper.png')
        _make_test_image().save(loop_img_path)
        log_file = io.StringIO()

        loopsettings = _make_loopsettings()
        engine, sm = _make_mocks(loopsettings)
        _init_state_with_pre_elaborated(state, sm, loopsettings)

        # Apply sticky override to frame 2 (iteration 2)
        state.apply_frame_override(2, {'denoise_amt': 0.99, 'cfg': 12.0})

        _handle_restart(
            restart_from=3,  # This re-runs iteration 2
            engine=engine,
            sm=sm,
            total_iter=10,
            loop_img_path=loop_img_path,
            output_folder=str(tmp_path),
            log_file=log_file,
            state=state,
            image_store=image_store,
        )

        # The pre-elaborated settings for iter 2 should still have the overrides
        ls = state.get_pre_elaborated(2)
        assert ls.denoise_amt == 0.99
        assert ls.cfg == 12.0

    def test_overrides_persist_after_restart(self, tmp_path):
        """After restart, frame overrides should still be present (they are persistent)."""
        state = LoopState(total_iterations=10, output_folder=str(tmp_path))
        image_store = _create_image_store_with_images(tmp_path, 5)
        loop_img_path = os.path.join(tmp_path, 'looper.png')
        _make_test_image().save(loop_img_path)
        log_file = io.StringIO()

        loopsettings = _make_loopsettings()
        engine, sm = _make_mocks(loopsettings)
        _init_state_with_pre_elaborated(state, sm, loopsettings)

        state.apply_frame_override(2, {'denoise_amt': 0.5})

        _handle_restart(
            restart_from=3,
            engine=engine,
            sm=sm,
            total_iter=10,
            loop_img_path=loop_img_path,
            output_folder=str(tmp_path),
            log_file=log_file,
            state=state,
            image_store=image_store,
        )

        # Overrides should NOT be cleared — they are persistent
        assert state.is_field_overridden(2, 'denoise_amt')
        assert state.get_all_overridden_frames() == {2: ['denoise_amt']}

    def test_restart_without_overrides_still_works(self, tmp_path):
        """Restart with no pending overrides should work normally."""
        state = LoopState(total_iterations=10, output_folder=str(tmp_path))
        image_store = _create_image_store_with_images(tmp_path, 5)
        loop_img_path = os.path.join(tmp_path, 'looper.png')
        _make_test_image().save(loop_img_path)
        log_file = io.StringIO()

        loopsettings = _make_loopsettings(denoise_amt=0.7)
        engine, sm = _make_mocks(loopsettings)
        _init_state_with_pre_elaborated(state, sm, loopsettings)

        next_iter, _ = _handle_restart(
            restart_from=3,
            engine=engine,
            sm=sm,
            total_iter=10,
            loop_img_path=loop_img_path,
            output_folder=str(tmp_path),
            log_file=log_file,
            state=state,
            image_store=image_store,
        )

        assert next_iter == 3
        engine.compute_iteration.assert_called_once()

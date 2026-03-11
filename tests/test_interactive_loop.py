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

    return engine, sm


class TestHandleRestartPreservesFrameOverrides:
    """Regression: frame overrides set before restart must survive into _run_iteration."""

    def test_overrides_applied_during_restart(self, tmp_path):
        """Frame overrides set before _handle_restart should be applied to the loopsettings."""
        state = LoopState(total_iterations=10, output_folder=str(tmp_path))
        image_store = _create_image_store_with_images(tmp_path, 5)
        loop_img_path = os.path.join(tmp_path, 'looper.png')
        _make_test_image().save(loop_img_path)
        log_file = io.StringIO()

        # User sets overrides, then requests restart from frame 3
        state.set_frame_overrides({'denoise_amt': 0.99, 'cfg': 12.0})

        loopsettings = _make_loopsettings()
        engine, sm = _make_mocks(loopsettings)

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

        # The overrides should have been applied to loopsettings by _run_iteration
        assert loopsettings.denoise_amt == 0.99
        assert loopsettings.cfg == 12.0

    def test_overrides_cleared_after_consumption(self, tmp_path):
        """After _handle_restart, frame overrides should be empty (consumed by _run_iteration)."""
        state = LoopState(total_iterations=10, output_folder=str(tmp_path))
        image_store = _create_image_store_with_images(tmp_path, 5)
        loop_img_path = os.path.join(tmp_path, 'looper.png')
        _make_test_image().save(loop_img_path)
        log_file = io.StringIO()

        state.set_frame_overrides({'denoise_amt': 0.5})

        loopsettings = _make_loopsettings()
        engine, sm = _make_mocks(loopsettings)

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

        # Overrides should have been consumed (get_and_clear)
        assert state.get_frame_overrides() == {}

    def test_restart_without_overrides_still_works(self, tmp_path):
        """Restart with no pending overrides should work normally."""
        state = LoopState(total_iterations=10, output_folder=str(tmp_path))
        image_store = _create_image_store_with_images(tmp_path, 5)
        loop_img_path = os.path.join(tmp_path, 'looper.png')
        _make_test_image().save(loop_img_path)
        log_file = io.StringIO()

        # No overrides set
        loopsettings = _make_loopsettings(denoise_amt=0.7)
        engine, sm = _make_mocks(loopsettings)

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

        # Should restart from redo_iter + 1
        assert next_iter == 3
        # Engine should have been called
        engine.compute_iteration.assert_called_once()

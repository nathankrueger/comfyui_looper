import pytest
import os
import tempfile
from PIL import Image

from comfyui_looper.utils.image_store import FilesystemImageStore, ZipImageStore


def _make_test_image(width=64, height=64, color=(255, 0, 0)):
    return Image.new('RGB', (width, height), color)


@pytest.fixture(params=[FilesystemImageStore, ZipImageStore])
def store(request, tmp_path):
    s = request.param(str(tmp_path))
    yield s
    s.close()


def test_write_and_read(store):
    img = _make_test_image()
    store.write_image('test_001.png', img)
    assert store.has_image('test_001.png')

    result = store.read_image('test_001.png')
    assert result.size == (64, 64)


def test_read_bytes(store):
    img = _make_test_image()
    store.write_image('test_001.png', img)

    raw = store.read_image_bytes('test_001.png')
    assert isinstance(raw, bytes)
    assert len(raw) > 0
    # should be valid PNG (starts with PNG signature)
    assert raw[:4] == b'\x89PNG'


def test_list_images(store):
    store.write_image('img_002.png', _make_test_image(color=(0, 255, 0)))
    store.write_image('img_001.png', _make_test_image(color=(255, 0, 0)))
    store.write_image('img_003.png', _make_test_image(color=(0, 0, 255)))

    names = store.list_images()
    assert names == ['img_001.png', 'img_002.png', 'img_003.png']


def test_has_image(store):
    assert not store.has_image('nonexistent.png')
    store.write_image('exists.png', _make_test_image())
    assert store.has_image('exists.png')


def test_delete_images(store):
    store.write_image('img_001.png', _make_test_image())
    store.write_image('img_002.png', _make_test_image())
    store.write_image('img_003.png', _make_test_image())

    store.delete_images(['img_002.png', 'img_003.png'])

    assert store.has_image('img_001.png')
    assert not store.has_image('img_002.png')
    assert not store.has_image('img_003.png')
    assert store.list_images() == ['img_001.png']


def test_copy_image_to_path(store, tmp_path):
    img = _make_test_image(color=(128, 128, 128))
    store.write_image('source.png', img)

    dest = str(tmp_path / 'exported.png')
    store.copy_image_to_path('source.png', dest)
    assert os.path.exists(dest)

    loaded = Image.open(dest)
    assert loaded.size == (64, 64)


def test_import_from_path(store, tmp_path):
    # create an image on disk
    src_path = str(tmp_path / 'external.png')
    img = _make_test_image(color=(42, 42, 42))
    img.save(src_path)

    store.import_from_path(src_path, 'imported.png')
    assert store.has_image('imported.png')

    result = store.read_image('imported.png')
    assert result.size == (64, 64)


def test_get_paths_for_animation(store):
    store.write_image('img_001.png', _make_test_image())
    store.write_image('img_002.png', _make_test_image())

    folder, needs_cleanup = store.get_paths_for_animation()
    assert os.path.isdir(folder)

    pngs = sorted(os.listdir(folder))
    # should contain at least the two images (filesystem store may have other files)
    assert 'img_001.png' in pngs
    assert 'img_002.png' in pngs

    if needs_cleanup:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)

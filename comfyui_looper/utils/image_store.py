import os
import io
import glob
import shutil
import zipfile
import tempfile
import threading
from abc import ABC, abstractmethod
from PIL import Image


class ImageStore(ABC):
    """Abstraction for reading/writing numbered output images."""

    @abstractmethod
    def write_image(self, filename: str, image: Image.Image, png_info=None) -> None:
        """Write a PIL Image under the given filename."""

    @abstractmethod
    def read_image(self, filename: str) -> Image.Image:
        """Read an image by filename, returns PIL Image."""

    @abstractmethod
    def read_image_bytes(self, filename: str) -> bytes:
        """Read raw PNG bytes for serving over HTTP."""

    @abstractmethod
    def list_images(self) -> list[str]:
        """Return sorted list of image filenames."""

    @abstractmethod
    def has_image(self, filename: str) -> bool:
        """Check if an image exists."""

    @abstractmethod
    def delete_images(self, filenames: list[str]) -> None:
        """Delete the given image files."""

    @abstractmethod
    def copy_image_to_path(self, filename: str, dest_path: str) -> None:
        """Copy an image out to an arbitrary filesystem path (e.g. for restart -> looper.png)."""

    @abstractmethod
    def import_from_path(self, src_path: str, filename: str) -> None:
        """Import an image from a filesystem path into the store."""

    @abstractmethod
    def get_paths_for_animation(self) -> tuple[str, bool]:
        """Return (folder_path, needs_cleanup) with image files suitable for the animator.
        For filesystem, returns the output folder directly (needs_cleanup=False).
        For zip, extracts to a temp dir (needs_cleanup=True)."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""


class FilesystemImageStore(ImageStore):
    """Stores images as individual files in a directory."""

    def __init__(self, output_folder: str):
        self._output_folder = output_folder

    def _path(self, filename: str) -> str:
        return os.path.join(self._output_folder, filename)

    def write_image(self, filename: str, image: Image.Image, png_info=None) -> None:
        image.save(self._path(filename), pnginfo=png_info, compress_level=0)

    def read_image(self, filename: str) -> Image.Image:
        return Image.open(self._path(filename))

    def read_image_bytes(self, filename: str) -> bytes:
        with open(self._path(filename), 'rb') as f:
            return f.read()

    def list_images(self) -> list[str]:
        paths = sorted(glob.glob(os.path.join(self._output_folder, '*.png')))
        return [os.path.basename(p) for p in paths]

    def has_image(self, filename: str) -> bool:
        return os.path.exists(self._path(filename))

    def delete_images(self, filenames: list[str]) -> None:
        for filename in filenames:
            path = self._path(filename)
            if os.path.exists(path):
                os.remove(path)

    def copy_image_to_path(self, filename: str, dest_path: str) -> None:
        shutil.copy(self._path(filename), dest_path)

    def import_from_path(self, src_path: str, filename: str) -> None:
        dest = self._path(filename)
        if os.path.abspath(src_path) != os.path.abspath(dest):
            shutil.copy(src_path, dest)

    def get_paths_for_animation(self) -> tuple[str, bool]:
        return (self._output_folder, False)

    def close(self) -> None:
        pass


class ZipImageStore(ImageStore):
    """Stores images inside a zip file in the output folder."""

    ZIP_FILENAME = 'images.zip'

    def __init__(self, output_folder: str):
        self._output_folder = output_folder
        self._zip_path = os.path.join(output_folder, self.ZIP_FILENAME)
        self._lock = threading.Lock()
        self._temp_dir = None  # lazy-created for animation extraction

    def write_image(self, filename: str, image: Image.Image, png_info=None) -> None:
        buf = io.BytesIO()
        image.save(buf, format='PNG', pnginfo=png_info, compress_level=0)
        png_bytes = buf.getvalue()

        with self._lock:
            with zipfile.ZipFile(self._zip_path, 'a', compression=zipfile.ZIP_STORED) as zf:
                zf.writestr(filename, png_bytes)

    def read_image(self, filename: str) -> Image.Image:
        raw = self.read_image_bytes(filename)
        return Image.open(io.BytesIO(raw))

    def read_image_bytes(self, filename: str) -> bytes:
        with self._lock:
            with zipfile.ZipFile(self._zip_path, 'r') as zf:
                return zf.read(filename)

    def list_images(self) -> list[str]:
        with self._lock:
            if not os.path.exists(self._zip_path):
                return []
            with zipfile.ZipFile(self._zip_path, 'r') as zf:
                return sorted(zf.namelist())

    def has_image(self, filename: str) -> bool:
        with self._lock:
            if not os.path.exists(self._zip_path):
                return False
            with zipfile.ZipFile(self._zip_path, 'r') as zf:
                return filename in zf.namelist()

    def delete_images(self, filenames: list[str]) -> None:
        filenames_set = set(filenames)
        with self._lock:
            if not os.path.exists(self._zip_path):
                return
            temp_path = self._zip_path + '.tmp'
            with zipfile.ZipFile(self._zip_path, 'r') as zin:
                with zipfile.ZipFile(temp_path, 'w', compression=zipfile.ZIP_STORED) as zout:
                    for item in zin.infolist():
                        if item.filename not in filenames_set:
                            zout.writestr(item, zin.read(item.filename))
            os.replace(temp_path, self._zip_path)

    def copy_image_to_path(self, filename: str, dest_path: str) -> None:
        raw = self.read_image_bytes(filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            f.write(raw)

    def import_from_path(self, src_path: str, filename: str) -> None:
        with open(src_path, 'rb') as f:
            png_bytes = f.read()
        with self._lock:
            with zipfile.ZipFile(self._zip_path, 'a', compression=zipfile.ZIP_STORED) as zf:
                zf.writestr(filename, png_bytes)

    def get_paths_for_animation(self, progress_callback=None) -> tuple[str, bool]:
        """Extract all images to a temp directory for the animator."""
        temp_dir = tempfile.mkdtemp(prefix='looper_anim_')
        with self._lock:
            with zipfile.ZipFile(self._zip_path, 'r') as zf:
                members = zf.namelist()
                total = len(members)
                for i, member in enumerate(members):
                    zf.extract(member, temp_dir)
                    if progress_callback and total > 0:
                        progress_callback((i + 1) / total)
        return (temp_dir, True)

    def close(self) -> None:
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

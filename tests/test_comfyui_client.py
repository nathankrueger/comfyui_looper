import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

from utils.comfyui_client import ComfyUIClient


@pytest.fixture
def client():
    return ComfyUIClient("http://localhost:8188")


class TestCheckServer:
    def test_success(self, client):
        with patch("utils.comfyui_client.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            client.check_server()
            mock_get.assert_called_once_with("http://localhost:8188/system_stats", timeout=10)

    def test_connection_error(self, client):
        with patch("utils.comfyui_client.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            with pytest.raises(Exception):
                client.check_server()


class TestUploadImage:
    def test_upload_returns_filename(self, client):
        with patch("utils.comfyui_client.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"name": "looper_input_abc12345.png", "subfolder": "", "type": "input"}
            )
            with patch("builtins.open", mock_open(read_data=b"fake png data")):
                result = client.upload_image("/tmp/test.png", "test_upload.png")

            assert result == "looper_input_abc12345.png"
            mock_post.assert_called_once()

    def test_upload_generates_filename_if_none(self, client):
        with patch("utils.comfyui_client.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"name": "looper_input_generated.png"}
            )
            with patch("builtins.open", mock_open(read_data=b"fake png data")):
                result = client.upload_image("/tmp/test.png")

            assert result == "looper_input_generated.png"


class TestQueuePrompt:
    def test_returns_prompt_id(self, client):
        with patch("utils.comfyui_client.requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"prompt_id": "abc-123"}
            )
            prompt_id = client._queue_prompt({"1": {"class_type": "Test", "inputs": {}}})
            assert prompt_id == "abc-123"


class TestGetHistory:
    def test_returns_history_for_prompt(self, client):
        history_data = {
            "abc-123": {
                "outputs": {
                    "10": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
                }
            }
        }
        with patch("utils.comfyui_client.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: history_data
            )
            result = client.get_history("abc-123")
            assert "outputs" in result

    def test_returns_empty_for_unknown_prompt(self, client):
        with patch("utils.comfyui_client.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {}
            )
            result = client.get_history("unknown")
            assert result == {}


class TestGetOutputImages:
    def test_extracts_images_from_history(self, client):
        history = {
            "outputs": {
                "10": {"images": [
                    {"filename": "out_00001_.png", "subfolder": "", "type": "output"},
                ]},
                "5": {"text": ["some text"]},
            }
        }
        images = client.get_output_images(history)
        assert len(images) == 1
        assert images[0]["filename"] == "out_00001_.png"

    def test_empty_outputs(self, client):
        images = client.get_output_images({"outputs": {}})
        assert images == []

    def test_no_outputs_key(self, client):
        images = client.get_output_images({})
        assert images == []


class TestDownloadImage:
    def test_downloads_and_saves(self, client):
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]

        with patch("utils.comfyui_client.requests.get", return_value=mock_response) as mock_get:
            m = mock_open()
            with patch("builtins.open", m):
                client.download_image("out.png", "", "output", "/tmp/result.png")

            mock_get.assert_called_once()
            m.assert_called_once_with("/tmp/result.png", "wb")


class TestWaitForCompletion:
    def test_completes_on_executing_null_node(self, client):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [
            json.dumps({"type": "progress", "data": {"value": 5, "max": 20}}),
            json.dumps({"type": "executing", "data": {"prompt_id": "abc-123", "node": "5"}}),
            json.dumps({"type": "executing", "data": {"prompt_id": "abc-123", "node": None}}),
        ]

        with patch("utils.comfyui_client.websocket.create_connection", return_value=mock_ws):
            client._wait_for_completion("abc-123")

        mock_ws.close.assert_called_once()

    def test_raises_on_execution_error(self, client):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [
            json.dumps({
                "type": "execution_error",
                "data": {"prompt_id": "abc-123", "exception_message": "Node failed"}
            }),
        ]

        with patch("utils.comfyui_client.websocket.create_connection", return_value=mock_ws):
            with pytest.raises(RuntimeError, match="Node failed"):
                client._wait_for_completion("abc-123")

        mock_ws.close.assert_called_once()

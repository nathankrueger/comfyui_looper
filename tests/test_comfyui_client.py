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
        with patch("utils.comfyui_client.requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"name": "looper_input_abc12345.png", "subfolder": "", "type": "input"}
            )
            with patch("builtins.open", mock_open(read_data=b"fake png data")):
                result = client.upload_image("/tmp/test.png", "test_upload.png")

            assert result == "looper_input_abc12345.png"
            mock_request.assert_called_once()

    def test_upload_generates_filename_if_none(self, client):
        with patch("utils.comfyui_client.requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"name": "looper_input_generated.png"}
            )
            with patch("builtins.open", mock_open(read_data=b"fake png data")):
                result = client.upload_image("/tmp/test.png")

            assert result == "looper_input_generated.png"


class TestQueuePrompt:
    def test_returns_prompt_id(self, client):
        with patch("utils.comfyui_client.requests.request") as mock_request:
            mock_request.return_value = MagicMock(
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
        with patch("utils.comfyui_client.requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: history_data
            )
            result = client.get_history("abc-123")
            assert "outputs" in result

    def test_returns_empty_for_unknown_prompt(self, client):
        with patch("utils.comfyui_client.requests.request") as mock_request:
            mock_request.return_value = MagicMock(
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

        with patch("utils.comfyui_client.requests.request", return_value=mock_response) as mock_request:
            m = mock_open()
            with patch("builtins.open", m):
                client.download_image("out.png", "", "output", "/tmp/result.png")

            mock_request.assert_called_once()
            m.assert_called_once_with("/tmp/result.png", "wb")


class TestWaitForCompletion:
    def test_completes_on_executing_null_node(self, client):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [
            json.dumps({"type": "progress", "data": {"value": 5, "max": 20}}),
            json.dumps({"type": "executing", "data": {"prompt_id": "abc-123", "node": "5"}}),
            json.dumps({"type": "executing", "data": {"prompt_id": "abc-123", "node": None}}),
        ]

        client._wait_for_completion(mock_ws, "abc-123")

    def test_raises_on_execution_error(self, client):
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [
            json.dumps({
                "type": "execution_error",
                "data": {"prompt_id": "abc-123", "exception_message": "Node failed"}
            }),
        ]

        with pytest.raises(RuntimeError, match="Node failed"):
            client._wait_for_completion(mock_ws, "abc-123")

    def test_raises_on_websocket_timeout(self, client):
        """WebSocketTimeoutException now propagates raw (retryable by execute_workflow)."""
        import websocket as ws_module
        mock_ws = MagicMock()
        mock_ws.recv.side_effect = ws_module.WebSocketTimeoutException()

        with pytest.raises(ws_module.WebSocketTimeoutException):
            client._wait_for_completion(mock_ws, "abc-123")


class TestExecuteWorkflow:
    def test_connects_ws_before_queuing(self, client):
        """Verify websocket is connected before the prompt is submitted."""
        call_order = []

        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [
            json.dumps({"type": "executing", "data": {"prompt_id": "abc-123", "node": None}}),
        ]

        def mock_create_connection(*args, **kwargs):
            call_order.append("ws_connect")
            return mock_ws

        mock_response = MagicMock(
            status_code=200,
            json=lambda: {"prompt_id": "abc-123"}
        )

        def mock_request(method, url, **kwargs):
            if "prompt" in url:
                call_order.append("queue_prompt")
            elif "history" in url:
                call_order.append("get_history")
                mock_resp = MagicMock(status_code=200, json=lambda: {"abc-123": {"outputs": {}}})
                return mock_resp
            return mock_response

        with patch("utils.comfyui_client.websocket.create_connection", side_effect=mock_create_connection):
            with patch("utils.comfyui_client.requests.request", side_effect=mock_request):
                client.execute_workflow({"1": {"class_type": "Test", "inputs": {}}})

        assert call_order.index("ws_connect") < call_order.index("queue_prompt"), \
            "WebSocket must be connected before prompt is queued"


class TestRetryLogic:
    def test_retries_on_connection_error(self, client):
        import requests as req_module
        with patch("utils.comfyui_client.requests.request") as mock_request:
            mock_request.side_effect = [
                req_module.ConnectionError("Connection refused"),
                req_module.ConnectionError("Connection refused"),
                MagicMock(status_code=200, json=lambda: {"abc-123": {"outputs": {}}}),
            ]
            with patch("utils.comfyui_client.time.sleep"):
                result = client.get_history("abc-123")
            assert "outputs" in result
            assert mock_request.call_count == 3

    def test_fails_after_max_retries(self, client):
        import requests as req_module
        with patch("utils.comfyui_client.requests.request") as mock_request:
            mock_request.side_effect = req_module.ConnectionError("Connection refused")
            with patch("utils.comfyui_client.time.sleep"):
                with pytest.raises(ConnectionError, match="Failed to connect"):
                    client.get_history("abc-123")
            assert mock_request.call_count == 3

    def test_does_not_retry_on_http_error(self, client):
        import requests as req_module
        mock_resp = MagicMock(status_code=400)
        mock_resp.raise_for_status.side_effect = req_module.HTTPError("Bad Request")
        with patch("utils.comfyui_client.requests.request", return_value=mock_resp):
            with pytest.raises(req_module.HTTPError):
                client.get_history("abc-123")

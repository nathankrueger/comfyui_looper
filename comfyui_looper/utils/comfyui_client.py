import io
import json
import time
import uuid
import logging
import requests
import websocket

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 2


class ComfyUIClient:
    """HTTP/WebSocket client for communicating with a running ComfyUI server."""

    def __init__(self, server_url: str = "http://localhost:8188"):
        self.server_url = server_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def check_server(self):
        """Verify the ComfyUI server is reachable."""
        try:
            resp = requests.get(f"{self.server_url}/system_stats", timeout=10)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(f"Cannot connect to ComfyUI server at {self.server_url}")

    def upload_image(self, local_path: str, filename: str = None) -> str:
        """Upload an image to ComfyUI's input folder. Returns the server-side filename."""
        if filename is None:
            filename = f"looper_input_{uuid.uuid4().hex[:8]}.png"

        with open(local_path, "rb") as f:
            files = {"image": (filename, f, "image/png")}
            data = {"overwrite": "true"}
            resp = self._request_with_retry(
                "POST", f"{self.server_url}/upload/image", files=files, data=data, timeout=30
            )

        result = resp.json()
        return result["name"]

    def execute_workflow(self, workflow: dict) -> dict:
        """Submit a workflow, wait for completion via WebSocket, return output metadata.

        The websocket is connected BEFORE submitting the prompt to avoid a race
        condition where the completion message arrives before we start listening.

        If the WebSocket connection drops (e.g. laptop sleep), retries with backoff.
        """
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            ws = None
            try:
                ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
                ws = websocket.create_connection(
                    f"{ws_url}/ws?clientId={self.client_id}", timeout=600
                )
                prompt_id = self._queue_prompt(workflow)
                self._wait_for_completion(ws, prompt_id)
                return self.get_history(prompt_id)
            except (websocket.WebSocketException, OSError, ConnectionError) as e:
                last_exc = e
                logger.warning(
                    "WebSocket connection failed (attempt %d/%d): %s. Retrying...",
                    attempt, MAX_RETRIES, e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECS * attempt)
                    # Verify server is reachable before retrying
                    try:
                        self.check_server()
                    except ConnectionError:
                        pass  # Will retry anyway
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
        raise ConnectionError(
            f"Lost connection to ComfyUI after {MAX_RETRIES} attempts: {last_exc}"
        )

    def _queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow to the prompt queue. Returns the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        resp = self._request_with_retry(
            "POST", f"{self.server_url}/prompt", json=payload, timeout=30
        )
        return resp.json()["prompt_id"]

    def _wait_for_completion(self, ws, prompt_id: str):
        """Block until the given prompt finishes executing, using an already-connected WebSocket.

        Raises websocket.WebSocketException or OSError on connection loss (handled
        by execute_workflow's retry loop), and RuntimeError on ComfyUI execution errors
        or timeouts.
        """
        try:
            while True:
                message = json.loads(ws.recv())
                msg_type = message.get("type")

                if msg_type == "executing":
                    data = message.get("data", {})
                    if data.get("prompt_id") == prompt_id and data.get("node") is None:
                        # Execution complete
                        break

                elif msg_type == "execution_error":
                    data = message.get("data", {})
                    if data.get("prompt_id") == prompt_id:
                        raise RuntimeError(
                            f"ComfyUI execution error: {data.get('exception_message', 'unknown error')}"
                        )
        except websocket.WebSocketTimeoutException:
            raise RuntimeError(
                f"Timed out waiting for ComfyUI to complete prompt {prompt_id}. "
                "The server may be overloaded or unreachable."
            )
        except (websocket.WebSocketException, OSError):
            raise  # Let execute_workflow's retry loop handle connection errors

    def get_history(self, prompt_id: str) -> dict:
        """Get execution history/results for a prompt."""
        resp = self._request_with_retry(
            "GET", f"{self.server_url}/history/{prompt_id}", timeout=30
        )
        return resp.json().get(prompt_id, {})

    def download_image(self, filename: str, subfolder: str, image_type: str, save_path: str):
        """Download a generated image from ComfyUI server to a local path."""
        params = {"filename": filename, "subfolder": subfolder, "type": image_type}
        resp = self._request_with_retry(
            "GET", f"{self.server_url}/view", params=params, timeout=30, stream=True
        )

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    def get_output_images(self, history: dict) -> list[dict]:
        """Extract output image info from execution history."""
        images = []
        for node_id, node_output in history.get("outputs", {}).items():
            if "images" in node_output:
                for img in node_output["images"]:
                    images.append(img)
        return images

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an HTTP request with retry logic for transient network errors."""
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_SECS * attempt
                    logger.warning(
                        "Request to %s failed (attempt %d/%d): %s. Retrying in %ds...",
                        url, attempt, MAX_RETRIES, e, wait,
                    )
                    time.sleep(wait)
            except requests.HTTPError:
                raise  # Don't retry on 4xx/5xx
        raise ConnectionError(
            f"Failed to connect to ComfyUI after {MAX_RETRIES} attempts: {last_exc}"
        )

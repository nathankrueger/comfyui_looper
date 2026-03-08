import io
import json
import uuid
import requests
import websocket


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
            resp = requests.post(f"{self.server_url}/upload/image", files=files, data=data, timeout=30)
            resp.raise_for_status()

        result = resp.json()
        return result["name"]

    def execute_workflow(self, workflow: dict) -> dict:
        """Submit a workflow, wait for completion via WebSocket, return output metadata."""
        prompt_id = self._queue_prompt(workflow)
        self._wait_for_completion(prompt_id)
        return self.get_history(prompt_id)

    def _queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow to the prompt queue. Returns the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        resp = requests.post(f"{self.server_url}/prompt", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def _wait_for_completion(self, prompt_id: str):
        """Block until the given prompt finishes executing, using WebSocket."""
        ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
        ws = websocket.create_connection(f"{ws_url}/ws?clientId={self.client_id}", timeout=600)

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
        finally:
            ws.close()

    def get_history(self, prompt_id: str) -> dict:
        """Get execution history/results for a prompt."""
        resp = requests.get(f"{self.server_url}/history/{prompt_id}", timeout=30)
        resp.raise_for_status()
        return resp.json().get(prompt_id, {})

    def download_image(self, filename: str, subfolder: str, image_type: str, save_path: str):
        """Download a generated image from ComfyUI server to a local path."""
        params = {"filename": filename, "subfolder": subfolder, "type": image_type}
        resp = requests.get(f"{self.server_url}/view", params=params, timeout=30, stream=True)
        resp.raise_for_status()

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

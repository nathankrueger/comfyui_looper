from workflow.looper_workflow import WorkflowEngine
from workflow.api_engine import APIWorkflowEngine
from utils.comfyui_client import ComfyUIClient

ALL_WORKFLOWS = {"sdxl", "flux1d", "sd3.5"}

def create_workflow(name: str, client: ComfyUIClient) -> WorkflowEngine:
    if name not in ALL_WORKFLOWS:
        raise ValueError(f"Unknown workflow: {name}. Available: {ALL_WORKFLOWS}")
    return APIWorkflowEngine(model_type=name, client=client)

def get_all_workflows() -> set[str]:
    return set(ALL_WORKFLOWS)

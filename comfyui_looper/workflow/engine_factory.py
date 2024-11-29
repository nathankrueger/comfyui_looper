from workflow.looper_workflow import WorkflowEngine
from workflow.flux1d_engine import Flux1DWorkflowEngine
from workflow.sdxl_engine import SDXLWorkflowEngine

WORKFLOW_ENGINES = {
    Flux1DWorkflowEngine,
    SDXLWorkflowEngine
}
WORKFLOW_LIBRARY: dict[str, WorkflowEngine] = {c.get_name(): c for c in WORKFLOW_ENGINES}

def create_workflow(name: str) -> WorkflowEngine:
    return WORKFLOW_LIBRARY[name]()

def get_all_workflows() -> set[str]:
    result = set(WORKFLOW_LIBRARY.keys())
    return result
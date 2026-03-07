import pytest
from utils.json_spec import LoopSettings, LoraFilter, ConDelta, Canny
from workflow.api_workflow_builder import (
    build_sdxl_workflow,
    build_flux1d_workflow,
    build_sd3p5_workflow,
    SAVE_PREFIX,
)


def _make_basic_settings(**kwargs):
    """Create a minimal LoopSettings for testing."""
    defaults = {
        "loop_iterations": 1,
        "seed": 42,
        "prompt": "a psychedelic landscape",
        "neg_prompt": "text, watermark",
        "denoise_amt": 0.6,
        "denoise_steps": 20,
        "cfg": 8.0,
        "checkpoint": "sdXL_v10VAEFix.safetensors",
        "loras": [],
        "con_deltas": [],
        "transforms": [],
    }
    defaults.update(kwargs)
    ls = LoopSettings(**defaults)
    ls.offset = 0
    return ls


def _validate_workflow_graph(workflow: dict):
    """Validate that all node references in the workflow resolve to existing node IDs."""
    node_ids = set(workflow.keys())

    for node_id, node in workflow.items():
        assert "class_type" in node, f"Node {node_id} missing class_type"
        assert "inputs" in node, f"Node {node_id} missing inputs"

        for input_name, input_val in node["inputs"].items():
            if isinstance(input_val, list) and len(input_val) == 2 and isinstance(input_val[0], str):
                ref_id = input_val[0]
                assert ref_id in node_ids, (
                    f"Node {node_id} ({node['class_type']}) input '{input_name}' "
                    f"references non-existent node '{ref_id}'"
                )


def _find_node_by_class(workflow: dict, class_type: str) -> list[tuple[str, dict]]:
    """Find all nodes of a given class type."""
    return [(nid, n) for nid, n in workflow.items() if n["class_type"] == class_type]


class TestSDXLWorkflow:
    def test_basic_structure(self):
        settings = _make_basic_settings()
        workflow = build_sdxl_workflow("input.png", settings, {})

        _validate_workflow_graph(workflow)

        # Should have these key nodes
        class_types = {n["class_type"] for n in workflow.values()}
        assert "CheckpointLoaderSimple" in class_types
        assert "CLIPTextEncodeSDXL" in class_types
        assert "LoadImage" in class_types
        assert "VAEEncode" in class_types
        assert "KSampler" in class_types
        assert "VAEDecode" in class_types
        assert "SaveImage" in class_types

    def test_ksampler_params(self):
        settings = _make_basic_settings(seed=12345, denoise_amt=0.7, denoise_steps=25, cfg=7.5)
        workflow = build_sdxl_workflow("input.png", settings, {})

        samplers = _find_node_by_class(workflow, "KSampler")
        assert len(samplers) == 1
        inputs = samplers[0][1]["inputs"]
        assert inputs["seed"] == 12345
        assert inputs["denoise"] == 0.7
        assert inputs["steps"] == 25
        assert inputs["cfg"] == 7.5
        assert inputs["sampler_name"] == "euler"
        assert inputs["scheduler"] == "normal"

    def test_with_loras(self):
        loras = [
            LoraFilter(lora_path="style_lora.safetensors", lora_strength=0.8),
            LoraFilter(lora_path="detail_lora.safetensors", lora_strength=0.5),
        ]
        settings = _make_basic_settings(loras=loras)
        workflow = build_sdxl_workflow("input.png", settings, {})

        _validate_workflow_graph(workflow)

        lora_nodes = _find_node_by_class(workflow, "LoraLoader")
        assert len(lora_nodes) == 2
        assert lora_nodes[0][1]["inputs"]["lora_name"] == "style_lora.safetensors"
        assert lora_nodes[1][1]["inputs"]["lora_name"] == "detail_lora.safetensors"

    def test_with_canny(self):
        settings = _make_basic_settings(canny=Canny(low_thresh=100, high_thresh=200, strength=0.5))
        workflow = build_sdxl_workflow("input.png", settings, {})

        _validate_workflow_graph(workflow)

        canny_nodes = _find_node_by_class(workflow, "Canny")
        assert len(canny_nodes) == 1
        assert canny_nodes[0][1]["inputs"]["low_threshold"] == 100

        cn_apply = _find_node_by_class(workflow, "ControlNetApply")
        assert len(cn_apply) == 1

    def test_with_con_deltas(self):
        con_deltas = [
            ConDelta(pos="vibrant colors", neg="dull colors", strength=0.6),
        ]
        settings = _make_basic_settings(con_deltas=con_deltas)
        workflow = build_sdxl_workflow("input.png", settings, {})

        _validate_workflow_graph(workflow)

        subtract_nodes = _find_node_by_class(workflow, "ConditioningSubtract")
        assert len(subtract_nodes) == 1
        apply_nodes = _find_node_by_class(workflow, "ConditioningAddConDelta")
        assert len(apply_nodes) == 1
        assert apply_nodes[0][1]["inputs"]["conditioning_delta_strength"] == 0.6

    def test_save_node_has_prefix(self):
        settings = _make_basic_settings()
        workflow = build_sdxl_workflow("input.png", settings, {})

        save_nodes = _find_node_by_class(workflow, "SaveImage")
        assert len(save_nodes) == 1
        assert save_nodes[0][1]["inputs"]["filename_prefix"] == SAVE_PREFIX

    def test_load_image_uses_input_name(self):
        settings = _make_basic_settings()
        workflow = build_sdxl_workflow("my_upload.png", settings, {})

        load_nodes = _find_node_by_class(workflow, "LoadImage")
        assert len(load_nodes) == 1
        assert load_nodes[0][1]["inputs"]["image"] == "my_upload.png"


class TestFlux1DWorkflow:
    def test_basic_structure(self):
        settings = _make_basic_settings(
            clip=["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors"],
            checkpoint="flux1-dev-fp8.safetensors",
        )
        workflow = build_flux1d_workflow("input.png", settings, {})

        _validate_workflow_graph(workflow)

        class_types = {n["class_type"] for n in workflow.values()}
        assert "UNETLoader" in class_types
        assert "DualCLIPLoader" in class_types
        assert "VAELoader" in class_types
        assert "CLIPTextEncode" in class_types
        assert "FluxGuidance" in class_types
        assert "BasicGuider" in class_types
        assert "RandomNoise" in class_types
        assert "KSamplerSelect" in class_types
        assert "BasicScheduler" in class_types
        assert "SamplerCustomAdvanced" in class_types
        assert "VAEDecode" in class_types
        assert "SaveImage" in class_types

    def test_dual_clip_params(self):
        settings = _make_basic_settings(
            clip=["t5xxl.safetensors", "clip_l.safetensors"],
            checkpoint="flux.safetensors",
        )
        workflow = build_flux1d_workflow("input.png", settings, {})

        clip_nodes = _find_node_by_class(workflow, "DualCLIPLoader")
        assert len(clip_nodes) == 1
        inputs = clip_nodes[0][1]["inputs"]
        assert inputs["clip_name1"] == "t5xxl.safetensors"
        assert inputs["clip_name2"] == "clip_l.safetensors"
        assert inputs["type"] == "flux"

    def test_scheduler_uses_beta(self):
        settings = _make_basic_settings(checkpoint="flux.safetensors", denoise_amt=0.65)
        workflow = build_flux1d_workflow("input.png", settings, {})

        scheduler_nodes = _find_node_by_class(workflow, "BasicScheduler")
        assert len(scheduler_nodes) == 1
        assert scheduler_nodes[0][1]["inputs"]["scheduler"] == "beta"


class TestSD3p5Workflow:
    def test_basic_structure(self):
        settings = _make_basic_settings(
            checkpoint="sd3.5_large.safetensors",
            clip=["t5xxl_fp8_e4m3fn.safetensors", "clip_g.safetensors"],
        )
        workflow = build_sd3p5_workflow("input.png", settings, {})

        _validate_workflow_graph(workflow)

        class_types = {n["class_type"] for n in workflow.values()}
        assert "CheckpointLoaderSimple" in class_types
        assert "DualCLIPLoader" in class_types
        assert "CLIPTextEncode" in class_types
        assert "KSampler" in class_types
        assert "VAEDecode" in class_types
        assert "SaveImage" in class_types

    def test_ksampler_uses_beta_scheduler(self):
        settings = _make_basic_settings(
            checkpoint="sd3.5_large.safetensors",
            clip=["t5xxl.safetensors", "clip_g.safetensors"],
        )
        workflow = build_sd3p5_workflow("input.png", settings, {})

        samplers = _find_node_by_class(workflow, "KSampler")
        assert len(samplers) == 1
        assert samplers[0][1]["inputs"]["scheduler"] == "beta"

    def test_with_loras(self):
        settings = _make_basic_settings(
            checkpoint="sd3.5_large.safetensors",
            clip=["t5xxl.safetensors", "clip_g.safetensors"],
            loras=[LoraFilter(lora_path="sd3_lora.safetensors", lora_strength=0.7)],
        )
        workflow = build_sd3p5_workflow("input.png", settings, {})

        _validate_workflow_graph(workflow)

        lora_nodes = _find_node_by_class(workflow, "LoraLoader")
        assert len(lora_nodes) == 1
        assert lora_nodes[0][1]["inputs"]["strength_model"] == 0.7

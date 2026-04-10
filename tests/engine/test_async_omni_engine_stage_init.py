import os
import types

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_initialize_stages_restores_device_visibility_after_diffusion_init(monkeypatch):
    """Regression test for stage device env leakage across stage init.

    Diffusion init mutates process-level CUDA visibility. Ensure AsyncOmniEngine
    restores the previous value after diffusion stage setup.
    """
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.platforms import current_omni_platform

    engine = object.__new__(AsyncOmniEngine)
    engine.model = "dummy-model"
    engine.config_path = "dummy-config"
    engine.num_stages = 1
    engine.async_chunk = False
    engine.diffusion_batch_size = 1
    engine.stage_configs = [types.SimpleNamespace(stage_id=0, stage_type="diffusion")]

    env_var = current_omni_platform.device_control_env_var
    old_env = os.environ.get(env_var)
    os.environ[env_var] = "0,1"

    diffusion_client = types.SimpleNamespace(is_comprehension=False)

    metadata = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        runtime_cfg={"devices": "1"},
        prompt_expand_func=None,
    )

    monkeypatch.setattr(engine_mod, "prepare_engine_environment", lambda: None)
    monkeypatch.setattr(engine_mod, "load_omni_transfer_config_for_model", lambda *_: None)
    monkeypatch.setattr(engine_mod, "extract_stage_metadata", lambda _cfg: metadata)
    monkeypatch.setattr(engine_mod, "get_stage_connector_spec", lambda **_: {})
    monkeypatch.setattr(engine_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))

    def _fake_setup_stage_devices(_stage_id, _runtime_cfg):
        # Simulate diffusion setup mutating process-global visibility.
        current_omni_platform.set_device_control_env_var("1")

    monkeypatch.setattr(engine_mod, "setup_stage_devices", _fake_setup_stage_devices)
    monkeypatch.setattr(engine_mod, "_inject_kv_stage_info", lambda *_: None)
    monkeypatch.setattr(engine_mod, "initialize_diffusion_stage", lambda *_, **__: diffusion_client)
    monkeypatch.setattr(
        engine_mod,
        "finalize_initialized_stages",
        lambda stage_clients, _input_processor: (
            stage_clients,
            [types.SimpleNamespace()],
            [{"final_output_type": "image"}],
        ),
    )

    try:
        engine._initialize_stages(stage_init_timeout=1)
        assert os.environ.get(env_var) == "0,1"
    finally:
        if old_env is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = old_env


def test_attach_llm_stage_uses_omni_input_preprocessor(monkeypatch):
    """Regression test for GLM-Image t2i preprocessing path.

    Stage-0 InputProcessor must use OmniInputPreprocessor so text prompts with
    mm_processor_kwargs go through multimodal preprocessing.
    """
    import vllm_omni.engine.async_omni_engine as engine_mod

    class DummyStageEngineCoreClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def shutdown(self):
            return None

    class DummyOutputProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyInputProcessor:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.renderer = object()
            self.input_preprocessor = object()

    class DummyOmniInputPreprocessor:
        def __init__(self, vllm_config, renderer=None):
            self.vllm_config = vllm_config
            self.renderer = renderer

    monkeypatch.setattr(engine_mod, "StageEngineCoreClient", DummyStageEngineCoreClient)
    monkeypatch.setattr(engine_mod, "MultimodalOutputProcessor", DummyOutputProcessor)
    monkeypatch.setattr(engine_mod, "InputProcessor", DummyInputProcessor)
    monkeypatch.setattr(engine_mod, "OmniInputPreprocessor", DummyOmniInputPreprocessor)

    started = types.SimpleNamespace(
        stage_id=0,
        metadata=types.SimpleNamespace(stage_id=0, engine_output_type="token_ids"),
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(skip_tokenizer_init=True)),
        executor_class=object,
        engine_manager=object(),
        coordinator=object(),
        proc=None,
        addresses=types.SimpleNamespace(
            inputs=["inproc://input"],
            outputs=["inproc://output"],
            frontend_stats_publish_address=None,
        ),
    )

    engine = object.__new__(AsyncOmniEngine)

    _stage_client, _out_proc, _vllm_cfg, input_processor = engine._attach_llm_stage(started)

    assert input_processor is not None
    assert isinstance(input_processor.input_preprocessor, DummyOmniInputPreprocessor)
    assert input_processor.input_preprocessor.renderer is input_processor.renderer


def test_initialize_diffusion_stage_populates_multimodal_limits(monkeypatch):
    """Diffusion serving should expose enriched image-count limits to the API layer."""
    import vllm_omni.engine.stage_init_utils as init_mod
    from vllm_omni.engine.stage_init_utils import StageMetadata, initialize_diffusion_stage

    captured = {}

    class DummyStageDiffusionClient:
        def __init__(self, model, od_config, metadata, batch_size=1):
            captured["model"] = model
            captured["od_config"] = od_config
            captured["metadata"] = metadata
            captured["batch_size"] = batch_size

    monkeypatch.setattr(
        "vllm_omni.diffusion.stage_diffusion_client.StageDiffusionClient",
        DummyStageDiffusionClient,
    )
    monkeypatch.setattr(
        init_mod,
        "get_hf_file_to_dict",
        lambda filename, _model: {"_class_name": "QwenImageEditPlusPipeline"} if filename == "model_index.json" else None,
    )

    metadata = StageMetadata(
        stage_id=0,
        stage_type="diffusion",
        engine_output_type="image",
        is_comprehension=False,
        requires_multimodal_data=False,
        engine_input_source=[],
        final_output=True,
        final_output_type="image",
        default_sampling_params=types.SimpleNamespace(),
        custom_process_input_func=None,
        model_stage=None,
        runtime_cfg=None,
    )
    stage_cfg = types.SimpleNamespace(engine_args={})

    initialize_diffusion_stage("Qwen/Qwen-Image-Edit-2511", stage_cfg, metadata, batch_size=1)

    assert captured["model"] == "Qwen/Qwen-Image-Edit-2511"
    assert captured["od_config"].model_class_name == "QwenImageEditPlusPipeline"
    assert captured["od_config"].supports_multimodal_inputs is True
    assert captured["od_config"].max_input_images == 3

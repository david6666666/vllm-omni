# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Callable, cast

import torch
from torch import nn
from huggingface_hub import hf_hub_download
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import (
    download_gguf,
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    gguf_quant_weights_iterator,
    maybe_download_from_modelscope,
    safetensors_weights_iterator,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.registry import initialize_model

logger = init_logger(__name__)


MODEL_INDEX = "model_index.json"
DIFFUSION_MODEL_WEIGHTS_INDEX = "diffusion_pytorch_model.safetensors.index.json"


class DiffusersPipelineLoader:
    """Model loader that can load diffusers pipeline components from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class ComponentSource:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        subfolder: str | None
        """The subfolder inside the model repo."""

        revision: str | None
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

        # TODO(Isotr0py): Enable multithreaded weight loading
        # extra_config = load_config.model_loader_extra_config
        # allowed_keys = {"enable_multithread_load", "num_threads"}
        # unexpected_keys = set(extra_config.keys()) - allowed_keys

        # if unexpected_keys:
        #     raise ValueError(
        #         f"Unexpected extra config keys for load format {load_config.load_format}: {unexpected_keys}"
        #     )

    def _prepare_weights(
        self,
        model_name_or_path: Path,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = maybe_download_from_modelscope(model_name_or_path, revision) or model_name_or_path

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = DIFFUSION_MODEL_WEIGHTS_INDEX
        index_file_with_subfolder = f"{subfolder}/{index_file}" if subfolder else index_file

        # only hf is supported currently
        if load_format == "auto":
            load_format = "hf"

        # Some quantized models use .pt files for storing the weights.
        if load_format == "hf":
            allow_patterns = ["*.safetensors", "*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        if subfolder is not None:
            allow_patterns = [f"{subfolder}/{pattern}" for pattern in allow_patterns]

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                # Decide by actual files rather than pattern name (patterns may include subfolders).
                use_safetensors = any(f.endswith(".safetensors") for f in hf_weights_files)
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file_with_subfolder,
                    self.load_config.download_dir,
                    revision,
                )
            # Some diffusers pipelines keep component weights under a
            # subfolder (e.g. "transformer/") and the corresponding index file
            # uses filenames relative to that subfolder. vLLM's
            # `filter_duplicate_safetensors_files` expects weight_map entries
            # to be relative to the `hf_folder` we pass in, so we point it to
            # the component subfolder to avoid filtering out all shards.
            filter_folder = os.path.join(hf_folder, subfolder) if subfolder is not None else hf_folder
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files,
                filter_folder,
                index_file,
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(self, source: "ComponentSource") -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        weights_iterator = safetensors_weights_iterator(
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
            self.load_config.safetensors_load_strategy,
        )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def get_all_weights(
        self,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        sources = cast(
            Iterable[DiffusersPipelineLoader.ComponentSource],
            getattr(model, "weights_sources", ()),
        )
        for source in sources:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_name_or_path=model_config.model,
            subfolder=None,
            revision=model_config.revision,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )

    def load_model(self, od_config: OmniDiffusionConfig, load_device: str) -> nn.Module:
        """Load a model with the given configurations."""
        target_device = torch.device(load_device)
        with set_default_torch_dtype(od_config.dtype):
            with target_device:
                model = initialize_model(od_config)

            logger.debug("Loading weights on %s ...", load_device)
            if self._is_gguf_quantization(od_config):
                self._load_weights_with_gguf(model, od_config)
            else:
                # Quantization does not happen in `load_weights` but after it
                self.load_weights(model)

            # Process weights after loading for quantization (e.g., FP8 online quantization)
            # This is needed for vLLM's quantization methods that need to transform weights
            self._process_weights_after_loading(model, target_device)

        return model.eval()

    def _process_weights_after_loading(self, model: nn.Module, target_device: torch.device) -> None:
        """Process weights after loading for quantization methods.

        This handles vLLM's quantization methods that need to process weights
        after loading (e.g., FP8 online quantization from BF16/FP16 weights).
        """
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                # Move module to target device for processing if needed
                module_device = next(module.parameters(), None)
                if module_device is not None:
                    module_device = module_device.device
                needs_device_move = module_device != target_device

                if needs_device_move:
                    module.to(target_device)

                quant_method.process_weights_after_loading(module)

                if needs_device_move:
                    module.to(module_device)

    def load_weights(self, model: nn.Module) -> None:
        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(self.get_all_weights(model))

        self.counter_after_loading_weights = time.perf_counter()
        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
        # TODO(Isotr0py): Enable weights loading check after decoupling
        # all components' weights loading (AutoModel.from_pretrained etc).
        # We only enable strict check for non-quantized models
        # that have loaded weights tracking currently.
        if loaded_weights is not None:
            _ = weights_to_load - loaded_weights
        #     if weights_not_loaded:
        #         raise ValueError(
        #             "Following weights were not initialized from "
        #             f"checkpoint: {weights_not_loaded}"
        #         )

    def _is_gguf_quantization(self, od_config: OmniDiffusionConfig) -> bool:
        quant_config = od_config.quantization_config
        if quant_config is None:
            return False
        try:
            is_gguf = quant_config.get_name() == "gguf"
        except Exception:
            return False
        if not is_gguf:
            return False
        gguf_model = getattr(quant_config, "gguf_model", None)
        if gguf_model is None:
            raise ValueError("GGUF quantization requires quantization_config.gguf_model")
        return True

    def _is_transformer_source(self, source: "ComponentSource") -> bool:
        if source.subfolder == "transformer":
            return True
        return source.prefix.startswith("transformer.")

    def _get_model_loadable_names(self, model: nn.Module) -> set[str]:
        # Use state_dict keys to include both parameters and buffers.
        return set(model.state_dict().keys())

    def _resolve_gguf_model_path(self, gguf_model: str, revision: str | None) -> str:
        if os.path.isfile(gguf_model):
            return gguf_model
        # raw HTTPS link
        if gguf_model.startswith(("http://", "https://")) and gguf_model.endswith(".gguf"):
            return hf_hub_download(url=gguf_model)
        # repo_id/filename.gguf
        if "/" in gguf_model and gguf_model.endswith(".gguf"):
            repo_id, filename = gguf_model.rsplit("/", 1)
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=self.load_config.download_dir,
            )
        # repo_id:quant_type
        if "/" in gguf_model and ":" in gguf_model:
            repo_id, quant_type = gguf_model.rsplit(":", 1)
            return download_gguf(
                repo_id,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        raise ValueError(
            f"Unrecognized GGUF reference: {gguf_model!r} (expected local file, "
            "raw URL, <repo_id>/<filename>.gguf, or <repo_id>:<quant_type>)"
        )

    def _get_gguf_name_mapper(self, od_config: OmniDiffusionConfig) -> Callable[[str], str | None]:
        model_class = od_config.model_class_name
        if model_class in {
            "QwenImagePipeline",
            "QwenImageEditPipeline",
            "QwenImageEditPlusPipeline",
            "QwenImageLayeredPipeline",
        }:
            return lambda name: name
        if model_class == "Flux2KleinPipeline":
            return self._map_flux2_klein_gguf_name
        raise ValueError(f"GGUF mapping is not implemented for model_class_name={model_class!r}")

    @staticmethod
    def _map_flux2_klein_gguf_name(name: str) -> str | None:
        if name.startswith("double_stream_modulation_img.lin."):
            return name.replace("double_stream_modulation_img.lin.", "double_stream_modulation_img.linear.", 1)
        if name.startswith("double_stream_modulation_txt.lin."):
            return name.replace("double_stream_modulation_txt.lin.", "double_stream_modulation_txt.linear.", 1)
        if name.startswith("single_stream_modulation.lin."):
            return name.replace("single_stream_modulation.lin.", "single_stream_modulation.linear.", 1)
        if name.startswith("img_in."):
            return name.replace("img_in.", "x_embedder.", 1)
        if name.startswith("txt_in."):
            return name.replace("txt_in.", "context_embedder.", 1)
        if name.startswith("time_in.in_layer."):
            return name.replace(
                "time_in.in_layer.",
                "time_guidance_embed.timestep_embedder.linear_1.",
                1,
            )
        if name.startswith("time_in.out_layer."):
            return name.replace(
                "time_in.out_layer.",
                "time_guidance_embed.timestep_embedder.linear_2.",
                1,
            )
        if name.startswith("final_layer.adaLN_modulation.1."):
            return name.replace("final_layer.adaLN_modulation.1.", "norm_out.linear.", 1)
        if name.startswith("final_layer.linear."):
            return name.replace("final_layer.linear.", "proj_out.", 1)

        if name.startswith("double_blocks."):
            name = name.replace("double_blocks.", "transformer_blocks.", 1)
            if ".img_attn.qkv." in name:
                return name.replace(".img_attn.qkv.", ".attn.to_qkv.", 1)
            if ".img_attn.proj." in name:
                return name.replace(".img_attn.proj.", ".attn.to_out.0.", 1)
            if name.endswith(".img_attn.norm.query_norm.scale"):
                return name.replace(".img_attn.norm.query_norm.scale", ".attn.norm_q.weight", 1)
            if name.endswith(".img_attn.norm.key_norm.scale"):
                return name.replace(".img_attn.norm.key_norm.scale", ".attn.norm_k.weight", 1)
            if ".txt_attn.qkv." in name:
                return name.replace(".txt_attn.qkv.", ".attn.add_kv_proj.", 1)
            if ".txt_attn.proj." in name:
                return name.replace(".txt_attn.proj.", ".attn.to_add_out.", 1)
            if name.endswith(".txt_attn.norm.query_norm.scale"):
                return name.replace(".txt_attn.norm.query_norm.scale", ".attn.norm_added_q.weight", 1)
            if name.endswith(".txt_attn.norm.key_norm.scale"):
                return name.replace(".txt_attn.norm.key_norm.scale", ".attn.norm_added_k.weight", 1)
            if ".img_mlp.0." in name:
                return name.replace(".img_mlp.0.", ".ff.linear_in.", 1)
            if ".img_mlp.2." in name:
                return name.replace(".img_mlp.2.", ".ff.linear_out.", 1)
            if ".txt_mlp.0." in name:
                return name.replace(".txt_mlp.0.", ".ff_context.linear_in.", 1)
            if ".txt_mlp.2." in name:
                return name.replace(".txt_mlp.2.", ".ff_context.linear_out.", 1)
            return None

        if name.startswith("single_blocks."):
            name = name.replace("single_blocks.", "single_transformer_blocks.", 1)
            if ".linear1." in name:
                return name.replace(".linear1.", ".attn.to_qkv_mlp_proj.", 1)
            if ".linear2." in name:
                return name.replace(".linear2.", ".attn.to_out.", 1)
            if name.endswith(".norm.query_norm.scale"):
                return name.replace(".norm.query_norm.scale", ".attn.norm_q.weight", 1)
            if name.endswith(".norm.key_norm.scale"):
                return name.replace(".norm.key_norm.scale", ".attn.norm_k.weight", 1)
            return None

        return None

    def _build_gguf_name_map(
        self,
        gguf_file: str,
        od_config: OmniDiffusionConfig,
    ) -> dict[str, str]:
        try:
            import gguf  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "GGUF support requires the 'gguf' package to be installed."
            ) from exc

        mapper = self._get_gguf_name_mapper(od_config)
        reader = gguf.GGUFReader(gguf_file)
        gguf_to_model_map: dict[str, str] = {}
        for tensor in reader.tensors:
            mapped = mapper(tensor.name)
            if mapped is None:
                continue
            gguf_to_model_map[tensor.name] = mapped
        if not gguf_to_model_map:
            raise RuntimeError(
                f"No GGUF tensors were mapped for model_class_name={od_config.model_class_name!r}."
            )
        return gguf_to_model_map

    def _get_gguf_weights_iterator(
        self,
        source: "ComponentSource",
        od_config: OmniDiffusionConfig,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        quant_config = od_config.quantization_config
        gguf_model = getattr(quant_config, "gguf_model", None)
        if gguf_model is None:
            raise ValueError("GGUF quantization requires quantization_config.gguf_model")
        gguf_file = self._resolve_gguf_model_path(gguf_model, od_config.revision)
        gguf_name_map = self._build_gguf_name_map(gguf_file, od_config)
        weights_iter = gguf_quant_weights_iterator(gguf_file, gguf_name_map)
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iter)

    def _load_weights_with_gguf(self, model: nn.Module, od_config: OmniDiffusionConfig) -> set[str]:
        sources = cast(
            Iterable[DiffusersPipelineLoader.ComponentSource],
            getattr(model, "weights_sources", ()),
        )
        loaded: set[str] = set()
        loadable_names: set[str] | None = None

        for source in sources:
            if self._is_transformer_source(source):
                loaded |= model.load_weights(self._get_gguf_weights_iterator(source, od_config))

                # Load any remaining float weights (e.g., non-quantized layers)
                # from the base HF checkpoint while skipping already-loaded names.
                loadable_names = loadable_names or self._get_model_loadable_names(model)
                hf_iter = self._get_weights_iterator(source)
                hf_iter = (
                    (name, tensor)
                    for (name, tensor) in hf_iter
                    if name in loadable_names and name not in loaded
                )
                loaded |= model.load_weights(hf_iter)
            else:
                loaded |= model.load_weights(self._get_weights_iterator(source))
        return loaded

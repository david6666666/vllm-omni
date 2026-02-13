# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import shutil
import tempfile
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import cast
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from huggingface_hub import hf_hub_download
from torch import nn
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
    maybe_download_from_modelscope,
    safetensors_weights_iterator,
)
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.gguf_adapters import get_gguf_adapter
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

    def load_model(
        self,
        od_config: OmniDiffusionConfig,
        load_device: str,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        target_device = torch.device(load_device)
        with set_default_torch_dtype(od_config.dtype):
            with target_device:
                if load_format == "default":
                    model = initialize_model(od_config)
                elif load_format == "custom_pipeline":
                    model_cls = resolve_obj_by_qualname(custom_pipeline_name)
                    model = model_cls(od_config=od_config)
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
        # Fast path: mapping-style config (e.g., DictConfig)
        if isinstance(quant_config, dict):
            method = str(quant_config.get("method", "")).lower()
            if method != "gguf":
                return False
            gguf_model = quant_config.get("gguf_model")
            if not gguf_model:
                raise ValueError("GGUF quantization requires quantization_config.gguf_model")
            return True

        # Normal path: DiffusionQuantizationConfig
        try:
            is_gguf = quant_config.get_name() == "gguf"
        except AttributeError:
            # Fallback: if it carries gguf_model, treat as GGUF
            gguf_model = getattr(quant_config, "gguf_model", None)
            return bool(gguf_model)
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
        # Avoid model.state_dict() here because GGUF uses UninitializedParameter
        # which raises during detach(). Collect names directly.
        names = {name for name, _ in model.named_parameters()}
        names.update(name for name, _ in model.named_buffers())
        return names

    def _resolve_gguf_model_path(self, gguf_model: str, revision: str | None) -> str:
        if os.path.isfile(gguf_model):
            return gguf_model
        # raw HTTPS link
        if gguf_model.startswith(("http://", "https://")) and gguf_model.endswith(".gguf"):
            return self._download_raw_gguf_url(gguf_model)
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

    def _download_raw_gguf_url(self, url: str) -> str:
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename:
            raise ValueError(f"Cannot infer GGUF filename from URL: {url!r}")

        cache_dir = self.load_config.download_dir
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "vllm-omni",
                "gguf",
            )
        os.makedirs(cache_dir, exist_ok=True)
        target_path = os.path.join(cache_dir, filename)
        if os.path.exists(target_path):
            return target_path

        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".gguf",
            prefix="gguf-",
            dir=cache_dir,
        )
        os.close(tmp_fd)
        try:
            with urlopen(url) as response, open(tmp_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            os.replace(tmp_path, target_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        return target_path

    def _get_gguf_weights_iterator(
        self,
        source: "ComponentSource",
        model: nn.Module,
        od_config: OmniDiffusionConfig,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        quant_config = od_config.quantization_config
        gguf_model = getattr(quant_config, "gguf_model", None)
        if gguf_model is None:
            raise ValueError("GGUF quantization requires quantization_config.gguf_model")
        gguf_file = self._resolve_gguf_model_path(gguf_model, od_config.revision)
        adapter = get_gguf_adapter(gguf_file, model, source, od_config)
        weights_iter = adapter.weights_iterator()
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
                loaded |= model.load_weights(self._get_gguf_weights_iterator(source, model, od_config))

                # Load any remaining float weights (e.g., non-quantized layers)
                # from the base HF checkpoint while skipping already-loaded names.
                loadable_names = loadable_names or self._get_model_loadable_names(model)
                hf_iter = self._get_weights_iterator(source)
                hf_iter = (
                    (name, tensor) for (name, tensor) in hf_iter if name in loadable_names and name not in loaded
                )
                loaded |= model.load_weights(hf_iter)
            else:
                loaded |= model.load_weights(self._get_weights_iterator(source))
        return loaded

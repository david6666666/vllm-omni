# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import glob
import os
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import cast
import torch
from torch import nn
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_gguf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    gguf,
    maybe_download_from_modelscope,
    safetensors_weights_iterator,
    gguf_quant_weights_iterator,
)
from vllm.utils.torch_utils import set_default_torch_dtype
from huggingface_hub import hf_hub_download

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
        self.quantized_weights: str | None = None
        self.quantization_scope: str = "transformer_only"

        # TODO(Isotr0py): Enable multithreaded weight loading
        # extra_config = load_config.model_loader_extra_config
        # allowed_keys = {"enable_multithread_load", "num_threads"}
        # unexpected_keys = set(extra_config.keys()) - allowed_keys

        # if unexpected_keys:
        #     raise ValueError(
        #         f"Unexpected extra config keys for load format {load_config.load_format}: {unexpected_keys}"
        #     )

    def _is_transformer_source(self, source: "ComponentSource") -> bool:
        if source.subfolder is not None and source.subfolder.startswith("transformer"):
            return True
        return source.prefix.startswith("transformer.")

    def _uses_external_quantized_source(self, source: "ComponentSource") -> bool:
        if self.quantized_weights is None:
            return False
        if self.quantization_scope != "transformer_only":
            return False
        return self._is_transformer_source(source)

    def _resolve_component_source(self, source: "ComponentSource") -> "ComponentSource":
        if not self._uses_external_quantized_source(source):
            return source
        return dataclasses.replace(source, model_or_path=self.quantized_weights)

    @staticmethod
    def _normalize_gguf_tensor_name(name: str) -> str:
        """Best-effort normalization from GGUF/legacy names to model param names."""
        replacements = [
            # FLUX.2 legacy naming
            ("double_blocks.", "transformer_blocks."),
            ("single_blocks.", "single_transformer_blocks."),
            (".img_attn.norm.query_norm.scale", ".attn.norm_q.weight"),
            (".img_attn.norm.key_norm.scale", ".attn.norm_k.weight"),
            (".txt_attn.norm.query_norm.scale", ".attn.norm_added_q.weight"),
            (".txt_attn.norm.key_norm.scale", ".attn.norm_added_k.weight"),
            (".txt_attn.to_q.", ".attn.add_q_proj."),
            (".txt_attn.to_k.", ".attn.add_k_proj."),
            (".txt_attn.to_v.", ".attn.add_v_proj."),
            (".txt_attn.to_out.", ".attn.to_add_out."),
            (".img_attn.to_out.", ".attn.to_out.0."),
            (".img_attn.", ".attn."),
            (".txt_attn.", ".attn."),
        ]
        for src, dst in replacements:
            name = name.replace(src, dst)

        if name.endswith(".scale"):
            name = name[:-6] + ".weight"
        if ".to_out." in name and ".to_out.0." not in name:
            name = name.replace(".to_out.", ".to_out.0.")
        return name

    def _candidate_hf_param_names(self, gguf_name: str, prefix: str) -> list[str]:
        """Generate candidate model parameter names for a GGUF tensor name."""
        candidates: list[str] = []

        def _add(cand: str) -> None:
            if cand and cand not in candidates:
                candidates.append(cand)

        seeds = [gguf_name, self._normalize_gguf_tensor_name(gguf_name)]
        for seed in list(seeds):
            for lead in ("model.", "diffusion_model.", "transformer."):
                if seed.startswith(lead):
                    stripped = seed[len(lead) :]
                    seeds.append(stripped)
                    seeds.append(self._normalize_gguf_tensor_name(stripped))

        for seed in seeds:
            _add(seed)
            norm = self._normalize_gguf_tensor_name(seed)
            _add(norm)
            if prefix:
                if not seed.startswith(prefix):
                    _add(f"{prefix}{seed}")
                if not norm.startswith(prefix):
                    _add(f"{prefix}{norm}")
                if seed.startswith(prefix):
                    _add(seed[len(prefix) :])
                if norm.startswith(prefix):
                    _add(norm[len(prefix) :])

        return candidates

    @staticmethod
    def _match_unique_suffix(model_params: set[str], candidates: list[str]) -> str | None:
        """Fallback: match a single model param by suffix."""
        for cand in candidates:
            suffix_matches = [
                p for p in model_params if p == cand or p.endswith(f".{cand}")
            ]
            if len(suffix_matches) == 1:
                return suffix_matches[0]
        return None

    @staticmethod
    def _select_fallback_candidate(candidates: list[str], prefix: str) -> str | None:
        """Pick a stable fallback name when direct param matching is unavailable."""
        if not candidates:
            return None
        if prefix:
            for cand in candidates:
                if cand.startswith(prefix):
                    return cand
        return candidates[0]

    @staticmethod
    def _get_model_weight_names(model: nn.Module) -> set[str]:
        """Collect parameter/buffer names without materializing state_dict.

        GGUF loading can involve uninitialized parameters; calling state_dict()
        would trigger detach() on those tensors and fail.
        """
        names: set[str] = set()
        try:
            names.update(name for name, _ in model.named_parameters(remove_duplicate=False))
        except TypeError:
            names.update(name for name, _ in model.named_parameters())
        try:
            names.update(name for name, _ in model.named_buffers(remove_duplicate=False))
        except TypeError:
            names.update(name for name, _ in model.named_buffers())
        return names

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

        if load_format == "gguf":
            return self._prepare_gguf_file(
                model_name_or_path,
                revision=revision,
            )

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

    def _prepare_gguf_file(
        self,
        model_name_or_path: Path,
        revision: str | None,
    ) -> tuple[str, list[str], bool]:
        model_name_or_path = str(model_name_or_path)
        if os.path.isfile(model_name_or_path):
            return model_name_or_path, [model_name_or_path], False
        if model_name_or_path.startswith(("http://", "https://")) and model_name_or_path.endswith(".gguf"):
            local_path = hf_hub_download(url=model_name_or_path)
            return local_path, [local_path], False
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
            return local_path, [local_path], False
        if "/" in model_name_or_path and ":" in model_name_or_path:
            repo_id, quant_type = model_name_or_path.rsplit(":", 1)
            if os.path.isdir(repo_id):
                # Support local directory + quant type, e.g. /path/to/repo:Q8_0
                # matching the same filename patterns used by download_gguf.
                allow_patterns = [
                    f"*-{quant_type}.gguf",
                    f"*-{quant_type}-*.gguf",
                    f"*/*-{quant_type}.gguf",
                    f"*/*-{quant_type}-*.gguf",
                ]
                local_files: list[str] = []
                for pattern in allow_patterns:
                    local_files.extend(glob.glob(os.path.join(repo_id, pattern)))
                if not local_files:
                    raise ValueError(
                        f"Could not find local GGUF file in {repo_id!r} for quant_type {quant_type!r}"
                    )
                local_files = sorted(set(local_files), key=lambda x: (x.count("-"), x))
                local_path = local_files[0]
                return local_path, [local_path], False
            local_path = download_gguf(
                repo_id,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
            return local_path, [local_path], False
        raise ValueError(
            f"Unrecognized GGUF reference: {model_name_or_path} "
            "(expected local file, raw URL, <repo_id>/<filename>.gguf, "
            "or <repo_id>:<quant_type>)"
        )

    def _get_weights_iterator(self, source: "ComponentSource") -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        if self.load_config.load_format == "gguf":
            raise RuntimeError("GGUF weights should be loaded via load_weights() path.")
        source = self._resolve_component_source(source)
        try:
            hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
                source.model_or_path,
                source.subfolder,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
        except RuntimeError as e:
            # Some quantized transformer repos keep weights at repository root
            # (no "transformer/" subfolder). Retry once without subfolder.
            if self._uses_external_quantized_source(source) and source.subfolder is not None:
                logger.info(
                    "No weights found under subfolder '%s' in quantized source '%s'; retrying repository root.",
                    source.subfolder,
                    source.model_or_path,
                )
                source = dataclasses.replace(source, subfolder=None)
                hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
                    source.model_or_path,
                    source.subfolder,
                    source.revision,
                    source.fall_back_to_pt,
                    source.allow_patterns_overrides,
                )
            else:
                raise
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
        self.quantized_weights = od_config.quantized_weights
        self.quantization_scope = od_config.quantization_scope
        if self.quantized_weights and self.quantization_scope != "transformer_only":
            logger.warning(
                "quantized_weights currently supports quantization_scope='transformer_only', got '%s'.",
                self.quantization_scope,
            )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(od_config.dtype):
            with target_device:
                model = initialize_model(od_config)

            logger.debug("Loading weights on %s ...", load_device)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model)
            model_config_like = type("ModelConfigLike", (), {})()
            model_config_like.dtype = od_config.dtype
            model_config_like.quantization = od_config.quantization
            process_weights_after_loading(model, model_config_like, target_device)
        return model.eval()

    def load_weights(self, model: nn.Module) -> None:
        if self.load_config.load_format == "gguf":
            sources = cast(
                Iterable[DiffusersPipelineLoader.ComponentSource],
                getattr(model, "weights_sources", ()),
            )
            sources = list(sources)
            if len(sources) != 1:
                raise ValueError("GGUF loading currently supports a single ComponentSource.")
            source = self._resolve_component_source(sources[0])
            if self.counter_before_loading_weights == 0.0:
                self.counter_before_loading_weights = time.perf_counter()
            gguf_file, _, _ = self._prepare_gguf_file(Path(source.model_or_path), source.revision)

            model_params = self._get_model_weight_names(model)
            prefix = source.prefix or ""
            gguf_to_hf_name_map: dict[str, str] = {}
            exact_match_count = 0
            suffix_match_count = 0
            fallback_match_count = 0
            reader = gguf.GGUFReader(gguf_file)
            for tensor in reader.tensors:
                gguf_name = tensor.name
                candidates = self._candidate_hf_param_names(gguf_name, prefix)
                mapped_name = next((cand for cand in candidates if cand in model_params), None)
                if mapped_name is None:
                    mapped_name = self._match_unique_suffix(model_params, candidates)
                    if mapped_name is not None:
                        suffix_match_count += 1
                else:
                    exact_match_count += 1
                if mapped_name is None:
                    mapped_name = self._select_fallback_candidate(candidates, prefix)
                    if mapped_name is not None:
                        fallback_match_count += 1
                if mapped_name is not None:
                    gguf_to_hf_name_map[gguf_name] = mapped_name

            if not gguf_to_hf_name_map:
                raise ValueError(
                    "No GGUF tensor names matched model parameters. "
                    "Ensure GGUF tensor names align with model state_dict (optionally with the prefix)."
                )
            if exact_match_count + suffix_match_count == 0:
                logger.warning(
                    "No GGUF tensor names directly matched model params; "
                    "using heuristic name mapping for %d tensors.",
                    fallback_match_count,
                )
            logger.info(
                "Prepared GGUF name map: exact=%d, suffix=%d, heuristic=%d, total=%d",
                exact_match_count,
                suffix_match_count,
                fallback_match_count,
                len(gguf_to_hf_name_map),
            )

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                gguf_quant_weights_iterator(gguf_file, gguf_to_hf_name_map)
            )
        else:
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

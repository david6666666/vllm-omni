# vLLM-Omni Unified Quantization Feature Design Doc

Time context: this design is updated on 2026-02-09.

# 1 Overview

Define a unified quantization architecture for vLLM-Omni so diffusion and Omni/LLM stages share one consistent abstraction, argument contract, and rollout path for FP8/GGUF/INT8/AWQ.

## 1.1 Motivation

vLLM-Omni currently has working diffusion quantization paths for GGUF and FP8, and an initial diffusion quantization abstraction from PR #1034 (online FP8 for DiT). This is a good starting point, but quantization behavior is still fragmented across:

- diffusion-specific parsing and loading logic,
- stage/Omni argument propagation logic,
- and vLLM-native model quantization behavior for LLM stages.

As a result, users can see inconsistent semantics for similar parameters (`quantization`, `load_format`, external quantized weight source), and maintainers face duplicated work when adding new methods (INT8/AWQ) or extending to Omni models.

This feature introduces a single quantization design across stage types with clear method capability boundaries, stable user-facing CLI/API contracts, and a staged roadmap. It uses PR #1034 as Phase-1 baseline and evolves toward production-grade multi-method support.

This unified design must explicitly handle three real topology archetypes in vLLM-Omni:
- Single-stage diffusion model (for example `Qwen/Qwen-Image-2512`, `stage_type=diffusion`).
- Two-stage hybrid model (for example BAGEL, `stage-0 llm thinker` + `stage-1 diffusion dit`).
- Multi-stage all-LLM Omni model (for example Qwen3-Omni, `model_stage=thinker/talker/code2wav`, all `stage_type=llm`).

Related context:
- PR #1034: diffusion online FP8 abstraction and docs.
- Existing diffusion split-loading work: base repo + quantized transformer source.
- vLLM quantization ecosystem: FP8/GGUF/INT8/AWQ and method-specific configs.

## 1.2 Target

### Feature

In scope:

- Define one canonical quantization abstraction used by:
  - diffusion stage,
  - Omni/LLM stages,
  - Omni/Stage config paths (Python API, CLI, stage YAML).
- Preserve and standardize current diffusion behavior:
  - split source loading (`model` + optional `quantized_weights`),
  - GGUF support (`.gguf`, URL, `<repo>/<file>.gguf`, `<repo>:<quant_type>`),
  - native FP8 + online FP8 fallback.
- Add planned method expansion framework:
  - INT8 support path (first for Omni/LLM and diffusion where compatible),
  - AWQ support path (first Omni/LLM, diffusion as model-gated follow-up).
- Define consistent offline/online usage and argument rules.
- Define phased PR roadmap and acceptance gates.

Out of scope (this design document phase):

- Implementing all methods immediately in one PR.
- New quantization kernel algorithm work.
- Quantizing non-transformer diffusion components (VAE/text encoder) in first rollout.
- Dynamic runtime switching of quantization method per request.

### Accuracy

Correctness criteria:

- Same quantization input should resolve to the same method/behavior regardless of entrypoint (Omni Python API, `vllm serve --omni`, stage YAML).
- Split-loading must guarantee:
  - base pipeline assets from base model source,
  - quantized weights applied only to declared scope.
- Method-specific constraints must fail fast with actionable errors:
  - GGUF must use `load_format=gguf`,
  - incompatible stage type or hardware capability must return clear error.
- Backward compatibility:
  - existing non-quantized flows must remain unchanged,
  - existing diffusion GGUF/FP8 commands remain valid.

Validation methods:

- Unit tests for resolver precedence and config normalization.
- End-to-end smoke tests for representative model/method/stage combinations.
- Method-specific load-time assertions (mapped tensor coverage, uninitialized parameter detection).

### Performance

Targets:

- No performance regression for non-quantized baseline.
- Expected memory reduction and/or throughput gain with quantized methods:
  - FP8 and GGUF for diffusion transformer,
  - AWQ/INT8 for supported Omni/LLM checkpoints.
- Keep quantization decision overhead negligible (config-only at startup).
- Keep startup logging explicit for observability:
  - requested method,
  - resolved method,
  - source path,
  - load format,
  - fallback and skipped modules.

Trade-offs:

- Online quantization (FP8/INT8 where applicable) can increase load latency.
- More robust mapping logic (especially GGUF legacy names) adds complexity but avoids runtime uninitialized-parameter failures.

# 2 Design

## 2.1 Overview of Design

### Design principles

- One abstraction, multiple stage types: diffusion and Omni/LLM use the same quantization spec and resolution flow.
- Explicit source split: `model` is base assets; `quantized_weights` is optional quantized weight source.
- Method plugin style: leverage vLLM quantization configs as backend capability wherever possible.
- Strict method constraints: enforce compatibility at startup.
- Topology-aware stage targeting: stage overrides must support `stage_id`, `stage_type`, and `model_stage` to correctly address BAGEL and Qwen3-Omni pipelines.

### Core architecture

1. `QuantizationSpec` (canonical user intent)
- Carries method, format, source, scope, and method config.

2. `QuantizationResolver` (canonical decision engine)
- Normalizes and resolves final plan with deterministic precedence.

3. `ResolvedQuantPlan` (runtime executable plan)
- Contains effective method, effective source, vLLM quant config object, and validation outcomes.

4. Stage adapters
- Diffusion adapter: split-loading and method-specific loader behavior.
- Omni/LLM adapter: pass-through to vLLM model quantization semantics with unified logging and validation.

### High-level flow

```text
User Input (CLI / Python / Stage YAML)
  -> QuantizationSpec (normalized)
  -> QuantizationResolver
      - resolve method
      - resolve source + scope
      - resolve load format
      - build backend quant config
      - validate stage + hw compatibility
  -> ResolvedQuantPlan
      -> DiffusionModelRunner / LLM Engine init
          -> method-specific loader + process_weights_after_loading
  -> Ready for serving/inference
```

### Why this is better than per-stage ad-hoc logic

- Single place to reason about precedence and compatibility.
- Easier to add new methods (INT8/AWQ) without duplicating argument parsing behavior.
- Cleaner rollout: small PRs can add method adapters while keeping stable user contract.

### Model topology profiles in scope

1. `Qwen-Image-2512` type (single-stage diffusion)
- Typical stage layout: one stage, `stage_type=diffusion`.
- Quant focus: GGUF / native FP8 / online FP8 for diffusion transformer.
- Key requirement: split-loading with base assets + quantized transformer source.

2. `BAGEL` type (2-stage hybrid LLM + diffusion)
- Typical stage layout:
  - stage-0: `stage_type=llm`, `model_stage=thinker`
  - stage-1: `stage_type=diffusion`, `model_stage=dit`
- Quant focus:
  - thinker can use LLM-side quantization methods (for example AWQ/INT8 where supported),
  - dit can use diffusion-side methods (GGUF/FP8).
- Key requirement: stage override must distinguish thinker vs dit to avoid applying wrong method.

3. `Qwen3-Omni` type (3-stage all-LLM)
- Typical stage layout:
  - stage-0: `stage_type=llm`, `model_stage=thinker`
  - stage-1: `stage_type=llm`, `model_stage=talker`
  - stage-2: `stage_type=llm`, `model_stage=code2wav`
- Quant focus: Omni/LLM-side methods with per-model-stage control.
- Key requirement: `stage_type=llm` alone is insufficient; must support `model_stage` or `stage_id`.

## 2.2 Use Cases

1. `Qwen-Image-2512` type: diffusion online serving with GGUF transformer (primary)
- Scenario: `Qwen/Qwen-Image-2512` base assets + `unsloth/...-GGUF:Q8_0` quantized transformer.
- Configuration:
  - `model=Qwen/Qwen-Image-2512`
  - `quantized_weights=unsloth/Qwen-Image-2512-GGUF:Q8_0`
  - `quantization=gguf`
  - `load_format=gguf`
- Benefit: robust production startup and clear separation of responsibilities.

2. `BAGEL` type: 2-stage mixed quantization (llm thinker + diffusion dit)
- Scenario: stage-0 thinker uses LLM quant (for example AWQ), stage-1 dit uses GGUF.
- Configuration:
  - stage override by `model_stage`:
    - `model_stage=thinker -> method=awq`
    - `model_stage=dit -> method=gguf, load_format=gguf, quantized_weights=<gguf-source>`
- Benefit: one pipeline can mix LLM and diffusion quantization safely.

3. `Qwen3-Omni` type: 3-stage all-LLM targeted quantization
- Scenario: thinker/talker quantized, code2wav left unquantized for stability.
- Configuration:
  - stage override by `model_stage` (or `stage_id`) to set method per stage.
- Benefit: granular control where all stages share `stage_type=llm`.

4. Diffusion offline inference with native FP8 or online FP8 (primary)
- Scenario: user has FP8 checkpoint repo (native) or BF16 checkpoint (online FP8 fallback).
- Configuration:
  - native FP8: `quantization=auto` or `fp8`, `load_format=auto`.
  - online FP8 fallback: `quantization=fp8` with BF16 source.
- Benefit: compatible path for both checkpoint styles with unified flags.

### Topology-specific support matrix (Phase-1 to Phase-4)

| Model topology | Typical stage layout | Preferred selector | Quantization methods (target) | Entrypoint contract |
| --- | --- | --- | --- | --- |
| Qwen-Image-2512 type | single stage `diffusion` | `stage_type=diffusion` (or default) | GGUF, FP8 (native + online); INT8/AWQ are follow-up and model-gated | offline `text_to_image.py` and online `vllm serve --omni` both allow flat args; profile optional |
| BAGEL type | `stage-0 llm thinker`, `stage-1 diffusion dit` | `model_stage=thinker` and `model_stage=dit` | thinker: AWQ/INT8 (when supported); dit: GGUF/FP8 | online `vllm serve --omni` should use profile; flat global args are only safe when both stages use same method |
| Qwen3-Omni type | `stage-0 llm thinker`, `stage-1 llm talker`, `stage-2 llm code2wav` | `model_stage=*` or exact `stage_id` | per-stage LLM methods (AWQ/INT8/none by stage) | online `vllm serve --omni` should use profile; avoid only `stage_type=llm` selector |

Stage-targeting requirements:
- For BAGEL and Qwen3-Omni, profile-based stage overrides are the default recommended path.
- For Qwen3-Omni, `stage_type=llm` is too broad and should be treated as a coarse fallback only.
- For diffusion-only models, flat args remain first-class for backward compatibility.

## 2.3 API Design

### Current Component Changes

This section describes required or planned component-level changes on top of current implementation.

`vllm_omni/diffusion/data.py` (`OmniDiffusionConfig`)
- Current:
  - has `quantization`, `quantization_config_file`, `quantization_config_dict_json`,
    `load_format`, `quantization_scope`, `quantized_weights`, `quant_config`.
- Planned:
  - keep backward compatibility,
  - add optional structured `quantization_spec` field (or equivalent normalized dict),
  - normalize legacy fields into canonical spec during initialization.
- Why:
  - enable one unified resolver and reduce field coupling logic in runners.

`vllm_omni/model_executor/model_loader/quant_utils.py`
- Current:
  - diffusion-focused method inference (`fp8`/`gguf`) and config resolution.
- Planned:
  - evolve into shared resolver utility with stage awareness,
  - support additional methods (`int8`, `awq`) behind capability checks,
  - return structured `ResolvedQuantPlan`.
- Why:
  - avoid per-runner custom logic and enable consistent error handling.

`vllm_omni/diffusion/worker/diffusion_model_runner.py`
- Current:
  - resolves method and quant config before model load.
- Planned:
  - consume `ResolvedQuantPlan` and avoid method-specific branching spread.
- Why:
  - keep runner orchestration-focused and make method behavior testable in resolver/adapter.

`vllm_omni/diffusion/model_loader/diffusers_loader.py`
- Current:
  - split-loading for transformer source replacement,
  - GGUF source parsing and name mapping.
- Planned:
  - method adapter hooks from resolved plan,
  - stricter mapped coverage diagnostics,
  - maintain model-specific mapping extensions in model loaders.
- Why:
  - isolate GGUF complexity while preserving generic loader path.

`vllm_omni/entrypoints/omni.py`, `vllm_omni/entrypoints/async_omni.py`, `vllm_omni/entrypoints/omni_stage.py`
- Current:
  - inject diffusion quant args from kwargs into stage configs.
- Planned:
  - inject canonical quantization spec globally and stage overrides consistently,
  - keep old flags as compatibility aliases.
- Why:
  - ensure Omni and Stage behavior are identical by contract.

`vllm_omni/entrypoints/cli/serve.py` and offline example scripts
- Current:
  - already have diffusion quantization args including `--quantized-weights`.
- Planned:
  - document and standardize naming/precedence,
  - add optional structured override argument for multi-stage mixed quantization.
- Why:
  - reduce ambiguity for users and simplify future expansion.

### New APIs

#### A. Canonical Quantization Spec (internal/public config schema)

```python
@dataclass
class QuantizationSpec:
    method: str | None = None
    load_format: str | None = None
    quantized_weights: str | None = None
    scope: str = "transformer_only"
    config_file: str | None = None
    config_json: str | None = None
    config_dict: dict[str, Any] | None = None
```

Notes:
- `method` examples: `fp8`, `gguf`, `int8`, `awq`, `auto`, `None`.
- `scope` default remains `transformer_only` for diffusion.

#### B. Stage-aware Quantization Profile (for multi-stage override)

```python
@dataclass
class StageSelector:
    stage_id: int | None = None
    stage_type: str | None = None
    model_stage: str | None = None

@dataclass
class StageQuantizationOverride:
    selector: StageSelector
    spec: QuantizationSpec

@dataclass
class QuantizationProfile:
    default: QuantizationSpec
    stage_overrides: list[StageQuantizationOverride] | None = None
```

Selector examples:
- `{"stage_type": "diffusion"}` for qwen-image-type single-stage diffusion.
- `{"model_stage": "dit"}` and `{"model_stage": "thinker"}` for BAGEL.
- `{"model_stage": "talker"}` or `{"stage_id": 2}` for Qwen3-Omni.

Selector matching priority:
1. exact `stage_id`
2. exact `model_stage`
3. exact `stage_type`
4. fallback to `default`
5. if multiple overrides tie at the same priority, the first declaration order wins.

Selector validation rules:
- at least one of `stage_id`, `stage_type`, or `model_stage` must be provided.
- unknown `model_stage` values must fail fast with available stage metadata.
- duplicate `stage_id` selectors should fail schema validation.

#### C. Resolver API

```python
def resolve_quantization_plan(
    *,
    model: str | None,
    stage_type: str,
    spec: QuantizationSpec,
    hw_context: dict[str, Any] | None = None,
) -> ResolvedQuantPlan:
    ...
```

Expected output:
- `resolved_method`
- `resolved_load_format`
- `resolved_source`
- `resolved_scope`
- `vllm_quant_config`
- `warnings`
- `hard_errors` (raised before execution)

#### D. Backward compatibility adapters

Legacy fields remain accepted:
- `quantization`
- `load_format`
- `quantized_weights`
- `quantization_config_file`
- `quantization_config_dict_json`
- `quantization_scope`

These are normalized into `QuantizationSpec` with deterministic precedence.
If both profile and flat args are provided, profile override wins for matched stages.

#### E. API placement and directory plan

To avoid scattering quantization logic across unrelated modules, new APIs should be placed in a dedicated quantization package and only referenced by runners/loaders/entrypoints.

Recommended layout:

```text
vllm_omni/
  quantization/
    __init__.py
    spec.py                  # QuantizationSpec / QuantizationProfile dataclasses
    resolver.py              # resolve_quantization_plan(...)
    plan.py                  # ResolvedQuantPlan dataclass
    capabilities.py          # stage/method/hardware support checks
    adapters/
      __init__.py
      diffusion.py           # diffusion-specific adapter helpers
      llm.py                 # Omni/LLM-stage adapter helpers
    compat/
      __init__.py
      legacy_args.py         # normalize old flat args -> QuantizationSpec
```

API-to-file mapping:

1. `QuantizationSpec`
- File: `vllm_omni/quantization/spec.py`
- Why: core schema shared by all stage types and entrypoints.

2. `QuantizationProfile`
- File: `vllm_omni/quantization/spec.py`
- Why: keep default + stage overrides in one schema module.

3. `ResolvedQuantPlan`
- File: `vllm_omni/quantization/plan.py`
- Why: explicit resolved output contract decouples resolver from runtime executors.

4. `resolve_quantization_plan(...)`
- File: `vllm_omni/quantization/resolver.py`
- Why: single place for precedence, source resolution, and backend config creation.

5. method/stage/hardware validators
- File: `vllm_omni/quantization/capabilities.py`
- Why: centralized compatibility matrix and fail-fast checks.

6. legacy compatibility normalizer
- File: `vllm_omni/quantization/compat/legacy_args.py`
- Why: keep backward compatibility logic out of runner business logic.

7. diffusion adapter helpers
- File: `vllm_omni/quantization/adapters/diffusion.py`
- Why: bind resolved plan to diffusion loader behavior (split-loading, scope handling, method constraints).

8. Omni/LLM adapter helpers
- File: `vllm_omni/quantization/adapters/llm.py`
- Why: bind resolved plan to vLLM model quantization path for LLM stages.

Integration points (existing files to update):

- `vllm_omni/diffusion/worker/diffusion_model_runner.py`
  - consume `ResolvedQuantPlan` instead of ad-hoc per-field resolution.
- `vllm_omni/diffusion/model_loader/diffusers_loader.py`
  - consume diffusion adapter outputs and keep weight-loading responsibilities only.
- `vllm_omni/model_executor/model_loader/quant_utils.py`
  - transitional bridge; eventually delegate to `vllm_omni/quantization/resolver.py`.
- `vllm_omni/diffusion/data.py`
  - normalize diffusion fields into `QuantizationSpec` (or store `quantization_spec` directly).
- `vllm_omni/entrypoints/omni.py`
  - inject default spec and stage overrides consistently.
- `vllm_omni/entrypoints/async_omni.py`
  - keep async path identical to sync path for quantization fields.
- `vllm_omni/entrypoints/omni_stage.py`
  - ensure stage worker receives normalized spec payload.
- `vllm_omni/entrypoints/cli/serve.py`
  - parse both legacy flat flags and optional profile JSON into canonical spec.
- `examples/offline_inference/text_to_image/text_to_image.py`
  - keep current flags; optionally add profile JSON input for future mixed-stage parity.

Import boundary rules:

- Entrypoints and configs may import only `vllm_omni.quantization.spec` and compatibility helpers.
- Runners may import resolver + adapters.
- Loaders should not own precedence logic; they execute resolved plans.
- Model files should not parse CLI/config fields directly; they consume resolved quant config only.

### User-facing CLI / args design

This section defines both current stable usage and proposed unified expansion.

#### 1) Current stable args (already in code)

Online serving (`vllm serve ... --omni`):
- `--quantization`
- `--load-format`
- `--quantized-weights`
- `--quantization-config-file`
- `--quantization-config-dict-json`
- `--quantization-scope`

Offline diffusion example (`text_to_image.py`):
- `--quantization`
- `--load_format`
- `--quantized_weights`
- `--quantization_config_file`
- `--quantization_config_dict_json`

#### 2) Proposed unified args (for future multi-stage mixed quantization)

Keep all current flags; add optional structured input:

- Online:
  - `--quantization-profile-json '<json>'`
- Offline/Python:
  - `quantization_profile=<dict>`

Example JSON:

```json
{
  "default": {
    "method": "auto",
    "load_format": "auto"
  },
  "stage_overrides": [
    {
      "selector": {
        "model_stage": "dit"
      },
      "spec": {
        "method": "gguf",
        "load_format": "gguf",
        "quantized_weights": "unsloth/Qwen-Image-2512-GGUF:Q8_0",
        "scope": "transformer_only"
      }
    },
    {
      "selector": {
        "model_stage": "thinker"
      },
      "spec": {
        "method": "awq"
      }
    },
    {
      "selector": {
        "model_stage": "code2wav"
      },
      "spec": {
        "method": null
      }
    }
  ]
}
```

Notes:
- For Qwen3-Omni, use `model_stage` (`thinker`/`talker`/`code2wav`) or `stage_id`; `stage_type=llm` alone is not enough.
- For BAGEL, use `model_stage=thinker` and `model_stage=dit`.

Rationale:
- avoids exploding CLI flags for each stage,
- keeps one consistent multi-stage contract,
- supports precise stage selection for bagel/qwen3-omni.

#### 3) Topology-specific CLI and stage YAML contract

1. Qwen-Image-2512 type (single diffusion stage)
- Recommended: flat args (`--quantization`, `--load-format`, `--quantized-weights`).
- Optional: profile JSON with `default` only.
- Stage YAML requirement: none beyond standard single diffusion stage.

2. BAGEL type (2-stage `thinker + dit`)
- Recommended: profile JSON with two overrides:
  - `selector.model_stage=thinker` for LLM-side method.
  - `selector.model_stage=dit` for diffusion-side method and optional `quantized_weights`.
- Stage YAML requirement: stages should include stable `model_stage` labels (`thinker`, `dit`).
- Fallback: if profile is absent, global flat args apply to both stages and may be rejected by capability checks.

3. Qwen3-Omni type (3-stage all-LLM)
- Recommended: profile JSON with `model_stage` or exact `stage_id` selectors for each LLM stage.
- Stage YAML requirement: stage definitions must expose unambiguous `model_stage` values (`thinker`, `talker`, `code2wav`).
- Constraint: `selector.stage_type=llm` should only be used as broad default, then narrowed by specific overrides.

### Recommended command examples

#### A. `Qwen-Image-2512` type: diffusion online serving with GGUF

```bash
vllm serve Qwen/Qwen-Image-2512 --omni --port 8091 \
  --quantized-weights "unsloth/Qwen-Image-2512-GGUF:Q8_0" \
  --quantization gguf \
  --load-format gguf
```

#### B. `Qwen-Image-2512` type: diffusion offline inference with GGUF

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2512 \
  --quantized_weights "unsloth/Qwen-Image-2512-GGUF:Q8_0" \
  --quantization gguf \
  --load_format gguf \
  --prompt "a watercolor painting of tokyo in the rain" \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 28 \
  --output outputs/qwen_image_gguf.png
```

#### C. `Qwen-Image-2512` type: diffusion online serving with native FP8

```bash
vllm serve Qwen/Qwen-Image-2512 --omni --port 8091 \
  --quantized-weights "unsloth/Qwen-Image-2512-FP8" \
  --quantization fp8 \
  --load-format auto
```

#### D. `Qwen-Image-2512` type: diffusion offline inference with native FP8 / online FP8 fallback

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2512 \
  --quantized_weights "unsloth/Qwen-Image-2512-FP8" \
  --quantization fp8 \
  --load_format auto \
  --prompt "cinematic photo of an arctic fox under aurora" \
  --height 1328 \
  --width 1328 \
  --num_inference_steps 30 \
  --output outputs/qwen_image_fp8.png
```

#### E. `BAGEL` type: 2-stage mixed profile (thinker + dit)

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/bagel.yaml \
  --quantization-profile-json '{"default":{"method":"auto"},"stage_overrides":[{"selector":{"model_stage":"thinker"},"spec":{"method":"awq"}},{"selector":{"model_stage":"dit"},"spec":{"method":"gguf","load_format":"gguf","quantized_weights":"<bagel-dit-gguf-source>"}}]}'
```

#### F. `Qwen3-Omni` type: 3-stage llm profile (thinker/talker quantized, code2wav unquantized)

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml \
  --quantization-profile-json '{"default":{"method":"auto"},"stage_overrides":[{"selector":{"model_stage":"thinker"},"spec":{"method":"awq"}},{"selector":{"model_stage":"talker"},"spec":{"method":"int8"}},{"selector":{"model_stage":"code2wav"},"spec":{"method":null}}]}'
```

#### G. Omni/LLM single-method serving (planned unified contract, method depends on checkpoint)

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni \
  --quantization awq
```

```bash
vllm serve <int8-supported-model> --omni \
  --quantization int8
```

#### H. Generic multi-stage profile (future)

```bash
vllm serve <omni-pipeline-model> --omni \
  --quantization-profile-json '{"default":{"method":"auto"},"stage_overrides":[{"selector":{"stage_type":"diffusion"},"spec":{"method":"gguf","load_format":"gguf","quantized_weights":"unsloth/Qwen-Image-2512-GGUF:Q8_0"}},{"selector":{"stage_type":"llm"},"spec":{"method":"awq"}}]}'
```

## 2.4 API call dependency

### A. Runtime call flow (canonical)

1. User passes quantization inputs (flat flags and/or profile JSON).
2. Entrypoint normalizes into `QuantizationSpec` or `QuantizationProfile`.
3. Stage construction injects default and stage override specs.
4. Stage runner calls `resolve_quantization_plan(...)`.
5. Resolver performs:
   - precedence resolution,
   - source resolution,
   - load format validation,
   - backend config construction,
   - compatibility checks.
6. Runner initializes model loader/engine with resolved plan.
7. Weight load + post-load quant processing.
8. Logs final resolved state for observability.

### B. Resolution precedence

For each stage:

1. explicit stage override spec (selector match priority: `stage_id` > `model_stage` > `stage_type`)
2. explicit default spec
3. explicit legacy flat args
4. quant config file/json
5. quantized weight source model config
6. base model config
7. no quantization

### C. Error paths

- `method=gguf` and `load_format != gguf`: fail fast.
- `quantized_weights` format unresolved (e.g., bad local `:Q8_0` path): fail fast with source hint.
- resolved method unsupported for stage type: fail fast with support matrix hint.
- invalid stage selector (unknown `model_stage`/missing target stage): fail fast with available stage metadata in error.
- hardware capability mismatch for method/backend: fail fast with required capability.
- unresolved or low-coverage GGUF tensor mapping: fail with unmatched examples and migration hint.
- deprecated method path: warning or error by policy flag.

# 3 Test cases

## 3.1 Unit Test (UT) design

### Resolver / spec normalization

File suggestion: `tests/quantization/test_resolver.py`

1. `test_resolver_precedence_stage_override_wins`
- Purpose: ensure stage override beats default/legacy fields.

2. `test_resolver_prefers_quantized_weights_config_over_base_model`
- Purpose: preserve split-loading detection semantics.

3. `test_resolver_gguf_requires_gguf_load_format`
- Purpose: ensure fail-fast on invalid pairing.

4. `test_resolver_backward_compat_flat_fields`
- Purpose: old flags map to same resolved plan.

5. `test_resolver_stage_type_capability_validation`
- Purpose: invalid method-stage combinations fail early.

6. `test_resolver_model_stage_selector_priority`
- Purpose: verify `model_stage` override wins over `stage_type` for qwen3-omni-like all-llm topologies.

7. `test_resolver_stage_id_selector_priority`
- Purpose: verify `stage_id` selector has highest priority.

### Diffusion loader behavior

File suggestion: `tests/diffusion/test_quant_loader.py`

1. `test_diffusion_transformer_source_replaced_only_for_transformer_scope`
2. `test_diffusion_gguf_local_repo_with_quant_type_suffix`
3. `test_diffusion_gguf_mapping_coverage_and_unmatched_reporting`
4. `test_diffusion_no_state_dict_call_on_uninitialized_gguf_params`

### Omni/LLM stage integration

File suggestion: `tests/omni/test_quantization_stage_injection.py`

1. `test_omni_stage_injects_default_quant_spec`
2. `test_omni_stage_injects_stage_overrides`
3. `test_async_omni_and_omni_consistent_quant_args`
4. `test_omni_stage_selector_matches_bagel_layout`
5. `test_omni_stage_selector_matches_qwen3_omni_layout`

## 3.2 Smoke Test (ST) design

File suggestion: `tests/e2e/quantization/`

1. `test_qwen_image_2512_diffusion_gguf_offline_smoke`
- Setup: base model + gguf quant source.
- Verify: startup, one image generation, no uninitialized parameter error.

2. `test_qwen_image_2512_diffusion_fp8_online_smoke`
- Setup: bf16 base model + `quantization=fp8`.
- Verify: successful load and one generation request.

3. `test_qwen_image_2512_diffusion_fp8_native_smoke`
- Setup: native FP8 checkpoint source.
- Verify: resolved method/logging + successful generation.

4. `test_bagel_two_stage_mixed_quant_profile_smoke` (planned)
- Setup: bagel stage config, thinker override + dit override.
- Verify: stage-0 and stage-1 resolve to expected methods independently.

5. `test_qwen3_omni_three_stage_quant_profile_smoke` (planned)
- Setup: qwen3-omni stage config, thinker/talker/code2wav overrides by `model_stage`.
- Verify: resolver applies correct plan to each llm stage.

6. `test_omni_llm_awq_smoke` (planned)
- Setup: AWQ-compatible checkpoint.
- Verify: model loads under Omni mode and responds.

7. `test_omni_llm_int8_smoke` (planned)
- Setup: INT8-compatible checkpoint and hardware.
- Verify: model loads and responds.

8. `test_multi_stage_mixed_quant_profile_smoke` (future)
- Setup: profile with diffusion GGUF + LLM AWQ.
- Verify: both stages resolve correctly and pipeline completes one request.

# 4 Roadmap

## 4.1 PR sequence

### PR-1 (baseline, first stage)

- Base: PR #1034 + existing diffusion split-loading implementation.
- Scope:
  - diffusion quantization abstraction (FP8 online),
  - current GGUF/FP8 diffusion behavior stabilization,
  - docs + UT for baseline behavior.
- Exit criteria:
  - diffusion FP8 online/native and GGUF single-stage flows are stable.

### PR-2 (unified resolver and spec contract)

- Scope:
  - introduce canonical `QuantizationSpec` and `resolve_quantization_plan`,
  - normalize legacy fields into canonical spec,
  - unify Omni/AsyncOmni/Stage injection semantics.
- Exit criteria:
  - same quant intent resolves identically across entrypoints.

### PR-3 (Omni/LLM method unification: AWQ first)

- Scope:
  - unified handling for Omni/LLM stages,
  - AWQ support path validated with vLLM native quant backend.
- Exit criteria:
  - Omni/LLM AWQ smoke test passes.

### PR-4 (INT8 integration)

- Scope:
  - INT8 support path through unified resolver and stage adapters,
  - method capability checks and docs.
- Exit criteria:
  - Omni/LLM INT8 smoke test passes;
  - diffusion INT8 path enabled where compatible.

### PR-5 (diffusion AWQ experimental)

- Scope:
  - model-gated diffusion AWQ support,
  - mapping and loader adaptation where needed.
- Exit criteria:
  - at least one diffusion model AWQ e2e path stable.

### PR-6 (hardening and observability)

- Scope:
  - final support matrix docs,
  - richer startup diagnostics,
  - regression and compatibility matrix in CI.
- Exit criteria:
  - production-readiness gate for all officially supported methods/stages.

## 4.2 Support matrix policy

Method support must be explicitly declared by:
- stage type (`diffusion`, `llm`, `omni-mixed`),
- model family (where required),
- hardware capability.

Any unsupported combination must fail fast with a deterministic message and suggested alternative method.

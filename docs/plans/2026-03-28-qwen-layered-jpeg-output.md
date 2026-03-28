# Qwen-Image-Layered JPEG Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent `/v1/images/edits` from returning HTTP 500 when `Qwen-Image-Layered` outputs RGBA layers and the request asks for `output_format=jpeg`.

**Architecture:** Keep the fix in the OpenAI image response encoding layer so model internals stay unchanged. Convert alpha-bearing PIL images to a JPEG-safe RGB image only when the requested output format does not support alpha, while preserving existing PNG and WEBP behavior.

**Tech Stack:** FastAPI, Pillow, pytest, vLLM-Omni OpenAI image serving path.

---

### Task 1: Confirm the failing path

**Files:**
- Modify: `vllm_omni/entrypoints/openai/api_server.py`
- Test: `tests/entrypoints/openai_api/test_image_server.py`

**Step 1: Capture current root cause**

Verify that layered responses can contain RGBA images and that Pillow raises `OSError: cannot write mode RGBA as JPEG` when saving them as JPEG without conversion.

**Step 2: Identify the narrow fix point**

Keep the change inside `_encode_image_base64_with_compression()` so both image generation and image editing paths benefit without changing model outputs.

### Task 2: Add a regression test

**Files:**
- Test: `tests/entrypoints/openai_api/test_image_server.py`

**Step 1: Create a fake layered engine output**

Return an `RGBA` PIL image from the existing mocked image edit path.

**Step 2: Verify JPEG response succeeds**

Add a test that posts to `/v1/images/edits` with `output_format=jpeg`, decodes the returned bytes, and asserts the image format is JPEG and the mode is JPEG-compatible.

### Task 3: Implement the minimal fix

**Files:**
- Modify: `vllm_omni/entrypoints/openai/api_server.py`

**Step 1: Normalize non-alpha-safe formats**

When the requested format is `jpg` or `jpeg`, flatten alpha-bearing images onto a solid background before calling `image.save()`.

**Step 2: Preserve existing behavior elsewhere**

Do not alter PNG handling. Keep WEBP on the existing path unless the image mode requires a safe conversion.

### Task 4: Verify locally

**Files:**
- Test: `tests/entrypoints/openai_api/test_image_server.py`

**Step 1: Run targeted pytest**

Run the new regression test together with the existing JPEG image edit test.

**Step 2: Record gaps**

If the local environment cannot execute the relevant tests, document the exact blocker and the remote validation command the user should run.

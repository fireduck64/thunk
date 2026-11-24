import base64
from datetime import datetime, timezone
import io
import json
import os
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from PIL import Image

from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.image import downsize_image_bytes, encode_jpeg_data_url
from src.utils.metrics import MetricSpan, count_tokens


DEFAULT_IMAGE_MODEL = "gemini-3-pro-image"

MODEL_ALIASES: Dict[str, str] = {
    "pro": "gemini-3-pro-image",
    "nano-banana-pro": "gemini-3-pro-image",
    "banana-pro": "gemini-3-pro-image",
    "2.1": "gemini-nano-banana-2.1",
    "nano-banana-2.1": "gemini-nano-banana-2.1",
    "flash": "gemini-3.1-flash-image",
    "nano-banana-2": "gemini-3.1-flash-image",
    "lite": "gemini-3.1-flash-lite-image",
    "nano-banana-2-lite": "gemini-3.1-flash-lite-image",
}

VALID_ASPECT_RATIOS = {
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "5:4",
    "4:5",
    "21:9",
}

VALID_IMAGE_SIZES = {"0.5K", "512", "1K", "2K", "4K"}

MIME_TO_EXT: Dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

EXT_TO_MIME: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class ImageGenComponent(BaseComponent):
    """
    Provides native multimodal Gemini image generation and reference-guided
    character/scene editing (`generate_image`), reusing the configured Gemini API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_output_library: Optional[str] = None,
        libraries: Optional[Dict[str, str]] = None,
        shared_workspace_dir: Optional[str] = None,
        client: Optional[Any] = None,
        enabled: Optional[bool] = None,
    ):
        super().__init__()
        self.config = load_config()
        ig_cfg = self.config.get("image_gen", {})
        oracle_cfg = self.config.get("oracle", {})
        llm_cfg = self.config.get("llm", {})

        self.enabled = True if enabled is None else bool(enabled)
        self.api_key = (
            api_key
            or ig_cfg.get("gemini_api_key")
            or ig_cfg.get("api_key")
            or oracle_cfg.get("gemini_api_key")
            or llm_cfg.get("gemini_api_key")
        )
        self.default_model = self._normalize_model_name(
            default_model
            or ig_cfg.get("default_model")
            or ig_cfg.get("model")
            or DEFAULT_IMAGE_MODEL
        )
        self.default_output_library = (
            default_output_library
            or ig_cfg.get("default_output_library")
            or "discord_file"
        )

        self.container_workdir = (
            self.config.get("workspace", {})
            .get("container_workdir", "/home/agent/workspace")
            .rstrip("/")
        )
        self.libraries = self._build_libraries_map(libraries)
        self.shared_workspace_dir = self._resolve_shared_workspace_dir(shared_workspace_dir)

        if client is not None:
            self.client = client
        elif self.api_key and self.enabled:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            if self.enabled:
                print(
                    "[ImageGen] Warning: No Gemini API key found in [image_gen], [oracle], or [llm]. "
                    "generate_image tool will be disabled."
                )

    def _build_libraries_map(self, explicit_libs: Optional[Dict[str, str]]) -> Dict[str, str]:
        if explicit_libs is not None:
            return {k: os.path.abspath(v) for k, v in explicit_libs.items() if v}

        raw_libs = dict(self.config.get("file_reader", {}).get("libraries") or {})
        ws_dir = self.config.get("workspace", {}).get("workspace_dir")
        if ws_dir and "workspace" not in raw_libs:
            raw_libs["workspace"] = ws_dir

        discord_cfg = self.config.get("discord", {})
        img_dir = discord_cfg.get("files_dir") or discord_cfg.get("images_dir")
        if img_dir and "discord_file" not in raw_libs:
            raw_libs["discord_file"] = img_dir

        return {k: os.path.abspath(v) for k, v in raw_libs.items() if v}

    def _resolve_shared_workspace_dir(self, explicit_shared: Optional[str]) -> Optional[str]:
        if explicit_shared is not None:
            return os.path.abspath(explicit_shared) if explicit_shared else None
        ws_cfg = self.config.get("workspace", {})
        if ws_cfg.get("shared_workspace_dir"):
            return os.path.abspath(ws_cfg["shared_workspace_dir"])
        ws_base = self.libraries.get("workspace")
        if ws_base:
            if os.path.basename(os.path.dirname(ws_base)) == "test":
                return os.path.abspath(os.path.join(os.path.dirname(ws_base), "shared_workspace"))
            return os.path.abspath("data/shared/workspace")
        return None

    @staticmethod
    def _normalize_model_name(model_str: Optional[str]) -> str:
        clean = (model_str or "").strip()
        if not clean:
            return DEFAULT_IMAGE_MODEL
        if clean.startswith("models/"):
            clean = clean[len("models/") :]
        low = clean.lower()
        if low in MODEL_ALIASES:
            return MODEL_ALIASES[low]
        return clean

    def get_system_prompt_addition(self) -> str:
        if not self.enabled or not self.client:
            return ""
        return (
            "--- IMAGE GENERATION & CHARACTER REFERENCE STUDIO (GEMINI IMAGE) ---\n"
            f"You can create and edit images using `generate_image` (default model: `{self.default_model}`).\n"
            "- **Reference Images:** Pass up to 14 local images in `reference_images` (e.g. `['discord_file:photo.jpg', 'workspace:shared/art/char_sheet.png']`) "
            "to preserve character identity, style, or objects across new scenes, or to edit an existing image.\n"
            "- **Models:** Omit `model` to use `{default}` (Nano Banana Pro — highest quality, multi-character consistency, and crisp text), "
            "or specify `'gemini-nano-banana-2.1'`, `'gemini-3.1-flash-image'`, or `'gemini-3.1-flash-lite-image'` for faster drafts.\n"
            "- **Sharing & Saving:** Generated images are attached to your visual context for inspection and saved atomically to `output_library` "
            "(`'discord_file'` or `'workspace'`, including `'shared/...'`), ready to send via `send_message(attachment_library=..., attachment_path=...)`."
        ).replace("{default}", self.default_model)

    def get_tools(self) -> List[Callable]:
        if not self.enabled or not self.client:
            return []
        return [self.generate_image]

    def _resolve_library_key(self, lib_name: str) -> str:
        clean = (lib_name or "").strip()
        low = clean.lower()
        aliases = {
            "discord_images": "discord_file",
            "discord_files": "discord_file",
            "images": "discord_file",
            "shared": "workspace",
            "shared_workspace": "workspace",
        }
        if clean in self.libraries:
            return clean
        for k in self.libraries:
            if k.lower() == low:
                return k
        if low in aliases and aliases[low] in self.libraries:
            return aliases[low]
        raise ValueError(
            f"Unknown library '{lib_name}'. Available libraries: {list(self.libraries.keys())}"
        )

    def _resolve_library_path(
        self,
        library: str,
        rel_path: str,
        must_exist: bool = False,
    ) -> Tuple[str, str, str]:
        raw_lib_low = (library or "").strip().lower()
        resolved_lib = self._resolve_library_key(library)
        base_dir = self.libraries[resolved_lib]

        clean_rel = (rel_path or "").strip()
        if "[path: " in clean_rel and clean_rel.endswith("]"):
            clean_rel = clean_rel.split("[path: ", 1)[1][:-1].strip()

        if resolved_lib.lower() == "workspace":
            if clean_rel == self.container_workdir:
                clean_rel = ""
            elif clean_rel.startswith(self.container_workdir + "/"):
                clean_rel = clean_rel[len(self.container_workdir) + 1 :]

        if raw_lib_low in ("shared", "shared_workspace") and not (
            clean_rel == "shared" or clean_rel.startswith("shared/")
        ):
            clean_rel = f"shared/{clean_rel.lstrip('/')}" if clean_rel.lstrip("/") else "shared"

        safe_rel = clean_rel.lstrip(os.sep)
        if not safe_rel:
            raise ValueError("File path cannot be empty.")

        norm_parts = os.path.normpath(safe_rel).split(os.sep)
        if ".." in norm_parts:
            raise ValueError(f"Access denied: path traversal ('..') is not allowed in '{rel_path}'.")

        if (
            resolved_lib.lower() == "workspace"
            and self.shared_workspace_dir
            and (safe_rel == "shared" or safe_rel.startswith("shared/"))
        ):
            sub_rel = safe_rel[len("shared") :].lstrip(os.sep)
            if not sub_rel:
                raise ValueError("Path must specify a file inside 'shared/', not the directory itself.")
            abs_path = os.path.abspath(os.path.join(self.shared_workspace_dir, sub_rel))
            if os.path.commonpath([self.shared_workspace_dir, abs_path]) != self.shared_workspace_dir:
                raise ValueError(f"Access denied: '{rel_path}' is outside shared workspace.")
            sub_norm = os.path.relpath(abs_path, self.shared_workspace_dir)
            display_rel = f"shared/{sub_norm}"
        else:
            abs_path = os.path.abspath(os.path.join(base_dir, safe_rel))
            if os.path.commonpath([base_dir, abs_path]) != base_dir:
                raise ValueError(f"Access denied: '{rel_path}' is outside library '{resolved_lib}'.")
            display_rel = os.path.relpath(abs_path, base_dir)

        if must_exist:
            if not os.path.exists(abs_path):
                raise FileNotFoundError(
                    f"Reference image '{display_rel}' does not exist in library '{resolved_lib}'."
                )
            if not os.path.isfile(abs_path):
                raise ValueError(
                    f"Reference path '{display_rel}' in library '{resolved_lib}' is not a regular file."
                )
        return resolved_lib, abs_path, display_rel

    def _resolve_reference_spec(self, spec: str) -> Tuple[str, str, str]:
        raw = (spec or "").strip()
        if not raw:
            raise ValueError("Empty reference image entry.")

        if raw.startswith(self.container_workdir + "/") and "workspace" in self.libraries:
            return self._resolve_library_path("workspace", raw, must_exist=True)

        if ":" in raw and not raw.startswith("/"):
            lib_part, path_part = raw.split(":", 1)
            if lib_part.strip() and path_part.strip():
                return self._resolve_library_path(lib_part.strip(), path_part.strip(), must_exist=True)

        if (raw.startswith("shared/") or raw.startswith("/shared/")) and "workspace" in self.libraries:
            return self._resolve_library_path("workspace", raw.lstrip("/"), must_exist=True)

        # Bare filename/subpath fallback: try discord_file first, then workspace
        search_order = [
            k for k in ("discord_file", "workspace") if k in self.libraries
        ] + [k for k in self.libraries if k not in ("discord_file", "workspace")]
        last_err: Optional[Exception] = None
        for cand_lib in search_order:
            try:
                return self._resolve_library_path(cand_lib, raw, must_exist=True)
            except FileNotFoundError as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        raise FileNotFoundError(f"Could not resolve reference image '{raw}' in any configured library.")

    @staticmethod
    def _parse_reference_list(raw_refs: Any) -> List[str]:
        if raw_refs is None:
            return []
        if isinstance(raw_refs, list):
            return [str(x).strip() for x in raw_refs if str(x or "").strip()]
        if isinstance(raw_refs, str):
            s = raw_refs.strip()
            if not s:
                return []
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x or "").strip()]
                except Exception:
                    pass
            return [part.strip() for part in re.split(r"[,\n]+", s) if part.strip()]
        return [str(raw_refs).strip()]

    @staticmethod
    def _load_reference_part(abs_path: str) -> Tuple[types.Part, int]:
        with open(abs_path, "rb") as f:
            raw_bytes = f.read()
        ext = os.path.splitext(abs_path)[1].lower()
        mime = EXT_TO_MIME.get(ext)
        if not mime:
            with Image.open(io.BytesIO(raw_bytes)) as im:
                fmt = (im.format or "").upper()
                if fmt == "PNG":
                    mime = "image/png"
                elif fmt in ("JPEG", "JPG"):
                    mime = "image/jpeg"
                elif fmt == "WEBP":
                    mime = "image/webp"
                else:
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, format="PNG")
                    raw_bytes = buf.getvalue()
                    mime = "image/png"
        return types.Part.from_bytes(data=raw_bytes, mime_type=mime), len(raw_bytes)

    @staticmethod
    def _save_bytes_atomic(data: bytes, dest_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        tmp_path = f"{dest_path}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.replace(tmp_path, dest_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _slugify_prompt(prompt: str, max_len: int = 28) -> str:
        words = re.findall(r"[a-z0-9]+", (prompt or "").lower())
        slug = "_".join(words[:5])[:max_len].strip("_")
        return slug or "image"

    async def generate_image(
        self,
        prompt: str,
        reference_images: Optional[List[str]] = None,
        model: str = "",
        aspect_ratio: str = "",
        image_size: str = "",
        output_library: str = "",
        output_filename: str = "",
    ) -> str:
        """
        Generates a new image or edits/composes from reference images using Gemini Image.
        Saves the output image atomically to a library and attaches a visual preview to your turn.

        Args:
            prompt: Detailed description of the scene, style, composition, or how to use the reference images (e.g. 'Use Reference Image 1 as the character design and depict them in...').
            reference_images: Optional list of 1-14 local image paths to use as character, style, or editing references (e.g. ['discord_file:sketch.png', 'workspace:shared/hero.png']).
            model: Optional Gemini image model override. Defaults to 'gemini-3-pro-image' (also supports 'gemini-nano-banana-2.1', 'gemini-3.1-flash-image', 'gemini-3.1-flash-lite-image').
            aspect_ratio: Optional output aspect ratio ('1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3', '21:9'). Leave empty to let the model choose automatically.
            image_size: Optional resolution tier ('1K', '2K', '4K'). Leave empty for model default.
            output_library: Target library to save the generated image ('discord_file' or 'workspace', or 'shared'). Defaults to 'discord_file'.
            output_filename: Optional relative filename/subpath inside output_library (e.g. 'shared/art/panel_01.png'). Auto-generated if omitted.
        """
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            return "Error: 'prompt' cannot be empty."
        if not self.client:
            return "Error: Gemini Image client is not configured (missing API key)."

        chosen_model = self._normalize_model_name(model or self.default_model)

        clean_ar = (aspect_ratio or "").strip()
        if clean_ar.lower() == "auto":
            clean_ar = ""
        if clean_ar and clean_ar not in VALID_ASPECT_RATIOS:
            return (
                f"Error: Unsupported aspect_ratio '{aspect_ratio}'. "
                f"Supported values: {sorted(VALID_ASPECT_RATIOS)} (or omit for auto)."
            )

        clean_size = (image_size or "").strip().upper()
        if clean_size == "512":
            clean_size = "0.5K"
        if clean_size and clean_size not in VALID_IMAGE_SIZES:
            return "Error: Unsupported image_size. Use '0.5K', '1K', '2K', '4K', or omit."

        ref_specs = self._parse_reference_list(reference_images)
        if len(ref_specs) > 14:
            return "Error: At most 14 reference_images can be passed in a single call."

        target_lib_arg = (output_library or self.default_output_library).strip()
        try:
            # Validate output library and output_filename early before making the cloud API call
            self._resolve_library_key(target_lib_arg)
            if (output_filename or "").strip():
                self._resolve_library_path(target_lib_arg, output_filename.strip(), must_exist=False)
        except Exception as e:
            return f"Error: {e}"

        ref_parts: List[types.Part] = []
        ref_labels: List[str] = []
        total_ref_bytes = 0
        try:
            for spec in ref_specs:
                r_lib, r_abs, r_rel = self._resolve_reference_spec(spec)
                part, nbytes = self._load_reference_part(r_abs)
                ref_parts.append(part)
                ref_labels.append(f"{r_lib}:{r_rel}")
                total_ref_bytes += nbytes
        except Exception as e:
            return f"Error loading reference_images: {e}"

        agent_obj = getattr(self, "agent", None)
        agent_name = (
            getattr(agent_obj, "name", None)
            or self.config.get("agent", {}).get("name", "Thunk")
        )
        thread_id = getattr(agent_obj, "thread_id", None) or "main"

        img_cfg_kwargs: Dict[str, Any] = {}
        if clean_ar:
            img_cfg_kwargs["aspect_ratio"] = clean_ar
        if clean_size:
            img_cfg_kwargs["image_size"] = clean_size

        gen_cfg_kwargs: Dict[str, Any] = {
            "response_modalities": ["TEXT", "IMAGE"],
        }
        if img_cfg_kwargs:
            gen_cfg_kwargs["image_config"] = types.ImageConfig(**img_cfg_kwargs)

        contents: List[Any] = [*ref_parts, clean_prompt]

        with MetricSpan(
            operation="image_gen",
            agent=agent_name,
            thread=thread_id,
            component="image_gen",
            type="generate_image",
            model=chosen_model,
            exec_location="cloud",
        ) as span:
            span.inc("request_bytes", len(clean_prompt.encode("utf-8")) + total_ref_bytes)
            span.inc("reference_images", len(ref_parts))

            try:
                with span.subtimer("llm_call_ms"):
                    response = await self.client.aio.models.generate_content(
                        model=chosen_model,
                        contents=contents,
                        config=types.GenerateContentConfig(**gen_cfg_kwargs),
                    )
            except Exception as e:
                span.fail()
                return f"Error generating image with '{chosen_model}': {e}"

            image_blobs: List[Tuple[bytes, str]] = []
            thought_image_blobs: List[Tuple[bytes, str]] = []
            text_notes: List[str] = []

            candidates = getattr(response, "candidates", None) or []
            for cand in candidates:
                content_obj = getattr(cand, "content", None)
                parts = getattr(content_obj, "parts", None) or []
                for p in parts:
                    is_thought = bool(getattr(p, "thought", False))
                    inline = getattr(p, "inline_data", None)
                    if inline is not None and getattr(inline, "data", None):
                        raw_data = inline.data
                        if isinstance(raw_data, str):
                            raw_data = base64.b64decode(raw_data)
                        mime = str(getattr(inline, "mime_type", None) or "image/png").lower()
                        if is_thought:
                            thought_image_blobs.append((bytes(raw_data), mime))
                        else:
                            image_blobs.append((bytes(raw_data), mime))
                    elif getattr(p, "text", None) and not is_thought:
                        txt = str(p.text).strip()
                        if txt:
                            text_notes.append(txt)

            if not image_blobs and thought_image_blobs:
                image_blobs = [thought_image_blobs[-1]]

            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta is not None:
                req_tok = getattr(usage_meta, "prompt_token_count", None)
                resp_tok = getattr(usage_meta, "candidates_token_count", None)
                thought_tok = getattr(usage_meta, "thoughts_token_count", None)
                if req_tok is not None:
                    span.inc("request_tokens", int(req_tok))
                if resp_tok is not None:
                    span.inc("response_tokens", int(resp_tok))
                if thought_tok is not None:
                    span.inc("thinking_tokens", int(thought_tok))
            else:
                span.inc("request_tokens", count_tokens(clean_prompt, agent=agent_obj))

            if not image_blobs:
                span.fail()
                finish_reasons = [
                    str(getattr(c, "finish_reason", ""))
                    for c in candidates
                    if getattr(c, "finish_reason", None)
                ]
                extra_msg = " ".join(text_notes).strip()
                reason_str = f" (finish_reason={','.join(finish_reasons)})" if finish_reasons else ""
                detail = f": {extra_msg}" if extra_msg else "."
                return f"Error: Model '{chosen_model}' did not return an image{reason_str}{detail}"

            final_bytes, final_mime = image_blobs[-1]
            ext = MIME_TO_EXT.get(final_mime, ".png")

            try:
                with Image.open(io.BytesIO(final_bytes)) as im_check:
                    orig_w, orig_h = im_check.size
                    detected_fmt = (im_check.format or "").upper()
                    if detected_fmt == "PNG":
                        ext = ".png"
                    elif detected_fmt in ("JPEG", "JPG"):
                        ext = ".jpg"
                    elif detected_fmt == "WEBP":
                        ext = ".webp"
            except Exception as e:
                span.fail()
                return f"Error: Returned image bytes could not be decoded: {e}"

            if (output_filename or "").strip():
                base_rel = output_filename.strip()
                stem, given_ext = os.path.splitext(base_rel)
                if given_ext.lower() in EXT_TO_MIME:
                    rel_target = f"{stem}{ext}"
                else:
                    rel_target = f"{base_rel}{ext}"
            else:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                slug = self._slugify_prompt(clean_prompt)
                short_id = uuid.uuid4().hex[:6]
                rel_target = f"gen_{ts}_{slug}_{short_id}{ext}"

            try:
                resolved_lib, dest_abs, display_rel = self._resolve_library_path(
                    target_lib_arg, rel_target, must_exist=False
                )
                self._save_bytes_atomic(final_bytes, dest_abs)
            except Exception as e:
                span.fail()
                return f"Error saving generated image: {e}"

            span.inc("response_bytes", len(final_bytes))
            span.inc("generated_images", 1)

            try:
                preview_jpeg, prev_w, prev_h = downsize_image_bytes(final_bytes, max_edge=1024)
                data_url = encode_jpeg_data_url(preview_jpeg)
                if agent_obj is not None and hasattr(agent_obj, "pending_tool_images"):
                    agent_obj.pending_tool_images.append(data_url)
            except Exception:
                prev_w, prev_h = orig_w, orig_h

            lines = [
                f"Generated image saved to library '{resolved_lib}' as '{display_rel}' "
                f"({orig_w}x{orig_h}px, {len(final_bytes)} bytes, model='{chosen_model}').",
                f"Visual preview ({prev_w}x{prev_h}px) attached to your context for this turn.",
                f"To share on Discord/MQTT: send_message(..., attachment_library='{resolved_lib}', attachment_path='{display_rel}')",
                f"To reuse as a character/style reference later: reference_images=['{resolved_lib}:{display_rel}']",
            ]
            if ref_labels:
                lines.append(f"References used ({len(ref_labels)}): {', '.join(ref_labels)}")
            if text_notes:
                commentary = " ".join(text_notes)
                if len(commentary) > 400:
                    commentary = commentary[:397] + "..."
                lines.append(f"Model note: {commentary}")
            return "\n".join(lines)

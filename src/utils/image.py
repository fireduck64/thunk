import base64
import io
import os
from typing import Tuple
from PIL import Image, ImageOps


def downsize_image_bytes(
    raw_bytes: bytes, max_edge: int = 1024, quality: int = 85
) -> Tuple[bytes, int, int]:
    """
    Loads raw image bytes, normalizes EXIF orientation and RGB mode,
    downsizes so the longest edge is at most `max_edge` pixels,
    and encodes as JPEG. Returns (jpeg_bytes, width, height).
    """
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img = ImageOps.exif_transpose(img)

        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        longest = max(width, height)
        if longest > max_edge:
            scale = max_edge / float(longest)
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width, height = img.size

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue(), width, height


def encode_jpeg_data_url(jpeg_bytes: bytes) -> str:
    """Encodes JPEG bytes into an OpenAI-compatible data URL."""
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def save_image_atomic(jpeg_bytes: bytes, dest_path: str) -> None:
    """Writes image bytes to a temporary file and atomically moves it to dest_path."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    tmp_path = f"{dest_path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(jpeg_bytes)
    os.replace(tmp_path, dest_path)

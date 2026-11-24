import io
import json
import os
from types import SimpleNamespace
import pytest
from PIL import Image, ImageDraw

from src.core.agent import AgentCore
from src.components.image_gen import DEFAULT_IMAGE_MODEL, ImageGenComponent
from src.utils.schema import function_to_schema


def _make_png_bytes(width: int = 256, height: int = 256, color=(40, 120, 220)) -> bytes:
    im = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(im)
    m = max(4, min(width, height) // 5)
    draw.ellipse((m, m, width - m, height - m), fill=(250, 210, 80))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _mock_genai_response(final_png: bytes, commentary: str = "Rendered character scene.", include_thought_draft: bool = False):
    parts = []
    if include_thought_draft:
        draft_png = _make_png_bytes(64, 64, color=(255, 0, 0))
        parts.append(
            SimpleNamespace(
                thought=True,
                text="Planning composition...",
                inline_data=None,
            )
        )
        parts.append(
            SimpleNamespace(
                thought=True,
                text=None,
                inline_data=SimpleNamespace(data=draft_png, mime_type="image/png"),
            )
        )
    if commentary:
        parts.append(SimpleNamespace(thought=False, text=commentary, inline_data=None))
    parts.append(
        SimpleNamespace(
            thought=False,
            text=None,
            inline_data=SimpleNamespace(data=final_png, mime_type="image/png"),
        )
    )
    return SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=parts))],
        usage_metadata=SimpleNamespace(
            prompt_token_count=320,
            candidates_token_count=1290,
            thoughts_token_count=180,
        ),
    )


def test_generate_image_tool_schema_has_array_items(tmp_path):
    """
    Verifies function_to_schema produces valid 'items' for Optional[List[str]]
    so Gemini's strict FunctionDeclaration validator accepts generate_image.
    """
    comp = ImageGenComponent(
        api_key="dummy-key",
        libraries={"discord_file": str(tmp_path / "images")},
        client=SimpleNamespace(),
    )
    schema = function_to_schema(comp.generate_image)
    fn_def = schema["function"]
    assert fn_def["name"] == "generate_image"
    assert fn_def["parameters"]["required"] == ["prompt"]

    ref_prop = fn_def["parameters"]["properties"]["reference_images"]
    assert ref_prop["type"] == "array"
    assert ref_prop["items"] == {"type": "string"}


@pytest.mark.asyncio
async def test_default_pro_model_override_and_thought_part_filtering(tmp_path):
    """
    Verifies:
    - Defaults to 'gemini-3-pro-image'.
    - Allows per-call model selection (explicit model ID or alias).
    - Ignores intermediate thought=True draft images/text and saves the final image.
    - Attaches a visual preview data URL to agent.pending_tool_images.
    """
    discord_dir = tmp_path / "discord_images"
    ws_dir = tmp_path / "workspace"
    shared_dir = tmp_path / "shared_workspace"
    discord_dir.mkdir()
    ws_dir.mkdir()
    shared_dir.mkdir()

    calls = []
    final_img_bytes = _make_png_bytes(512, 384, color=(20, 160, 90))

    async def _fake_generate_content(**kwargs):
        calls.append(kwargs)
        return _mock_genai_response(
            final_img_bytes,
            commentary="Final owl character portrait.",
            include_thought_draft=True,
        )

    mock_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=_fake_generate_content))
    )

    comp = ImageGenComponent(
        api_key="dummy-key",
        libraries={"discord_file": str(discord_dir), "workspace": str(ws_dir)},
        shared_workspace_dir=str(shared_dir),
        client=mock_client,
    )
    assert comp.default_model == DEFAULT_IMAGE_MODEL == "gemini-3-pro-image"

    agent = AgentCore(thread_id="social")
    agent.register_component(comp)
    agent._build_tools()

    # 1. Default model call -> gemini-3-pro-image
    res1 = await agent.execute_tool(
        "generate_image",
        json.dumps(
            {
                "prompt": "A clockwork owl wearing brass spectacles",
                "aspect_ratio": "4:3",
            }
        ),
    )
    assert "model='gemini-3-pro-image'" in res1
    assert "512x384px" in res1  # saved the 512x384 final image, NOT the 64x64 thought draft!
    assert "Planning composition" not in res1
    assert "Final owl character portrait." in res1
    assert calls[-1]["model"] == "gemini-3-pro-image"
    assert calls[-1]["config"].image_config.aspect_ratio == "4:3"
    assert len(agent.pending_tool_images) == 1
    assert agent.pending_tool_images[0].startswith("data:image/jpeg;base64,")

    # 2. Per-call model override -> gemini-nano-banana-2.1
    agent.pending_tool_images.clear()
    res2 = await comp.generate_image(
        prompt="Quick sketch of the same owl",
        model="gemini-nano-banana-2.1",
        output_library="workspace",
        output_filename="shared/art/owl_ref.png",
    )
    assert "model='gemini-nano-banana-2.1'" in res2
    assert calls[-1]["model"] == "gemini-nano-banana-2.1"
    assert (shared_dir / "art" / "owl_ref.png").is_file()
    assert "attachment_library='workspace', attachment_path='shared/art/owl_ref.png'" in res2


@pytest.mark.asyncio
async def test_multi_reference_character_generation_and_security_guards(tmp_path):
    """
    Verifies:
    - Multi-image character + prop references resolve across discord_file and workspace:shared/...
    - Reference image parts precede the prompt in contents passed to generate_content.
    - Comma-separated string fallback for reference_images works cleanly.
    - Path traversal ('..') and missing reference files fail fast before calling Gemini API.
    """
    discord_dir = tmp_path / "discord_images"
    ws_dir = tmp_path / "workspace"
    shared_dir = tmp_path / "shared_workspace"
    discord_dir.mkdir()
    ws_dir.mkdir()
    (shared_dir / "chars").mkdir(parents=True)

    char_bytes = _make_png_bytes(200, 200, color=(120, 40, 180))
    prop_bytes = _make_png_bytes(180, 180, color=(200, 140, 30))
    (discord_dir / "incoming_sketch.png").write_bytes(char_bytes)
    (shared_dir / "chars" / "potato_monster.png").write_bytes(prop_bytes)

    calls = []

    async def _fake_generate_content(**kwargs):
        calls.append(kwargs)
        return _mock_genai_response(_make_png_bytes(640, 360))

    mock_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=_fake_generate_content))
    )

    comp = ImageGenComponent(
        api_key="dummy-key",
        libraries={"discord_file": str(discord_dir), "workspace": str(ws_dir)},
        shared_workspace_dir=str(shared_dir),
        client=mock_client,
    )

    # Path traversal rejected with 0 API calls
    err_traversal = await comp.generate_image(
        prompt="bad output path",
        output_filename="../../escape.png",
    )
    assert "Access denied" in err_traversal
    assert len(calls) == 0

    # Missing reference file rejected with 0 API calls
    err_missing = await comp.generate_image(
        prompt="use missing ref",
        reference_images=["discord_file:does_not_exist.png"],
    )
    assert "does not exist" in err_missing
    assert len(calls) == 0

    # Valid multi-reference call (1 bare filename in discord_file + 1 workspace:shared/... path)
    res = await comp.generate_image(
        prompt="Use Reference 1 as the character and Reference 2 as her companion under a starry sky.",
        reference_images=["incoming_sketch.png", "workspace:shared/chars/potato_monster.png"],
        aspect_ratio="16:9",
        output_filename="duet_scene.png",
    )
    assert len(calls) == 1
    sent_contents = calls[0]["contents"]
    assert len(sent_contents) == 3
    assert sent_contents[0].inline_data.data == char_bytes
    assert sent_contents[1].inline_data.data == prop_bytes
    assert "Use Reference 1 as the character" in sent_contents[2]
    assert "References used (2): discord_file:incoming_sketch.png, workspace:shared/chars/potato_monster.png" in res
    assert (discord_dir / "duet_scene.png").is_file()

    # String-formatted reference list fallback
    res_str_refs = await comp.generate_image(
        prompt="Another angle of the character",
        reference_images="discord_file:incoming_sketch.png, shared/chars/potato_monster.png",
    )
    assert "References used (2):" in res_str_refs
    assert len(calls) == 2

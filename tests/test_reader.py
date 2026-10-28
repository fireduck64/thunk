import io
import json
import os
import fitz
import pytest
from PIL import Image

from src.core.agent import AgentCore
from src.components.file_reader import FileReaderComponent
from src.utils.config import load_config
from src.utils.image import downsize_image_bytes, encode_jpeg_data_url, save_image_atomic


def test_unified_file_reader_epub_and_text_chunks():
    """Tests unified FileReaderComponent EPUB metadata, listing/search, chunk reading, read_next_chunk, and bookmark persistence."""
    cfg = load_config()
    db_path = cfg["file_reader"]["db_path"]
    if os.path.exists(db_path):
        os.remove(db_path)

    reader = FileReaderComponent(chunk_size=300, lines_per_chunk=20)

    # 1. Verify list_libraries includes ebook, pdf, discord_file, thunk_source
    libs_out = reader.list_libraries()
    assert "ebook" in libs_out
    assert "pdf" in libs_out
    assert "discord_file" in libs_out
    assert "thunk_source" in libs_out

    # 2. Verify list_files and query search on 'ebook' library
    listing = reader.list_files("ebook")
    assert "Pride and Prejudice: Test Edition by Jane Austen [path: MrDarcy.epub]" in listing

    search_out = reader.list_files("ebook", query="Austen, Jane")
    assert "Pride and Prejudice: Test Edition by Jane Austen [path: MrDarcy.epub]" in search_out

    # 3. Verify read_file opens EPUB at chunk 1, returns total_chunks and next_hint, and sets bookmark
    c1 = json.loads(reader.read_file("ebook", "MrDarcy.epub", chunk=1))
    assert c1["format"] == "epub"
    assert c1["chunk"] == 1
    assert c1["total_chunks"] >= 2
    assert "universally acknowledged" in c1["content"]
    assert "read_next_chunk()" in c1["next_hint"]
    assert reader.current_index == 1

    # 4. Verify read_next_chunk advances to chunk 2 of the active EPUB
    c2 = json.loads(reader.read_next_chunk())
    assert c2["format"] == "epub"
    assert c2["chunk"] == 2
    assert reader.current_index == 2

    # 5. Verify bookmark persistence across component re-instantiation
    reader_reloaded = FileReaderComponent(chunk_size=300)
    assert reader_reloaded.current_index == 2
    assert reader_reloaded.current_library == "ebook"
    assert "Pride and Prejudice: Test Edition" in reader_reloaded.current_book_label

    # 6. Verify chunked text/code file reading
    txt1 = json.loads(reader.read_file("thunk_source", "main.py", chunk=1))
    assert txt1["format"] == "text"
    assert txt1["chunk"] == 1
    assert txt1["total_chunks"] >= 2
    assert txt1["start_line"] == 1
    assert txt1["end_line"] == 20
    assert "1: " in txt1["content"]
    assert "chunk=2" in txt1["next_hint"]


@pytest.mark.asyncio
async def test_image_downsize_save_and_ephemeral_vision():
    """Tests 1024px image downsizing, atomic disk storage, view_image tool on discord_file, and ephemeral vision messages."""
    cfg = load_config()
    images_dir = cfg["file_reader"]["libraries"]["discord_file"]
    os.makedirs(images_dir, exist_ok=True)

    # 1. Create a 2400x1200 RGBA PNG in memory and verify downsizing to 1024x512 JPEG
    raw_buf = io.BytesIO()
    img = Image.new("RGBA", (2400, 1200), (20, 120, 220, 255))
    img.save(raw_buf, format="PNG")
    jpeg_bytes, w, h = downsize_image_bytes(raw_buf.getvalue(), max_edge=1024)
    assert (w, h) == (1024, 512)

    saved_file = os.path.join(images_dir, "test_attachment.jpg")
    save_image_atomic(jpeg_bytes, saved_file)
    assert os.path.exists(saved_file)

    try:
        # 2. Verify AgentCore attaches image on first LLM preparation and strips it after
        agent = AgentCore()
        fr = FileReaderComponent()
        agent.register_component(fr)

        data_url = encode_jpeg_data_url(jpeg_bytes)
        await agent.resume("Look at this image", images=[data_url])
        await agent._drain_inbox()

        wire_msgs = agent._prepare_llm_messages()
        assert isinstance(wire_msgs[-1]["content"], list)
        assert wire_msgs[-1]["content"][0]["type"] == "text"
        assert wire_msgs[-1]["content"][1]["type"] == "image_url"
        assert wire_msgs[-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

        # After the LLM call, _clear_ephemeral_images strips the base64 payload from working memory
        agent._clear_ephemeral_images()
        wire_msgs_after = agent._prepare_llm_messages()
        assert isinstance(wire_msgs_after[-1]["content"], str)

        # 3. Verify read_file on an image returns image dimensions and next_hint to call view_image
        img_meta = json.loads(fr.read_file("discord_file", "test_attachment.jpg"))
        assert img_meta["format"] == "image"
        assert (img_meta["width"], img_meta["height"]) == (1024, 512)
        assert "view_image" in img_meta["next_hint"]

        # 4. Verify view_image loads the saved image from 'discord_file' into pending_tool_images
        res = fr.view_image("discord_file", "test_attachment.jpg")
        assert "1024x512px" in res
        assert len(agent.pending_tool_images) == 1
        assert agent.pending_tool_images[0].startswith("data:image/jpeg;base64,")
    finally:
        if os.path.exists(saved_file):
            os.remove(saved_file)


def test_unified_file_reader_pdf_read_next_and_view_pages():
    """Tests PDF listing/search, page reading via read_file and read_next_chunk, visual detection, and page rendering via view_image."""
    cfg = load_config()
    pdf_lib = cfg["file_reader"]["libraries"]["pdf"]
    db_path = cfg["file_reader"]["db_path"]
    os.makedirs(pdf_lib, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    pdf_path = os.path.join(pdf_lib, "DungeonMap.pdf")
    try:
        # Build a 2-page test PDF: Page 1 text-only, Page 2 with text + embedded image
        doc = fitz.open()
        doc.set_metadata({"title": "Larracos Tactical Atlas", "author": "Mordecai"})

        p1 = doc.new_page(width=612, height=792)
        p1.insert_text((72, 72), "Page 1: Overview of the Larracos Saferoom Network.")

        p2 = doc.new_page(width=612, height=792)
        p2.insert_text((72, 72), "Page 2: Floor 9 Diagram and Bait Shop Entrance.")
        img_buf = io.BytesIO()
        Image.new("RGB", (200, 150), (180, 40, 40)).save(img_buf, format="PNG")
        p2.insert_image(fitz.Rect(72, 120, 272, 270), stream=img_buf.getvalue())

        tmp_pdf = f"{pdf_path}.tmp"
        doc.save(tmp_pdf)
        doc.close()
        os.replace(tmp_pdf, pdf_path)

        agent = AgentCore()
        fr = FileReaderComponent()
        agent.register_component(fr)

        # 1. Verify list_files and query search on 'pdf' library
        listing = fr.list_files("pdf")
        assert "Larracos Tactical Atlas by Mordecai (2 pages) [path: DungeonMap.pdf]" in listing
        search_res = fr.list_files("pdf", query="Mordecai")
        assert "DungeonMap.pdf" in search_res

        # 2. Read Page 1 via read_file(library='pdf', filepath='DungeonMap.pdf', chunk=1)
        p1_json = json.loads(fr.read_file("pdf", "DungeonMap.pdf", chunk=1))
        assert p1_json["format"] == "pdf"
        assert p1_json["chunk"] == 1
        assert p1_json["total_pages"] == 2
        assert p1_json["total_chunks"] == 2
        assert "Larracos Saferoom Network" in p1_json["content"]
        assert p1_json["has_unrenderable_visuals"] is False
        assert "read_next_chunk()" in p1_json["next_hint"]

        # 3. Advance to Page 2 via read_next_chunk() and verify embedded image detection
        p2_json = json.loads(fr.read_next_chunk())
        assert p2_json["chunk"] == 2
        assert p2_json["page_number"] == 2
        assert "Bait Shop Entrance" in p2_json["content"]
        assert p2_json["has_unrenderable_visuals"] is True
        assert p2_json["image_count"] == 1
        assert p2_json["images"] == [{"width": 200, "height": 150}]
        assert "view_image" in p2_json["notice"]

        # 4. Render Page 2 via view_image(library='pdf', filepath='DungeonMap.pdf', chunk=2)
        view_res = fr.view_image("pdf", "DungeonMap.pdf", chunk=2)
        assert "Rendered page 2/2" in view_res
        assert len(agent.pending_tool_images) == 1
        assert agent.pending_tool_images[0].startswith("data:image/jpeg;base64,")
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@pytest.mark.asyncio
async def test_discord_send_message_with_library_attachment():
    """Verifies SystemComponent.send_message and DiscordAdapter._on_send_to_operator attach library files safely."""
    from src.components.system import SystemComponent
    from src.adapters.discord_adapter import DiscordAdapter

    cfg = load_config()
    images_dir = cfg["file_reader"]["libraries"]["discord_file"]
    os.makedirs(images_dir, exist_ok=True)
    test_img_path = os.path.join(images_dir, "outbound_chart.png")

    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (50, 100, 200)).save(buf, format="PNG")
    save_image_atomic(buf.getvalue(), test_img_path)

    try:
        agent = AgentCore()
        sys_comp = SystemComponent()
        agent.register_component(sys_comp)
        adapter = DiscordAdapter(agent)
        adapter.last_channel_id = 999888777
        agent.event_bus.subscribe("send_to_operator", adapter._on_send_to_operator)

        sent_calls = []

        class DummySentMsg:
            id = 555444333

        class DummyChannel:
            id = 999888777

            async def send(self, content=None, **kwargs):
                sent_calls.append((content, kwargs))
                return DummySentMsg()

        adapter.client.get_channel = lambda cid: DummyChannel() if cid == 999888777 else None

        # 1. Path traversal attempt is rejected
        err_res = await sys_comp.send_message(
            "Trying traversal",
            medium="discord",
            attachment_library="discord_file",
            attachment_path="../../etc/passwd",
        )
        assert "Access denied" in err_res
        assert len(sent_calls) == 0

        # 2. Valid attachment from 'discord_file' library sends discord.File
        ok_res = await sys_comp.send_message(
            "Here is the chart!",
            medium="discord",
            attachment_library="discord_file",
            attachment_path="outbound_chart.png",
        )
        assert "Message sent successfully via discord" in ok_res
        assert "attached='outbound_chart.png'" in ok_res
        assert len(sent_calls) == 1
        sent_text, sent_kwargs = sent_calls[0]
        assert sent_text == "Here is the chart!"
        assert "file" in sent_kwargs
        assert sent_kwargs["file"].filename == "outbound_chart.png"
    finally:
        if os.path.exists(test_img_path):
            os.remove(test_img_path)


def test_gutenberg_gutindex_catalog_and_cross_library_search(tmp_path):
    """Verifies GUTINDEX.ALL SQLite indexing, eBook ID resolution, text read_next_chunk, and cross-library search."""
    cfg = load_config()
    ebook_lib = cfg["file_reader"]["libraries"]["ebook"]
    gutenberg_dir = tmp_path / "gutenberg"
    gutenberg_dir.mkdir()

    # Write synthetic GUTINDEX.ALL
    gutindex_content = (
        "TITLE and AUTHOR                                                     EBOOK NO.\n\n"
        "Pride and Prejudice, by Jane Austen                                       1342\n\n"
        "Frankenstein, by Mary Wollstonecraft (Godwin) Shelley                       84\n"
        " [Subtitle: or The Modern Prometheus]\n\n"
        "The temple, by H. P. (Howard Phillips) Lovecraft                         79842\n"
    )
    (gutenberg_dir / "GUTINDEX.ALL").write_text(gutindex_content, encoding="utf-8")

    # Create tiered directory 1/3/4/1342/1342-0.txt with 15 lines
    book_1342_dir = gutenberg_dir / "1" / "3" / "4" / "1342"
    book_1342_dir.mkdir(parents=True)
    lines = [f"Gutenberg Pride & Prejudice Line {i}" for i in range(1, 16)]
    (book_1342_dir / "1342-0.txt").write_text("\n".join(lines), encoding="utf-8")

    db_path = str(tmp_path / "gutenberg_state.db")
    fr = FileReaderComponent(
        libraries={"ebook": ebook_lib, "gutenberg": str(gutenberg_dir)},
        db_path=db_path,
        lines_per_chunk=5,
    )

    # 1. Search Gutenberg catalog via GUTINDEX.ALL (including typo alias 'gutenburg')
    g_search = fr.list_files("gutenburg", query="Lovecraft")
    assert "The temple, by H. P. (Howard Phillips) Lovecraft (#79842) [path: 7/9/8/4/79842/79842-0.txt]" in g_search

    # 2. Cross-library search (library='all') finds Jane Austen in both 'ebook' and 'gutenberg'
    cross_search = fr.list_files("all", query="Austen")
    assert "[ebook] Pride and Prejudice: Test Edition by Jane Austen [path: MrDarcy.epub]" in cross_search
    assert "[gutenberg] Pride and Prejudice, by Jane Austen (#1342) [path: 1/3/4/1342/1342-0.txt]" in cross_search

    # 3. Read Gutenberg book by eBook ID ('1342') and advance with read_next_chunk()
    c1 = json.loads(fr.read_file("gutenberg", "1342", chunk=1))
    assert c1["format"] == "text"
    assert c1["filepath"] == "1/3/4/1342/1342-0.txt"
    assert c1["chunk"] == 1
    assert c1["total_chunks"] == 3
    assert "Gutenberg Pride & Prejudice Line 1" in c1["content"]
    assert "read_next_chunk()" in c1["next_hint"]

    c2 = json.loads(fr.read_next_chunk())
    assert c2["chunk"] == 2
    assert "Gutenberg Pride & Prejudice Line 6" in c2["content"]


def test_file_reader_and_discord_shared_workspace_routing(tmp_path):
    """Verifies FileReaderComponent (list_files, read_file, view_image) and DiscordAdapter route workspace shared/ to shared_workspace_dir."""
    ws_dir = tmp_path / "agent_ws"
    shared_dir = tmp_path / "shared_ws"
    ws_dir.mkdir()
    shared_dir.mkdir()
    # Create empty mountpoint stub inside ws_dir/shared as Docker does
    (ws_dir / "shared").mkdir()

    (shared_dir / "joint_report.txt").write_text("Line 1 of joint report\nLine 2 of joint report", encoding="utf-8")
    Image.new("RGB", (500, 400), (120, 40, 180)).save(str(shared_dir / "joint_chart.png"), format="PNG")

    fr = FileReaderComponent(
        libraries={"workspace": str(ws_dir)},
        db_path=str(tmp_path / "reader_shared.db"),
        shared_workspace_dir=str(shared_dir),
    )
    agent = AgentCore()
    agent.register_component(fr)

    # 1. list_files on workspace root and workspace/shared
    root_listing = fr.list_files("workspace")
    assert "[DIR]  shared" in root_listing
    shared_listing = fr.list_files("workspace", path="shared")
    assert "[FILE] joint_report.txt" in shared_listing
    assert "[FILE] joint_chart.png" in shared_listing

    # 2. Cross-library search finds files inside shared_workspace_dir under workspace
    search_res = fr.list_files("all", query="joint_report")
    assert "[workspace] shared/joint_report.txt" in search_res

    # 3. read_file and view_image work via both library='workspace', filepath='shared/...' and library='shared'
    read_res = json.loads(fr.read_file("workspace", "shared/joint_report.txt", chunk=1))
    assert read_res["filepath"] == "shared/joint_report.txt"
    assert "Line 1 of joint report" in read_res["content"]

    img_res = fr.view_image("shared", "joint_chart.png")
    assert "Loaded image 'shared/joint_chart.png' from library 'workspace' (500x400px)" in img_res
    assert len(agent.pending_tool_images) == 1




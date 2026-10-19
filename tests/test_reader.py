import os
import pytest
from src.utils.config import load_config
from src.components.reader import SequentialReaderComponent


def test_sequential_reader_metadata_and_chunks():
    """Tests EPUB metadata extraction, caching, opening, and chunk headers in isolated data/test."""
    cfg = load_config()
    db_path = cfg["reader"]["db_path"]
    if os.path.exists(db_path):
        os.remove(db_path)

    reader = SequentialReaderComponent(chunk_size=300)

    # 1. Verify library catalog includes title, author, and relative path
    catalog = reader.list_library()
    assert len(catalog) >= 1
    assert any("Pride and Prejudice: Test Edition by Jane Austen [path: MrDarcy.epub]" in item for item in catalog)

    # 2. Verify search_library matches against title/author metadata as well as path
    matches = reader.search_library("Jane Austen")
    assert len(matches) == 1
    assert matches[0] == "Pride and Prejudice: Test Edition by Jane Austen [path: MrDarcy.epub]"

    # 3. Verify open_book accepts the full catalog entry and sets the book label
    open_msg = reader.open_book(matches[0])
    assert "Successfully opened" in open_msg
    assert "Pride and Prejudice: Test Edition by Jane Austen" in open_msg
    assert len(reader.current_chunks) >= 2

    # 4. Verify read_next_chunk and read_previous_chunk include the book label in the header
    chunk1 = reader.read_next_chunk()
    assert chunk1.startswith(
        "--- [Book: Pride and Prejudice: Test Edition by Jane Austen [path: MrDarcy.epub]] Chunk 1/"
    )
    assert "universally acknowledged" in chunk1
    assert reader.current_index == 1

    chunk2 = reader.read_next_chunk()
    assert "Chunk 2/" in chunk2
    assert reader.current_index == 2

    prev_chunk = reader.read_previous_chunk()
    assert "Chunk 1/" in prev_chunk
    assert reader.current_index == 1

    # 5. Verify bookmark persistence across component re-instantiation
    reader_reloaded = SequentialReaderComponent(chunk_size=300)
    assert reader_reloaded.current_index == 1
    assert "Pride and Prejudice: Test Edition" in reader_reloaded.current_book_label


@pytest.mark.asyncio
async def test_image_downsize_save_and_ephemeral_vision():
    """Tests 1024px image downsizing, atomic disk storage, view_image tool, and ephemeral vision messages."""
    import io
    from PIL import Image
    from src.core.agent import AgentCore
    from src.components.file_reader import FileReaderComponent
    from src.utils.image import downsize_image_bytes, encode_jpeg_data_url, save_image_atomic

    cfg = load_config()
    images_dir = cfg["discord"]["images_dir"]
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

        # 3. Verify view_image re-loads the saved image into pending_tool_images
        res = fr.view_image("discord_images", "test_attachment.jpg")
        assert "1024x512px" in res
        assert len(agent.pending_tool_images) == 1
        assert agent.pending_tool_images[0].startswith("data:image/jpeg;base64,")
    finally:
        if os.path.exists(saved_file):
            os.remove(saved_file)


def test_pdf_viewer_read_and_view_pages():
    """Tests PdfViewerComponent catalog, JSON page extraction, visual detection, and page rendering."""
    import io
    import json
    import fitz
    from PIL import Image
    from src.core.agent import AgentCore
    from src.components.pdf_viewer import PdfViewerComponent

    cfg = load_config()
    pdf_lib = cfg["pdf_viewer"]["library_dir"]
    pdf_db = cfg["pdf_viewer"]["db_path"]
    os.makedirs(pdf_lib, exist_ok=True)
    if os.path.exists(pdf_db):
        os.remove(pdf_db)

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
        viewer = PdfViewerComponent()
        agent.register_component(viewer)

        # 1. Verify list_pdfs and search_pdfs
        catalog = viewer.list_pdfs()
        assert any("Larracos Tactical Atlas by Mordecai (2 pages) [path: DungeonMap.pdf]" in c for c in catalog)
        matches = viewer.search_pdfs("Mordecai")
        assert len(matches) == 1

        # 2. Read Page 1 (pure text, no unrenderable visuals)
        p1_json = json.loads(viewer.read_pdf_page(page_number=1, filename=matches[0]))
        assert p1_json["page_number"] == 1
        assert p1_json["total_pages"] == 2
        assert "Larracos Saferoom Network" in p1_json["text"]
        assert p1_json["has_unrenderable_visuals"] is False
        assert p1_json["image_count"] == 0

        # 3. Read Page 2 with filename omitted (uses active PDF) and verify image detection
        p2_json = json.loads(viewer.read_pdf_page(page_number=2))
        assert p2_json["page_number"] == 2
        assert "Bait Shop Entrance" in p2_json["text"]
        assert p2_json["has_unrenderable_visuals"] is True
        assert p2_json["image_count"] == 1
        assert p2_json["images"] == [{"width": 200, "height": 150}]
        assert "view_pdf_page" in p2_json["notice"]

        # 4. View Page 2 (renders page to <=1024px JPEG and attaches to agent.pending_tool_images)
        view_res = viewer.view_pdf_page(page_number=2)
        assert "Rendered page 2/2" in view_res
        assert len(agent.pending_tool_images) == 1
        assert agent.pending_tool_images[0].startswith("data:image/jpeg;base64,")
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)





import os
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



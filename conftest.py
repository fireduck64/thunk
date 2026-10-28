import sys
import os
import pytest
from ebooklib import epub

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

from src.utils.config import set_config_path, load_config

TEST_DATA_DIR = os.path.join(ROOT_DIR, "data", "test")
DEFAULT_TEST_CONFIG = os.path.join(TEST_DATA_DIR, "config.toml")


def _ensure_test_environment(config_path: str) -> None:
    """Ensures data/test directories, default test config, and sample EPUB exist."""
    os.makedirs(os.path.join(TEST_DATA_DIR, "library"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "pdf_library"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "skills"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "workspace"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "shared_workspace"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "mind_ro"), exist_ok=True)

    directives_path = os.path.join(TEST_DATA_DIR, "directives.md")
    if not os.path.exists(directives_path):
        tmp_dir = f"{directives_path}.tmp"
        with open(tmp_dir, "w", encoding="utf-8") as f:
            f.write("# Test Agent Directives\nYou are TestAgent, running inside the automated test suite.\n")
        os.replace(tmp_dir, directives_path)

    need_write_config = not os.path.exists(config_path)
    if not need_write_config:
        with open(config_path, "r", encoding="utf-8") as f:
            existing_cfg_text = f.read()
        if "discord_file" not in existing_cfg_text or "[reader]" in existing_cfg_text or "metrics_dir" not in existing_cfg_text:
            need_write_config = True

    if need_write_config:
        tmp_cfg = f"{config_path}.tmp"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            f.write(
                '[agent]\nname = "TestAgent"\n\n'
                '[directives]\nfile_path = "data/test/directives.md"\n\n'
                '[llm]\napi_url = "http://localhost:11434/v1"\napi_key = "test"\nmodel_name = "gemma4:31b"\n\n'
                '[subconscious]\napi_url = "http://localhost:11434/v1"\napi_key = "test"\nmodel_name = "gemma4:e2b"\ntop_k = 8\n\n'
                '[tokenizer]\nmodel_name = "unsloth/gemma-7b"\n\n'
                '[memory]\nhigh_watermark = 300\nlow_watermark = 150\nmax_summarize_tokens = 300\nmax_summary_tokens = 200\ndb_path = "data/test/memory.db"\n\n'
                '[notes]\ndb_path = "data/test/notes.db"\n\n'
                '[tasks]\ndb_path = "data/test/tasks.db"\n\n'
                '[skills]\nskills_dir = "data/test/skills"\n\n'
                '[logging]\nlog_dir = "data/test/logs"\nmetrics_dir = "data/test/metrics"\nverbose = false\n\n'
                '[file_reader]\ndb_path = "data/test/reader_state.db"\n\n'
                '[file_reader.libraries]\nebook = "data/test/library"\npdf = "data/test/pdf_library"\ndiscord_file = "data/test/images"\nworkspace = "data/test/workspace"\nthunk_source = "src"\n\n'
                '[embeddings]\napi_url = "http://localhost:11434/api/embeddings"\napi_key = ""\nmodel_name = "nomic-embed-text"\n\n'
                '[vectordb]\nhost = "localhost"\nport = 6333\nknowledge_collection = "test_knowledge_base"\nmemory_collection = "test_agent_memory"\n\n'
                '[mqtt]\nhost = "localhost"\nport = 1883\ninbox_topic = "test/agent/inbox"\noutbox_topic = "test/agent/outbox"\n\n'
                '[discord]\nbot_token = ""\nimages_dir = "data/test/images"\n\n'
                '[oracle]\ngemini_api_key = ""\ngemini_model = "gemini-3.8-flash"\n\n'
                '[debug]\nstream_output = false\n'
            )
        os.replace(tmp_cfg, config_path)

    sample_epub = os.path.join(TEST_DATA_DIR, "library", "MrDarcy.epub")
    if not os.path.exists(sample_epub):
        book = epub.EpubBook()
        book.set_identifier("test-mrdarcy-001")
        book.set_title("Pride and Prejudice: Test Edition")
        book.set_language("en")
        book.add_author("Jane Austen")

        c1 = epub.EpubHtml(title="Chapter 1", file_name="chap_01.xhtml", lang="en")
        c1.content = (
            "<html><body><h1>Chapter 1</h1>"
            "<p>It is a truth universally acknowledged, that a single man in possession of a good fortune, "
            "must be in want of a wife. Mr. Bingley recently arrived at Netherfield Park with his friend Mr. Darcy. "
            + ("Elizabeth Bennet observed Mr. Darcy at the Meryton assembly with keen wit. " * 60)
            + "</p></body></html>"
        )
        book.add_item(c1)
        book.toc = (epub.Link("chap_01.xhtml", "Chapter 1", "chap_01"),)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", c1]

        tmp_epub = f"{sample_epub}.tmp"
        epub.write_epub(tmp_epub, book)
        os.replace(tmp_epub, sample_epub)


def _verify_test_config_isolation(cfg: dict) -> None:
    """Refuses to run tests if any configured path or Qdrant collection points outside test namespaces."""
    abs_test_dir = os.path.abspath(TEST_DATA_DIR)
    libs = cfg.get("file_reader", {}).get("libraries", {})
    paths_to_check = [
        cfg.get("directives", {}).get("file_path", ""),
        cfg.get("memory", {}).get("db_path", ""),
        cfg.get("notes", {}).get("db_path", ""),
        cfg.get("tasks", {}).get("db_path", ""),
        cfg.get("file_reader", {}).get("db_path", "data/test/reader_state.db"),
        libs.get("ebook", "data/test/library"),
        libs.get("pdf", "data/test/pdf_library"),
        libs.get("discord_file", "data/test/images"),
        cfg.get("skills", {}).get("skills_dir", ""),
        cfg.get("logging", {}).get("log_dir", ""),
        cfg.get("logging", {}).get("metrics_dir", "data/test/metrics"),
        cfg.get("discord", {}).get("images_dir", "data/test/images"),
        cfg.get("workspace", {}).get("workspace_dir", "data/test/workspace"),
        cfg.get("workspace", {}).get("shared_workspace_dir", "data/test/shared_workspace"),
        cfg.get("workspace", {}).get("mind_ro_dir", "data/test/mind_ro"),
        cfg.get("workspace", {}).get("live_db_path", "data/test/memory.db"),
    ]
    for p in paths_to_check:
        abs_p = os.path.abspath(p)
        if not abs_p.startswith(abs_test_dir + os.sep) and abs_p != abs_test_dir:
            raise RuntimeError(
                f"SAFETY ABORT: Test config path '{p}' ({abs_p}) is outside '{abs_test_dir}'!"
            )

    vectordb = cfg.get("vectordb", {})
    for key in ("knowledge_collection", "memory_collection"):
        col = vectordb.get(key, "")
        if not col.startswith("test_"):
            raise RuntimeError(
                f"SAFETY ABORT: Qdrant collection '{key} = {col}' in test config must start with 'test_'!"
            )


def pytest_configure(config):
    config_path = os.environ.get("THUNK_TEST_CONFIG", DEFAULT_TEST_CONFIG)
    _ensure_test_environment(config_path)
    set_config_path(config_path)
    cfg = load_config()
    _verify_test_config_isolation(cfg)



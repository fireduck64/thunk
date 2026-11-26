import asyncio
import base64
import fcntl
import io
import json
import os
import stat
from pathlib import Path

import pytest
from PIL import Image

from src.core.agent import AgentCore
from src.components.file_reader import FileReaderComponent
from src.components.video_watcher import (
    LobsterWatchManager,
    VideoWatcherComponent,
    compute_lobster_job_id,
)


def _make_jpeg_bytes(width: int = 1024, height: int = 768, color=(24, 48, 96)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _write_sample_package(
    packages_root: Path,
    rel_no_ext: str,
    source_path: Path,
    display_label: str = "Star Trek: Deep Space Nine S01E02 - Past Prologue",
    series_name: str = "Star Trek: Deep Space Nine",
    title: str = "Past Prologue",
    synopsis: str = "A reunion with a member of the Bajoran underground tests Kira's loyalty.",
    cast_names: tuple = ("Kira Nerys", "Elim Garak", "Julian Bashir"),
) -> bytes:
    pkg_dir = packages_root / rel_no_ext
    pages_dir = pkg_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    jpg1 = _make_jpeg_bytes(1024, 768, (30, 60, 110))
    jpg2 = _make_jpeg_bytes(1024, 768, (80, 40, 90))
    (pages_dir / "page_001.jpg").write_bytes(jpg1)
    (pages_dir / "page_002.jpg").write_bytes(jpg2)

    (pages_dir / "page_001.md").write_text(
        f"# {display_label} — Page 1/2 (00:00 – 00:39)\n"
        "Image: `pages/page_001.jpg`\n\n"
        "### [Panel 1 — 00:15]\n"
        "**Elim Garak** [00:18]: It's Dr. Bashir, isn't it?\n"
        "**Julian Bashir** [00:24]: Uh... yes.\n",
        encoding="utf-8",
    )
    (pages_dir / "page_002.md").write_text(
        f"# {display_label} — Page 2/2 (00:39 – 01:20)\n"
        "Image: `pages/page_002.jpg`\n\n"
        "### [Panel 1 — 00:42]\n"
        "**Kira Nerys** [00:44]: Plain, simple Garak?\n",
        encoding="utf-8",
    )

    manifest = {
        "version": 1,
        "source_path": str(source_path),
        "media_type": "episode",
        "display_label": display_label,
        "series_name": series_name,
        "title": title,
        "synopsis": synopsis,
        "duration_sec": 2727.0,
        "total_pages": 2,
        "total_panels": 8,
        "cast": [
            {"cluster_id": f"SPEAKER_{i:02d}", "name": name, "line_count": 50 - i * 10}
            for i, name in enumerate(cast_names)
        ],
        "pages": [
            {
                "page_number": 1,
                "start_timestamp": "00:00",
                "end_timestamp": "00:39",
                "chapter_title": "Teaser",
                "image_path": "pages/page_001.jpg",
                "markdown_path": "pages/page_001.md",
            },
            {
                "page_number": 2,
                "start_timestamp": "00:39",
                "end_timestamp": "01:20",
                "chapter_title": "Act I",
                "image_path": "pages/page_002.jpg",
                "markdown_path": "pages/page_002.md",
            },
        ],
    }
    (pkg_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return jpg1


def test_video_sandboxing_filtering_and_manifest_search(tmp_path):
    media_root = tmp_path / "video"
    movies_dir = media_root / "movies"
    shows_dir = media_root / "shows" / "Star Trek- Deep Space Nine" / "Season 01"
    weird_dir = media_root / "weird"
    youtube_dir = media_root / "youtube"
    packages_root = tmp_path / "packages"
    locks_dir = tmp_path / "locks"

    for d in (movies_dir, shows_dir, weird_dir, youtube_dir, packages_root, locks_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Real videos vs sidecar junk files (.astlog, .srt, dotfiles)
    (movies_dir / "hbomax-Pulp Fiction.mp4").write_bytes(b"fakevideo")
    (movies_dir / "hbomax-Pulp Fiction.astlog").write_text("junk log", encoding="utf-8")
    (movies_dir / "hbomax-Pulp Fiction.srt").write_text("1\n00:00:01 --> 00:00:02\nhi\n", encoding="utf-8")
    (movies_dir / ".hidden_temp.mp4").write_bytes(b"hidden")

    ep2 = shows_dir / "prime-DS9_S01E02.mp4"
    ep3 = shows_dir / "prime-DS9_S01E03.mkv"
    ep2.write_bytes(b"fake_ep2")
    ep3.write_bytes(b"fake_ep3")

    # Sibling directories under /vault/video that MUST NOT be accessible
    (weird_dir / "secret.mp4").write_bytes(b"secret")
    (youtube_dir / "clip.mp4").write_bytes(b"clip")

    _write_sample_package(
        packages_root=packages_root,
        rel_no_ext="shows/Star Trek- Deep Space Nine/Season 01/prime-DS9_S01E02",
        source_path=ep2,
    )

    mgr = LobsterWatchManager(
        lobster_watch_bin=str(tmp_path / "missing-bin"),
        lobster_watch_config=None,
        packages_root=str(packages_root),
        locks_dir=str(locks_dir),
    )
    vw = VideoWatcherComponent(
        db_path=str(tmp_path / "video_state.db"),
        media_root=str(media_root),
        allowed_subdirs=["movies", "shows"],
        packages_root=str(packages_root),
        locks_dir=str(locks_dir),
        manager=mgr,
    )

    # 1. Root listing strictly exposes movies and shows
    root_out = vw.list_videos()
    assert "[DIR]  movies" in root_out
    assert "[DIR]  shows" in root_out
    assert "weird" not in root_out
    assert "youtube" not in root_out

    # 2. Sibling dirs and '..' traversal are blocked
    assert "Access denied" in vw.list_videos("weird")
    assert "Access denied" in vw.list_videos("youtube")
    assert "Path traversal" in vw.list_videos("movies/../weird")
    err_watch = json.loads(asyncio.run(vw.watch_video("weird/secret.mp4")))
    assert "Access denied" in err_watch["error"]

    # 3. Non-video files (.astlog, .srt, hidden) are filtered out
    movies_out = vw.list_videos("movies")
    assert "[UNPROCESSED] hbomax-Pulp Fiction.mp4" in movies_out
    assert ".astlog" not in movies_out
    assert ".srt" not in movies_out
    assert ".hidden_temp" not in movies_out

    # 4. Season listing shows [READY] with page/duration badge vs [UNPROCESSED]
    season_out = vw.list_videos("shows/Star Trek- Deep Space Nine/Season 01")
    assert "[READY — 2 pages, 45m] Star Trek: Deep Space Nine S01E02 - Past Prologue" in season_out
    assert "[UNPROCESSED] prime-DS9_S01E03.mkv" in season_out

    # 5. ready_only=True and search across cast & synopsis
    ready_out = vw.list_videos(ready_only=True)
    assert "Past Prologue" in ready_out
    assert "Pulp Fiction" not in ready_out
    assert "S01E03" not in ready_out

    cast_search = vw.list_videos(query="garak")
    assert "Past Prologue" in cast_search

    synopsis_search = vw.list_videos(query="bajoran underground")
    assert "Past Prologue" in synopsis_search


def test_watch_video_multimodal_attachment_and_independent_bookmarks(tmp_path):
    async def _run():
        media_root = tmp_path / "video"
        shows_dir = media_root / "shows" / "DS9"
        ebook_dir = tmp_path / "ebooks"
        packages_root = tmp_path / "packages"
        locks_dir = tmp_path / "locks"
        for d in (shows_dir, ebook_dir, packages_root, locks_dir):
            d.mkdir(parents=True, exist_ok=True)

        ep2 = shows_dir / "DS9_S01E02.mp4"
        ep2.write_bytes(b"ep2")
        orig_jpg1 = _write_sample_package(
            packages_root=packages_root,
            rel_no_ext="shows/DS9/DS9_S01E02",
            source_path=ep2,
        )

        # Also create a text file in FileReaderComponent to verify book bookmark independence
        (ebook_dir / "novel.txt").write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")

        video_db = str(tmp_path / "video_state.db")
        reader_db = str(tmp_path / "reader_state.db")
        mgr = LobsterWatchManager(
            lobster_watch_bin=str(tmp_path / "missing-bin"),
            lobster_watch_config=None,
            packages_root=str(packages_root),
            locks_dir=str(locks_dir),
        )

        mind_agent = AgentCore(thread_id="mind")
        fr_mind = FileReaderComponent(
            libraries={"ebook": str(ebook_dir)},
            db_path=reader_db,
            thread_id="mind",
        )
        vw_mind = VideoWatcherComponent(
            db_path=video_db,
            media_root=str(media_root),
            packages_root=str(packages_root),
            locks_dir=str(locks_dir),
            thread_id="mind",
            manager=mgr,
        )
        mind_agent.register_component(fr_mind)
        mind_agent.register_component(vw_mind)
        mind_agent._build_tools()
        assert {"list_videos", "watch_video", "watch_next_page"}.issubset(set(mind_agent.tool_map.keys()))

        vw_social = VideoWatcherComponent(
            db_path=video_db,
            media_root=str(media_root),
            packages_root=str(packages_root),
            locks_dir=str(locks_dir),
            thread_id="social",
            manager=mgr,
        )

        # Open a book first on mind thread
        fr_mind.read_file("ebook", "novel.txt", chunk=1)
        assert fr_mind.current_rel_path == "novel.txt"
        assert fr_mind.current_index == 1

        # Call watch_video via AgentCore using 'video' + 'chunk' aliases AND extensionless stem!
        res_raw = await mind_agent.execute_tool(
            "watch_video",
            json.dumps({"video": "shows/DS9/DS9_S01E02", "chunk": 1}),
        )
        res1 = json.loads(res_raw)
        assert res1["format"] == "video_storyboard"
        assert res1["filepath"] == "shows/DS9/DS9_S01E02.mp4"
        assert res1["page"] == 1
        assert res1["total_pages"] == 2
        assert "Bajoran underground" in res1["synopsis"]
        assert any(c["character"] == "Elim Garak" for c in res1["cast"])
        assert "It's Dr. Bashir, isn't it?" in res1["script"]
        assert res1["storyboard_image"]["attached_to_vision"] is True

        # Verify raw 1024x768 JPEG bytes were attached without lossy re-compression
        assert len(mind_agent.pending_tool_images) == 1
        attached_url = mind_agent.pending_tool_images.pop()
        assert attached_url.startswith("data:image/jpeg;base64,")
        decoded_bytes = base64.b64decode(attached_url.split(",", 1)[1])
        assert decoded_bytes == orig_jpg1

        # Advance to Page 2 via watch_next_page
        res2 = json.loads(await vw_mind.watch_next_page())
        assert res2["page"] == 2
        assert "Plain, simple Garak?" in res2["script"]
        assert "synopsis" not in res2  # Synopsis/cast only repeated on Page 1
        assert len(mind_agent.pending_tool_images) == 1

        # Book bookmark on mind AND video bookmark on social were never clobbered
        assert fr_mind.current_rel_path == "novel.txt"
        assert fr_mind.current_index == 1
        assert vw_social.current_rel_path is None

        # Reload VideoWatcherComponent and verify per-thread persistence
        vw_mind_reloaded = VideoWatcherComponent(
            db_path=video_db,
            media_root=str(media_root),
            packages_root=str(packages_root),
            locks_dir=str(locks_dir),
            thread_id="mind",
            manager=mgr,
        )
        assert vw_mind_reloaded.current_rel_path == "shows/DS9/DS9_S01E02.mp4"
        assert vw_mind_reloaded.current_page == 2
        assert "page 2/2" in vw_mind_reloaded.get_system_prompt_addition()

    asyncio.run(_run())


def test_on_demand_processing_dedup_lock_wait_and_inbox_notification(tmp_path):
    async def _run():
        media_root = tmp_path / "video"
        movies_dir = media_root / "movies"
        packages_root = tmp_path / "packages"
        locks_dir = tmp_path / "locks"
        for d in (movies_dir, packages_root, locks_dir):
            d.mkdir(parents=True, exist_ok=True)

        mov1 = movies_dir / "The Princess Bride.mp4"
        mov2 = movies_dir / "Monty Python.mp4"
        mov1.write_bytes(b"movie1")
        mov2.write_bytes(b"movie2")

        # Write a mock lobster-watch script that verifies serial execution (max concurrency == 1)
        # and writes a valid package atomically
        log_file = tmp_path / "invocations.log"
        active_marker = tmp_path / "active_job.marker"
        overlap_flag = tmp_path / "overlap_detected"
        mock_bin = tmp_path / "mock-lobster-watch"
        py_exe = os.path.abspath("./venv/bin/python")

        mock_bin.write_text(
            f"""#!/usr/bin/env bash
set -eu
"{py_exe}" - "$@" <<'PY'
import io, json, os, sys, time
from pathlib import Path
from PIL import Image

args = sys.argv[1:]
video_arg = args[-1]
active_marker = Path({str(active_marker)!r})
overlap_flag = Path({str(overlap_flag)!r})
log_file = Path({str(log_file)!r})
media_root = Path({str(media_root)!r}).resolve()
packages_root = Path({str(packages_root)!r}).resolve()

if active_marker.exists():
    overlap_flag.write_text("OVERLAP", encoding="utf-8")
active_marker.write_text(video_arg, encoding="utf-8")
try:
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(video_arg + "\\n")
    time.sleep(0.12)
    rel = Path(video_arg).resolve().relative_to(media_root).with_suffix("")
    pkg_dir = packages_root / rel
    (pkg_dir / "pages").mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (640, 480), (40, 90, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    (pkg_dir / "pages" / "page_001.jpg").write_bytes(buf.getvalue())
    (pkg_dir / "pages" / "page_001.md").write_text("# Storyboard Page 1/1\\n**Inigo** [00:10]: Hello!\\n", encoding="utf-8")

    manifest = {{
        "version": 1,
        "source_path": video_arg,
        "display_label": rel.name,
        "duration_sec": 5880.0,
        "total_pages": 1,
        "total_panels": 4,
        "cast": [{{"cluster_id": "SPEAKER_00", "name": "Inigo Montoya", "line_count": 42}}],
        "pages": [{{
            "page_number": 1,
            "start_timestamp": "00:00",
            "end_timestamp": "00:45",
            "image_path": "pages/page_001.jpg",
            "markdown_path": "pages/page_001.md",
        }}],
    }}
    tmp_m = pkg_dir / "manifest.json.tmp"
    tmp_m.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(tmp_m, pkg_dir / "manifest.json")
    print(json.dumps({{"status": "completed", "total_pages": 1, "display_label": rel.name}}))
finally:
    active_marker.unlink(missing_ok=True)
PY
""",
            encoding="utf-8",
        )
        mock_bin.chmod(mock_bin.stat().st_mode | stat.S_IEXEC)

        # Simulate an active manual CLI run holding an external job lock initially
        manual_lock_path = locks_dir / "manual_cli_job_abcdef123456.lock"
        manual_fd = os.open(str(manual_lock_path), os.O_CREAT | os.O_RDWR, 0o664)
        fcntl.flock(manual_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        mgr = LobsterWatchManager(
            lobster_watch_bin=str(mock_bin),
            lobster_watch_config=None,
            packages_root=str(packages_root),
            locks_dir=str(locks_dir),
            poll_interval_sec=0.03,
        )

        mind_agent = AgentCore(thread_id="mind")
        social_agent = AgentCore(thread_id="social")
        mind_agent.suspended.clear()
        social_agent.suspended.clear()

        vw_mind = VideoWatcherComponent(
            db_path=str(tmp_path / "video_state.db"),
            media_root=str(media_root),
            packages_root=str(packages_root),
            locks_dir=str(locks_dir),
            thread_id="mind",
            manager=mgr,
        )
        vw_social = VideoWatcherComponent(
            db_path=str(tmp_path / "video_state.db"),
            media_root=str(media_root),
            packages_root=str(packages_root),
            locks_dir=str(locks_dir),
            thread_id="social",
            manager=mgr,
        )
        mind_agent.register_component(vw_mind)
        social_agent.register_component(vw_social)

        try:
            # Both threads request the same unprocessed movie + mind also queues a second movie
            r1 = json.loads(await vw_mind.watch_video("movies/The Princess Bride.mp4", page=1))
            r1_dup = json.loads(await vw_social.watch_video("movies/The Princess Bride.mp4", page=1))
            r2 = json.loads(await vw_mind.watch_video("movies/Monty Python.mp4", page=1))

            assert r1["status"] == "processing_started"
            assert r1_dup["status"] == "already_processing"
            assert r2["status"] == "processing_started"
            assert "[PROCESSING] The Princess Bride.mp4" in vw_mind.list_videos("movies")

            # While manual CLI lock is held, mock-lobster-watch has NOT started yet!
            await asyncio.sleep(0.08)
            assert not log_file.exists()

            # Release manual CLI lock -> queued jobs now run sequentially
            fcntl.flock(manual_fd, fcntl.LOCK_UN)
            os.close(manual_fd)
            manual_fd = None

            deadline = asyncio.get_running_loop().time() + 5.0
            while asyncio.get_running_loop().time() < deadline:
                if mind_agent.inbox.qsize() >= 2 and social_agent.inbox.qsize() >= 1:
                    break
                await asyncio.sleep(0.05)

            assert not overlap_flag.exists(), "Two lobster-watch jobs overlapped!"
            invocations = log_file.read_text(encoding="utf-8").strip().splitlines()
            # Princess Bride was deduplicated (1 run) + Monty Python (1 run) == 2 total runs
            assert len(invocations) == 2

            assert mind_agent.suspended.is_set()
            assert social_agent.suspended.is_set()

            mind_msg_1 = await mind_agent.inbox.get()
            mind_msg_2 = await mind_agent.inbox.get()
            social_msg_1 = await social_agent.inbox.get()

            assert "[Video Ready] 'The Princess Bride'" in mind_msg_1
            assert "[Video Ready] 'Monty Python'" in mind_msg_2
            assert "[Video Ready] 'The Princess Bride'" in social_msg_1

            # Now watching Princess Bride immediately returns Page 1 + image!
            ready_play = json.loads(await vw_mind.watch_video("movies/The Princess Bride.mp4", page=1))
            assert ready_play["format"] == "video_storyboard"
            assert "Inigo" in ready_play["script"]
            assert len(mind_agent.pending_tool_images) == 1
        finally:
            if manual_fd is not None:
                try:
                    fcntl.flock(manual_fd, fcntl.LOCK_UN)
                    os.close(manual_fd)
                except OSError:
                    pass
            await mgr.shutdown()

    asyncio.run(_run())


def test_on_demand_processing_failure_notifies_subscriber(tmp_path):
    async def _run():
        media_root = tmp_path / "video"
        movies_dir = media_root / "movies"
        packages_root = tmp_path / "packages"
        locks_dir = tmp_path / "locks"
        for d in (movies_dir, packages_root, locks_dir):
            d.mkdir(parents=True, exist_ok=True)

        bad_mov = movies_dir / "Corrupt.mp4"
        bad_mov.write_bytes(b"corrupt")

        fail_bin = tmp_path / "fail-lobster-watch"
        fail_bin.write_text(
            "#!/usr/bin/env bash\nset -eu\necho 'ffprobe failed: invalid data' >&2\nexit 2\n",
            encoding="utf-8",
        )
        fail_bin.chmod(fail_bin.stat().st_mode | stat.S_IEXEC)

        mgr = LobsterWatchManager(
            lobster_watch_bin=str(fail_bin),
            lobster_watch_config=None,
            packages_root=str(packages_root),
            locks_dir=str(locks_dir),
            poll_interval_sec=0.02,
        )
        agent = AgentCore(thread_id="mind")
        agent.suspended.clear()
        vw = VideoWatcherComponent(
            db_path=str(tmp_path / "video_state.db"),
            media_root=str(media_root),
            packages_root=str(packages_root),
            locks_dir=str(locks_dir),
            thread_id="mind",
            manager=mgr,
        )
        agent.register_component(vw)

        try:
            r = json.loads(await vw.watch_video("movies/Corrupt.mp4", page=1))
            assert r["status"] == "processing_started"

            deadline = asyncio.get_running_loop().time() + 3.0
            while agent.inbox.empty() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.03)

            assert not agent.inbox.empty()
            fail_notice = await agent.inbox.get()
            assert "[Video Processing Failed]" in fail_notice
            assert "ffprobe failed: invalid data" in fail_notice
        finally:
            await mgr.shutdown()

    asyncio.run(_run())

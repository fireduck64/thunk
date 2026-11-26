import asyncio
import fcntl
import hashlib
import io
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.image import downsize_image_bytes, encode_jpeg_data_url
from src.utils.metrics import MetricSpan


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".m4v",
    ".avi",
    ".mov",
    ".webm",
    ".ts",
    ".mpg",
    ".mpeg",
}

MAX_DIR_LIST_ENTRIES = 100
AGENT_QUEUE_LOCK_NAME = "_agent_queue.lock"


def compute_lobster_job_id(abs_video_path: str) -> str:
    """Matches lobster_watch.utils.compute_job_id for per-video lockfile inspection."""
    norm = os.path.abspath(abs_video_path)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(norm))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:48].strip("_") or "video"
    return f"{safe_stem}_{digest}"


def is_lockfile_held(lock_path: str) -> bool:
    """Non-blocking probe checking whether an fcntl exclusive lock is currently held."""
    if not lock_path or not os.path.isfile(lock_path):
        return False
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _any_external_job_lock_held(locks_dir: str) -> bool:
    """Returns True if any lobster-watch per-video lockfile in locks_dir is currently held."""
    if not locks_dir or not os.path.isdir(locks_dir):
        return False
    try:
        names = os.listdir(locks_dir)
    except OSError:
        return False
    for name in names:
        if not name.endswith(".lock") or name == AGENT_QUEUE_LOCK_NAME:
            continue
        if is_lockfile_held(os.path.join(locks_dir, name)):
            return True
    return False


def _format_duration_short(duration_sec: float) -> str:
    total = max(0, int(round(duration_sec or 0.0)))
    if total <= 0:
        return ""
    hrs, rem = divmod(total, 3600)
    mins, _ = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m"
    return f"{max(1, mins)}m"


def _prepare_storyboard_data_url(abs_jpg_path: str, max_edge: int = 1024) -> Tuple[str, int, int]:
    """
    Loads a storyboard page image and returns (data_url, width, height).
    Preserves raw JPEG bytes directly without lossy re-compression when already
    RGB JPEG <= max_edge so panel corner timestamp badges remain crisp.
    """
    with open(abs_jpg_path, "rb") as f:
        raw_bytes = f.read()

    with Image.open(io.BytesIO(raw_bytes)) as img:
        width, height = img.size
        fmt = (img.format or "").upper()
        mode = img.mode

    if fmt == "JPEG" and mode == "RGB" and max(width, height) <= max_edge:
        return encode_jpeg_data_url(raw_bytes), width, height

    jpeg_bytes, width, height = downsize_image_bytes(raw_bytes, max_edge=max_edge)
    return encode_jpeg_data_url(jpeg_bytes), width, height


@dataclass
class _PendingVideoJob:
    abs_video: str
    rel_path: str
    manifest_path: str
    agent_name: str = "Agent"
    # key -> (agent_core, thread_id)
    subscribers: Dict[str, Tuple[Any, str]] = field(default_factory=dict)


class LobsterWatchManager:
    """
    Process-wide background queue for on-demand `lobster-watch process` jobs.
    - Deduplicates requests for the same video across threads (`social` & `mind`).
    - Serializes execution across multiple agent processes AND manual CLI runs using
      fcntl locks in `locks_dir` so at most one CUDA diarization/whisper job runs at a time.
    - Wakes/notifies subscribed threads via `agent.resume(...)` when `manifest.json` lands.
    """

    _instances: Dict[Tuple[str, str], "LobsterWatchManager"] = {}

    def __init__(
        self,
        lobster_watch_bin: str,
        lobster_watch_config: Optional[str],
        packages_root: str,
        locks_dir: str,
        poll_interval_sec: float = 1.0,
    ):
        self.lobster_watch_bin = os.path.abspath(lobster_watch_bin) if lobster_watch_bin else ""
        self.lobster_watch_config = (
            os.path.abspath(lobster_watch_config)
            if lobster_watch_config and os.path.exists(lobster_watch_config)
            else lobster_watch_config
        )
        self.packages_root = os.path.abspath(packages_root)
        self.locks_dir = os.path.abspath(locks_dir)
        self.poll_interval_sec = max(0.02, float(poll_interval_sec))

        self._jobs: Dict[str, _PendingVideoJob] = {}
        self._queue: List[str] = []
        self._current_video: Optional[str] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._active_proc: Optional[asyncio.subprocess.Process] = None
        self._shutting_down: bool = False

    @classmethod
    def get_shared(
        cls,
        lobster_watch_bin: str,
        lobster_watch_config: Optional[str],
        packages_root: str,
        locks_dir: str,
        poll_interval_sec: float = 1.0,
    ) -> "LobsterWatchManager":
        key = (os.path.abspath(lobster_watch_bin or ""), os.path.abspath(packages_root))
        inst = cls._instances.get(key)
        if inst is None:
            inst = cls(
                lobster_watch_bin=lobster_watch_bin,
                lobster_watch_config=lobster_watch_config,
                packages_root=packages_root,
                locks_dir=locks_dir,
                poll_interval_sec=poll_interval_sec,
            )
            cls._instances[key] = inst
        return inst

    @classmethod
    async def shutdown_all(cls) -> None:
        for inst in list(cls._instances.values()):
            await inst.shutdown()

    def is_queued_or_active(self, abs_video: str) -> bool:
        norm = os.path.abspath(abs_video)
        return norm in self._jobs or norm == self._current_video

    def enqueue(
        self,
        abs_video: str,
        rel_path: str,
        manifest_path: str,
        agent_core: Any = None,
        thread_id: str = "main",
    ) -> Dict[str, Any]:
        norm = os.path.abspath(abs_video)
        sub_key = f"{id(agent_core)}:{thread_id}" if agent_core is not None else f"none:{thread_id}"
        agent_name = getattr(agent_core, "name", None) or "Agent"

        already_tracked = norm in self._jobs
        job = self._jobs.get(norm)
        if job is None:
            job = _PendingVideoJob(
                abs_video=norm,
                rel_path=rel_path,
                manifest_path=manifest_path,
                agent_name=agent_name,
            )
            self._jobs[norm] = job
            self._queue.append(norm)

        if agent_core is not None:
            job.subscribers[sub_key] = (agent_core, thread_id)

        self._shutting_down = False
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

        if norm == self._current_video:
            queue_pos = 1
        elif norm in self._queue:
            offset = 1 if self._current_video is not None else 0
            queue_pos = offset + self._queue.index(norm) + 1
        else:
            queue_pos = 1

        return {
            "already_queued": already_tracked,
            "queue_position": queue_pos,
        }

    async def shutdown(self) -> None:
        self._shutting_down = True
        proc = self._active_proc
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        task = self._worker_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._worker_task = None
        self._current_video = None

    def _try_acquire_agent_queue_lock(self) -> Optional[int]:
        os.makedirs(self.locks_dir, exist_ok=True)
        lock_path = os.path.join(self.locks_dir, AGENT_QUEUE_LOCK_NAME)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            return fd
        except BlockingIOError:
            os.close(fd)
            return None
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _release_lock_fd(fd: Optional[int]) -> None:
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    async def _worker_loop(self) -> None:
        while self._queue and not self._shutting_down:
            norm = self._queue.pop(0)
            job = self._jobs.get(norm)
            if job is None:
                continue
            self._current_video = norm
            try:
                ok, err_msg = await self._process_single_job(job)
                if not self._shutting_down:
                    await self._notify_subscribers(job, ok=ok, error_msg=err_msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self._shutting_down:
                    await self._notify_subscribers(job, ok=False, error_msg=str(e))
            finally:
                self._jobs.pop(norm, None)
                self._current_video = None

    async def _process_single_job(self, job: _PendingVideoJob) -> Tuple[bool, str]:
        # 1. Fast path if already completed before our turn in the queue
        if os.path.isfile(job.manifest_path):
            return True, ""

        job_id = compute_lobster_job_id(job.abs_video)
        per_video_lock = os.path.join(self.locks_dir, f"{job_id}.lock")

        # 2. If another process (CLI or sibling agent) is already processing THIS exact video,
        # wait politely for that lock to clear and reuse its output!
        while is_lockfile_held(per_video_lock) and not self._shutting_down:
            await asyncio.sleep(self.poll_interval_sec)

        if self._shutting_down:
            return False, "Cancelled during shutdown."

        if os.path.isfile(job.manifest_path):
            return True, ""

        if not self.lobster_watch_bin or not os.path.isfile(self.lobster_watch_bin):
            return False, f"lobster-watch executable not found at '{self.lobster_watch_bin}'."

        # 3. Acquire host-wide agent queue lock AND wait for any manual CLI lobster-watch jobs
        # to finish so we never run concurrent GPU diarization/whisper processes.
        lock_fd: Optional[int] = None
        try:
            while not self._shutting_down:
                if lock_fd is None:
                    lock_fd = self._try_acquire_agent_queue_lock()
                if lock_fd is not None and not _any_external_job_lock_held(self.locks_dir):
                    break
                if lock_fd is not None:
                    # Release while waiting on an external manual CLI run so we don't starve locks
                    self._release_lock_fd(lock_fd)
                    lock_fd = None
                await asyncio.sleep(self.poll_interval_sec)

            if self._shutting_down:
                return False, "Cancelled during shutdown."

            # Re-check manifest after acquiring the global lock in case sibling agent just built it
            if os.path.isfile(job.manifest_path):
                return True, ""

            cmd = [self.lobster_watch_bin]
            if self.lobster_watch_config and os.path.isfile(self.lobster_watch_config):
                cmd.extend(["-c", self.lobster_watch_config])
            cmd.extend(["process", "--json", job.abs_video])

            with MetricSpan(
                operation="video_process",
                agent=job.agent_name,
                thread="background",
                component="video_watcher",
                type="lobster_watch",
            ) as span:
                span.inc("request_bytes", len(job.abs_video.encode("utf-8")))
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._active_proc = proc
                try:
                    stdout_b, stderr_b = await proc.communicate()
                finally:
                    self._active_proc = None

                stdout_s = (stdout_b or b"").decode("utf-8", errors="replace").strip()
                stderr_s = (stderr_b or b"").decode("utf-8", errors="replace").strip()
                span.inc("response_bytes", len(stdout_b or b"") + len(stderr_b or b""))

                if proc.returncode == 0 and os.path.isfile(job.manifest_path):
                    return True, ""

                span.fail()
                tail = stderr_s[-400:] if stderr_s else stdout_s[-400:]
                return False, f"exit code {proc.returncode}: {tail or 'manifest.json was not produced'}"
        finally:
            self._release_lock_fd(lock_fd)

    async def _notify_subscribers(self, job: _PendingVideoJob, ok: bool, error_msg: str = "") -> None:
        if not job.subscribers:
            return

        if ok and os.path.isfile(job.manifest_path):
            label = job.rel_path
            total_pages = 0
            try:
                with open(job.manifest_path, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                label = mdata.get("display_label") or label
                total_pages = int(mdata.get("total_pages") or len(mdata.get("pages") or []))
            except Exception:
                pass
            notice = (
                f"[Video Ready] '{label}' (filepath='{job.rel_path}') has finished processing "
                f"({total_pages} pages). Call watch_video(filepath='{job.rel_path}', page=1) "
                "whenever you are ready to start watching."
            )
        else:
            notice = (
                f"[Video Processing Failed] Could not prepare '{job.rel_path}' via lobster-watch "
                f"({error_msg or 'unknown error'})."
            )

        for agent_core, _tid in list(job.subscribers.values()):
            try:
                await agent_core.resume(notice)
            except Exception as e:
                print(f"[VideoWatcher Warning] Failed to notify subscriber thread: {e}")


class VideoWatcherComponent(BaseComponent):
    """
    Allows agents to browse, search, prepare on-demand, and watch TV episodes and movies
    from `/vault/video/movies` and `/vault/video/shows` rendered as multimodal graphic-novel
    packages (2x2 key-frame storyboards + speaker-identified scripts) via `lobster-watch`.
    """

    def __init__(
        self,
        db_path: str = None,
        media_root: str = None,
        allowed_subdirs: List[str] = None,
        packages_root: str = None,
        locks_dir: str = None,
        lobster_watch_bin: str = None,
        lobster_watch_config: str = None,
        thread_id: str = "main",
        manager: Optional[LobsterWatchManager] = None,
    ):
        self.config = load_config()
        self.thread_id = (thread_id or "main").strip() or "main"
        vw_cfg = self.config.get("video_watcher", {})

        if db_path is None:
            mem_db = self.config.get("memory", {}).get("db_path", "data/thunk/memory.db")
            default_db = os.path.join(os.path.dirname(os.path.abspath(mem_db)), "video_state.db")
            db_path = vw_cfg.get("db_path") or default_db
        self.db_path = os.path.abspath(db_path)

        raw_media_root = media_root or vw_cfg.get("media_root", "/vault/video")
        self.media_root = os.path.realpath(os.path.abspath(raw_media_root))

        raw_subdirs = (
            allowed_subdirs
            if allowed_subdirs is not None
            else vw_cfg.get("allowed_subdirs", ["movies", "shows"])
        )
        self.allowed_subdirs: List[str] = []
        self.subdir_roots: Dict[str, str] = {}
        for sd in raw_subdirs:
            clean_sd = str(sd).strip().strip("/\\")
            if not clean_sd or "/" in clean_sd or "\\" in clean_sd or clean_sd.startswith("."):
                continue
            self.allowed_subdirs.append(clean_sd)
            self.subdir_roots[clean_sd] = os.path.realpath(os.path.join(self.media_root, clean_sd))

        raw_pkg_root = packages_root or vw_cfg.get("packages_root", "/vault/video/lobster-watch/packages")
        self.packages_root = os.path.abspath(raw_pkg_root)

        raw_locks_dir = locks_dir or vw_cfg.get("locks_dir", "/var/tmp/lobster-watch/locks")
        self.locks_dir = os.path.abspath(raw_locks_dir)

        self.lobster_watch_bin = (
            lobster_watch_bin
            if lobster_watch_bin is not None
            else vw_cfg.get("lobster_watch_bin", "/home/fireduck/projects/lobster-watch/lobster-watch")
        )
        self.lobster_watch_config = (
            lobster_watch_config
            if lobster_watch_config is not None
            else vw_cfg.get("lobster_watch_config", "/home/fireduck/projects/lobster-watch/config.toml")
        )

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.manager = manager or LobsterWatchManager.get_shared(
            lobster_watch_bin=self.lobster_watch_bin,
            lobster_watch_config=self.lobster_watch_config,
            packages_root=self.packages_root,
            locks_dir=self.locks_dir,
        )

        self.current_rel_path: Optional[str] = None
        self.current_page: int = 0
        self.current_label: Optional[str] = None
        self._manifest_mem_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        self._init_db()
        self._load_state()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS video_thread_state (
                        thread_id TEXT PRIMARY KEY,
                        current_rel_path TEXT,
                        current_page INTEGER,
                        current_label TEXT,
                        updated_at REAL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS video_manifest_cache (
                        rel_no_ext TEXT PRIMARY KEY,
                        mtime REAL,
                        display_label TEXT,
                        series_name TEXT,
                        title TEXT,
                        synopsis TEXT,
                        cast_names TEXT,
                        total_pages INTEGER,
                        duration_sec REAL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO video_thread_state (
                        thread_id, current_rel_path, current_page, current_label, updated_at
                    )
                    VALUES (?, NULL, 0, NULL, 0.0)
                    """,
                    (self.thread_id,),
                )
        finally:
            conn.close()

    def _load_state(self) -> None:
        conn = self._connect_db()
        try:
            row = conn.execute(
                "SELECT current_rel_path, current_page, current_label "
                "FROM video_thread_state WHERE thread_id = ?",
                (self.thread_id,),
            ).fetchone()
        finally:
            conn.close()

        if not row or not row[0]:
            return

        rel_p = str(row[0])
        try:
            _, abs_vid, norm_rel = self._resolve_video_file(rel_p)
            if os.path.isfile(abs_vid):
                self.current_rel_path = norm_rel
                self.current_page = int(row[1] or 0)
                self.current_label = row[2] or norm_rel
        except Exception:
            self.current_rel_path = None
            self.current_page = 0
            self.current_label = None

    def _save_state(self) -> None:
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO video_thread_state (
                        thread_id, current_rel_path, current_page, current_label, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        current_rel_path = excluded.current_rel_path,
                        current_page = excluded.current_page,
                        current_label = excluded.current_label,
                        updated_at = excluded.updated_at
                    """,
                    (
                        self.thread_id,
                        self.current_rel_path,
                        self.current_page,
                        self.current_label,
                        time.time(),
                    ),
                )
        finally:
            conn.close()

    @staticmethod
    def _is_video_filename(filename: str) -> bool:
        if not filename or filename.startswith("."):
            return False
        ext = os.path.splitext(filename)[1].lower()
        return ext in VIDEO_EXTENSIONS

    def _normalize_virtual_subpath(self, raw_path: str) -> str:
        clean = (raw_path or "").strip().replace("\\", "/")
        # Strip accidental leading /vault/video/ prefix if passed
        media_prefix = self.media_root.replace("\\", "/").rstrip("/") + "/"
        if clean.startswith(media_prefix):
            clean = clean[len(media_prefix) :]
        clean = clean.strip("/")
        if clean in ("", "."):
            return ""
        return clean

    def _resolve_dir_or_root(self, raw_path: str) -> Tuple[Optional[str], str, str]:
        """
        Resolves a directory path under the allowed video subdirectories (`movies`, `shows`).
        Returns (subdir_key_or_None_for_virtual_root, real_abs_dir, clean_rel_path).
        """
        clean = self._normalize_virtual_subpath(raw_path)
        if not clean:
            return None, self.media_root, ""

        parts = [p for p in clean.split("/") if p and p != "."]
        if any(p == ".." for p in parts):
            raise ValueError(f"Path traversal ('..') is not allowed: '{raw_path}'.")

        top = parts[0]
        if top not in self.subdir_roots:
            allowed_str = ", ".join(self.allowed_subdirs)
            raise ValueError(
                f"Access denied: '{raw_path}'. Only top-level directories [{allowed_str}] are accessible."
            )

        base_real = self.subdir_roots[top]
        sub_rel = os.path.join(*parts[1:]) if len(parts) > 1 else ""
        target_real = os.path.realpath(os.path.join(base_real, sub_rel)) if sub_rel else base_real
        if os.path.commonpath([base_real, target_real]) != base_real:
            raise ValueError(f"Access denied: Path '{raw_path}' escapes '{top}'.")

        norm_rel = "/".join(parts)
        return top, target_real, norm_rel

    def _resolve_video_file(self, raw_filepath: str) -> Tuple[str, str, str]:
        """
        Resolves a video filepath (either with extension e.g. 'shows/MASH/Season 01/dvd-MASH_S01E01.mkv'
        or extensionless stem 'shows/MASH/Season 01/dvd-MASH_S01E01') strictly within `movies` or `shows`.
        Returns (subdir_key, real_abs_video_path, canonical_rel_path_with_ext).
        """
        clean = self._normalize_virtual_subpath(raw_filepath)
        if not clean:
            raise ValueError("A video filepath under 'movies/' or 'shows/' is required.")

        parts = [p for p in clean.split("/") if p and p != "."]
        if any(p == ".." for p in parts):
            raise ValueError(f"Path traversal ('..') is not allowed: '{raw_filepath}'.")

        top = parts[0]
        if top not in self.subdir_roots:
            allowed_str = ", ".join(self.allowed_subdirs)
            raise ValueError(
                f"Access denied: '{raw_filepath}'. Video filepath must start with one of: {allowed_str}."
            )
        if len(parts) < 2:
            raise ValueError(f"'{raw_filepath}' is a top-level directory, not a video file.")

        base_real = self.subdir_roots[top]
        candidate_real = os.path.realpath(os.path.join(base_real, *parts[1:]))
        if os.path.commonpath([base_real, candidate_real]) != base_real:
            raise ValueError(f"Access denied: Path '{raw_filepath}' escapes '{top}'.")

        # Exact video file match
        if os.path.isfile(candidate_real) and self._is_video_filename(os.path.basename(candidate_real)):
            rel_inside = os.path.relpath(candidate_real, base_real).replace(os.sep, "/")
            return top, candidate_real, f"{top}/{rel_inside}"

        # Extensionless stem fallback (e.g. 'movies/hbomax-Pulp Fiction')
        parent_dir = os.path.dirname(candidate_real)
        stem = os.path.basename(candidate_real)
        if os.path.isdir(parent_dir):
            for entry in sorted(os.listdir(parent_dir)):
                if not self._is_video_filename(entry):
                    continue
                e_stem, _ = os.path.splitext(entry)
                if e_stem == stem:
                    matched_real = os.path.realpath(os.path.join(parent_dir, entry))
                    if os.path.commonpath([base_real, matched_real]) == base_real and os.path.isfile(matched_real):
                        rel_inside = os.path.relpath(matched_real, base_real).replace(os.sep, "/")
                        return top, matched_real, f"{top}/{rel_inside}"

        raise FileNotFoundError(f"Video file '{raw_filepath}' not found.")

    def _package_dir_for_rel_path(self, rel_path: str) -> Tuple[str, str, str]:
        """Returns (rel_no_ext, package_dir, manifest_path) for a canonical rel_path."""
        rel_no_ext, _ = os.path.splitext(rel_path)
        pkg_dir = os.path.join(self.packages_root, *rel_no_ext.split("/"))
        manifest_path = os.path.join(pkg_dir, "manifest.json")
        return rel_no_ext, pkg_dir, manifest_path

    def _get_video_state(self, abs_video: str, rel_path: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Returns (state, summary_dict_or_None) where state is 'ready', 'processing', or 'unprocessed'.
        """
        rel_no_ext, _pkg_dir, manifest_path = self._package_dir_for_rel_path(rel_path)
        if os.path.isfile(manifest_path):
            summary = self._get_cached_manifest_summary(rel_no_ext, manifest_path)
            return "ready", summary

        job_id = compute_lobster_job_id(abs_video)
        lock_path = os.path.join(self.locks_dir, f"{job_id}.lock")
        if self.manager.is_queued_or_active(abs_video) or is_lockfile_held(lock_path):
            return "processing", None
        return "unprocessed", None

    def _get_cached_manifest_summary(self, rel_no_ext: str, manifest_path: str) -> Dict[str, Any]:
        try:
            mtime = os.path.getmtime(manifest_path)
        except OSError:
            return {}

        conn = self._connect_db()
        try:
            row = conn.execute(
                """
                SELECT mtime, display_label, series_name, title, synopsis, cast_names, total_pages, duration_sec
                FROM video_manifest_cache
                WHERE rel_no_ext = ?
                """,
                (rel_no_ext,),
            ).fetchone()
            if row and row[0] == mtime:
                return {
                    "display_label": row[1] or rel_no_ext,
                    "series_name": row[2] or "",
                    "title": row[3] or "",
                    "synopsis": row[4] or "",
                    "cast_names": row[5] or "",
                    "total_pages": int(row[6] or 0),
                    "duration_sec": float(row[7] or 0.0),
                }

            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            display_label = str(data.get("display_label") or os.path.basename(rel_no_ext))
            series_name = str(data.get("series_name") or "")
            title = str(data.get("title") or "")
            synopsis = str(data.get("synopsis") or "")
            cast_list = [
                str(c.get("name") or "").strip()
                for c in (data.get("cast") or [])
                if c.get("name") and not str(c.get("name")).startswith("SPEAKER_")
            ]
            cast_names = ", ".join(cast_list)
            total_pages = int(data.get("total_pages") or len(data.get("pages") or []))
            duration_sec = float(data.get("duration_sec") or 0.0)

            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO video_manifest_cache (
                        rel_no_ext, mtime, display_label, series_name, title,
                        synopsis, cast_names, total_pages, duration_sec
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rel_no_ext,
                        mtime,
                        display_label,
                        series_name,
                        title,
                        synopsis,
                        cast_names,
                        total_pages,
                        duration_sec,
                    ),
                )
            return {
                "display_label": display_label,
                "series_name": series_name,
                "title": title,
                "synopsis": synopsis,
                "cast_names": cast_names,
                "total_pages": total_pages,
                "duration_sec": duration_sec,
            }
        except Exception:
            return {"display_label": os.path.basename(rel_no_ext), "total_pages": 0, "duration_sec": 0.0}
        finally:
            conn.close()

    def _load_full_manifest(self, manifest_path: str) -> Dict[str, Any]:
        mtime = os.path.getmtime(manifest_path)
        cached = self._manifest_mem_cache.get(manifest_path)
        if cached and cached[0] == mtime:
            return cached[1]
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Keep memory cache bounded to recent open manifests
        if len(self._manifest_mem_cache) >= 8:
            self._manifest_mem_cache.clear()
        self._manifest_mem_cache[manifest_path] = (mtime, data)
        return data

    def _format_video_entry_line(
        self,
        rel_path: str,
        state: str,
        summary: Optional[Dict[str, Any]],
    ) -> str:
        if state == "ready" and summary:
            label = summary.get("display_label") or os.path.basename(rel_path)
            pages = summary.get("total_pages") or 0
            dur = _format_duration_short(summary.get("duration_sec") or 0.0)
            meta = f"{pages} pages, {dur}" if dur else f"{pages} pages"
            return f"[READY — {meta}] {label} [path: {rel_path}]"
        if state == "processing":
            return f"[PROCESSING] {os.path.basename(rel_path)} [path: {rel_path}]"
        return f"[UNPROCESSED] {os.path.basename(rel_path)} [path: {rel_path}]"

    def get_system_prompt_addition(self) -> str:
        prompt = (
            "--- VIDEO STORYBOARD PLAYER (Movies & TV Shows) ---\n"
            "You can watch movies (`movies/`) and TV episodes (`shows/`) converted into multimodal graphic-novel "
            "packages (4-panel key-frame storyboards paired with speaker-identified dialogue scripts):\n"
            "- Use `list_videos(path='', query='', ready_only=False)` to browse `movies/` and `shows/`, see what is already "
            "`[READY]` vs `[UNPROCESSED]`, or search by title, series, synopsis, or character name.\n"
            "- Use `watch_video(filepath, page=1)` to watch a page: if `[READY]`, it attaches the 2x2 storyboard image to your "
            "visual context AND returns the synchronized dialogue script; if `[UNPROCESSED]`, it automatically starts background "
            "processing via `lobster-watch` and notifies your inbox as soon as it is ready.\n"
            "- Use `watch_next_page()` to advance to the next storyboard page (~40 seconds of story) of your active video."
        )
        if self.current_rel_path:
            label = self.current_label or self.current_rel_path
            _, _, manifest_path = self._package_dir_for_rel_path(self.current_rel_path)
            if os.path.isfile(manifest_path):
                rel_no_ext, _ = os.path.splitext(self.current_rel_path)
                summary = self._get_cached_manifest_summary(rel_no_ext, manifest_path)
                total = summary.get("total_pages") or 0
                if total > 0:
                    prompt += (
                        f"\n[Video Status: Active video is '{label}' "
                        f"('{self.current_rel_path}') at page {self.current_page}/{total}.]"
                    )
        return prompt

    def get_tools(self) -> List[Callable]:
        return [
            self.list_videos,
            self.watch_video,
            self.watch_next_page,
        ]

    def list_videos(self, path: str = "", query: str = "", ready_only: bool = False) -> str:
        """
        Browses or searches movies (`movies/`) and TV shows (`shows/`).
        Pass `path` (e.g. 'movies', 'shows', 'shows/Star Trek- Deep Space Nine/Season 01') to list a directory,
        `query` to search by title, show, character name, or synopsis, or `ready_only=True` to list ready-to-watch packages.
        """
        try:
            clean_query = (query or "").strip()
            top, target_dir, norm_rel = self._resolve_dir_or_root(path)

            # Search / ready_only walk mode across target scope
            if clean_query or ready_only:
                terms = [t for t in re.split(r"[\s,]+", clean_query.lower()) if t]
                roots_to_walk: List[Tuple[str, str]] = []
                if top is None:
                    for sd in self.allowed_subdirs:
                        sd_real = self.subdir_roots[sd]
                        if os.path.isdir(sd_real):
                            roots_to_walk.append((sd, sd_real))
                else:
                    if os.path.isdir(target_dir):
                        roots_to_walk.append((top, target_dir))
                    else:
                        return f"Error: Directory '{path}' does not exist."

                ready_hits: List[str] = []
                other_hits: List[str] = []
                max_results = 60

                for sd, walk_start in roots_to_walk:
                    sd_base = self.subdir_roots[sd]
                    for cur_root, dirs, files in os.walk(walk_start):
                        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                        for fname in sorted(files):
                            if not self._is_video_filename(fname):
                                continue
                            abs_f = os.path.realpath(os.path.join(cur_root, fname))
                            if os.path.commonpath([sd_base, abs_f]) != sd_base:
                                continue
                            rel_inside = os.path.relpath(abs_f, sd_base).replace(os.sep, "/")
                            rel_path = f"{sd}/{rel_inside}"
                            state, summary = self._get_video_state(abs_f, rel_path)
                            if ready_only and state != "ready":
                                continue

                            if terms:
                                hay_parts = [rel_path.lower()]
                                if summary:
                                    hay_parts.extend(
                                        [
                                            (summary.get("display_label") or "").lower(),
                                            (summary.get("series_name") or "").lower(),
                                            (summary.get("title") or "").lower(),
                                            (summary.get("synopsis") or "").lower(),
                                            (summary.get("cast_names") or "").lower(),
                                        ]
                                    )
                                haystack = " ".join(hay_parts)
                                if not all(t in haystack for t in terms):
                                    continue

                            line = self._format_video_entry_line(rel_path, state, summary)
                            if state == "ready":
                                ready_hits.append(line)
                            else:
                                other_hits.append(line)

                            if len(ready_hits) + len(other_hits) >= max_results * 3:
                                break

                combined = ready_hits + other_hits
                if not combined:
                    scope_desc = norm_rel or "movies, shows"
                    if clean_query:
                        return f"No videos matching '{clean_query}' found in {scope_desc}."
                    return f"No processed [READY] videos found in {scope_desc}."

                shown = combined[:max_results]
                hdr = (
                    f"Video search results for '{clean_query}':"
                    if clean_query
                    else f"Ready-to-watch videos in '{norm_rel or 'movies & shows'}':"
                )
                lines = [hdr] + [f"- {item}" for item in shown]
                if len(combined) > max_results:
                    lines.append(f"... and {len(combined) - max_results} more matches (refine 'query' or 'path').")
                return "\n".join(lines)

            # Virtual root listing: strictly movies/ and shows/
            if top is None:
                lines = ["Available video libraries:\n"]
                for sd in self.allowed_subdirs:
                    sd_real = self.subdir_roots[sd]
                    if os.path.isdir(sd_real):
                        lines.append(f"[DIR]  {sd}")
                if len(lines) == 1:
                    return "No configured video directories found on disk."
                lines.append(
                    "\nTip: Call list_videos(ready_only=True) to see episodes/movies already processed and ready immediately."
                )
                return "\n".join(lines)

            if not os.path.exists(target_dir):
                return f"Error: Directory '{path}' does not exist."
            if not os.path.isdir(target_dir):
                return f"Error: '{path}' is not a directory."

            sd_base = self.subdir_roots[top]
            entries = os.listdir(target_dir)
            dirs = sorted(
                d
                for d in entries
                if not d.startswith(".") and os.path.isdir(os.path.join(target_dir, d))
            )
            files = sorted(
                f
                for f in entries
                if self._is_video_filename(f) and os.path.isfile(os.path.join(target_dir, f))
            )

            if not dirs and not files:
                return f"Directory '{norm_rel}' contains no subdirectories or video files."

            out = [f"Contents of video : {norm_rel}\n"]
            shown = 0
            total_entries = len(dirs) + len(files)

            for d in dirs:
                if shown >= MAX_DIR_LIST_ENTRIES:
                    break
                out.append(f"[DIR]  {d}")
                shown += 1

            for fname in files:
                if shown >= MAX_DIR_LIST_ENTRIES:
                    break
                abs_f = os.path.realpath(os.path.join(target_dir, fname))
                rel_inside = os.path.relpath(abs_f, sd_base).replace(os.sep, "/")
                rel_path = f"{top}/{rel_inside}"
                state, summary = self._get_video_state(abs_f, rel_path)
                out.append(self._format_video_entry_line(rel_path, state, summary))
                shown += 1

            if total_entries > shown:
                out.append(f"... and {total_entries - shown} more entries (use 'query' to filter).")

            return "\n".join(out)
        except Exception as e:
            return f"Error listing videos: {str(e)}"

    async def watch_video(self, filepath: str, page: int = 1) -> str:
        """
        Watches a 1-indexed storyboard page of a movie or TV episode under `movies/` or `shows/`.
        If the video is `[READY]`, attaches the 4-panel storyboard image to your visual context and returns the
        speaker-identified script, updating your active video bookmark.
        If `[UNPROCESSED]` or `[PROCESSING]`, queues background processing via `lobster-watch` and notifies you when ready.
        """
        try:
            _top, abs_video, rel_path = self._resolve_video_file(filepath)
            rel_no_ext, pkg_dir, manifest_path = self._package_dir_for_rel_path(rel_path)

            if not os.path.isfile(manifest_path):
                agent_core = getattr(self, "agent", None)
                q_info = self.manager.enqueue(
                    abs_video=abs_video,
                    rel_path=rel_path,
                    manifest_path=manifest_path,
                    agent_core=agent_core,
                    thread_id=self.thread_id,
                )
                return json.dumps(
                    {
                        "status": "already_processing" if q_info["already_queued"] else "processing_started",
                        "filepath": rel_path,
                        "queue_position": q_info["queue_position"],
                        "message": (
                            f"'{os.path.basename(rel_path)}' is being converted into storyboards and speaker-attributed "
                            "dialogue in the background by lobster-watch (typically 3–12 minutes depending on runtime). "
                            "Do NOT poll—you will automatically receive a [Video Ready] inbox notification when it finishes."
                        ),
                    },
                    indent=2,
                )

            manifest = self._load_full_manifest(manifest_path)
            # Warm summary cache in SQLite if needed
            self._get_cached_manifest_summary(rel_no_ext, manifest_path)

            pages = manifest.get("pages") or []
            total_pages = int(manifest.get("total_pages") or len(pages))
            label = str(manifest.get("display_label") or os.path.basename(rel_no_ext))
            page_num = max(1, int(page if page is not None else 1))

            if total_pages <= 0:
                return json.dumps(
                    {
                        "format": "video_storyboard",
                        "filepath": rel_path,
                        "document": label,
                        "error": f"Package '{label}' contains 0 storyboard pages.",
                    },
                    indent=2,
                )

            if page_num > total_pages:
                return json.dumps(
                    {
                        "format": "video_storyboard",
                        "filepath": rel_path,
                        "document": label,
                        "page": page_num,
                        "total_pages": total_pages,
                        "error": f"You have reached the end of '{label}' (requested page {page_num}, total_pages={total_pages}).",
                    },
                    indent=2,
                )

            page_entry = pages[page_num - 1]
            md_rel = str(page_entry.get("markdown_path") or f"pages/page_{page_num:03d}.md")
            img_rel = str(page_entry.get("image_path") or f"pages/page_{page_num:03d}.jpg")
            abs_md = os.path.realpath(os.path.join(pkg_dir, md_rel))
            abs_img = os.path.realpath(os.path.join(pkg_dir, img_rel))
            real_pkg_dir = os.path.realpath(pkg_dir)

            if os.path.commonpath([real_pkg_dir, abs_md]) != real_pkg_dir:
                raise ValueError("Invalid markdown_path inside manifest.")
            if os.path.commonpath([real_pkg_dir, abs_img]) != real_pkg_dir:
                raise ValueError("Invalid image_path inside manifest.")

            if os.path.isfile(abs_md):
                with open(abs_md, "r", encoding="utf-8", errors="replace") as f:
                    script_text = f.read().strip()
            else:
                script_text = ""

            img_attached = False
            img_dims: Optional[Tuple[int, int]] = None
            if os.path.isfile(abs_img):
                data_url, w, h = _prepare_storyboard_data_url(abs_img, max_edge=1024)
                img_dims = (w, h)
                if hasattr(self, "agent") and self.agent is not None:
                    self.agent.pending_tool_images.append(data_url)
                    img_attached = True

            self.current_rel_path = rel_path
            self.current_page = page_num
            self.current_label = label
            self._save_state()

            if page_num < total_pages:
                next_hint = (
                    f"Call watch_next_page() to watch page {page_num + 1}/{total_pages}, "
                    f"or call watch_video(filepath='{rel_path}', page={page_num + 1})."
                )
            else:
                next_hint = f"End of '{label}' reached ({total_pages}/{total_pages} pages)."

            payload: Dict[str, Any] = {
                "format": "video_storyboard",
                "filepath": rel_path,
                "document": label,
                "page": page_num,
                "total_pages": total_pages,
                "time_range": f"{page_entry.get('start_timestamp', '')} – {page_entry.get('end_timestamp', '')}",
            }
            if page_entry.get("chapter_title"):
                payload["chapter_title"] = page_entry["chapter_title"]

            if page_num == 1:
                if manifest.get("synopsis"):
                    payload["synopsis"] = manifest["synopsis"]
                dur_str = _format_duration_short(float(manifest.get("duration_sec") or 0.0))
                if dur_str:
                    payload["duration"] = dur_str
                cast_summary = [
                    {"character": c.get("name"), "lines": c.get("line_count", 0)}
                    for c in (manifest.get("cast") or [])[:12]
                    if c.get("name") and not str(c.get("name")).startswith("SPEAKER_")
                ]
                if cast_summary:
                    payload["cast"] = cast_summary

            if img_dims is not None:
                payload["storyboard_image"] = {
                    "attached_to_vision": img_attached,
                    "width": img_dims[0],
                    "height": img_dims[1],
                    "layout": "2x2 key-frame grid matching [Panel 1..4] badges",
                }
            payload["script"] = script_text
            payload["next_hint"] = next_hint
            return json.dumps(payload, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Error watching video '{filepath}': {str(e)}"})

    async def watch_next_page(self) -> str:
        """
        Advances and watches the next storyboard page (image + speaker-attributed script)
        of your active movie or TV episode on this thread.
        """
        if not self.current_rel_path:
            return json.dumps(
                {
                    "error": "No active video is open on this thread. Call watch_video(filepath, page=1) first."
                }
            )
        return await self.watch_video(self.current_rel_path, page=self.current_page + 1)

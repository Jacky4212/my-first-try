"""FastAPI server for guitar audio transcription."""
import json
import os
import uuid
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from transcribe import transcribe

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Guitar Transcriber")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task store (access guarded by Lock for thread safety)
tasks: dict = {}
_tasks_lock = threading.Lock()
# Cancel signal flags per task
_cancel_events: dict[str, threading.Event] = {}

# Allowed audio formats
ALLOWED_EXTS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'}
# Max file size: 50 MB
MAX_FILE_SIZE = 50 * 1024 * 1024


def _read_cache(task_id: str) -> dict | None:
    """Read cached transcription result from disk."""
    cache_path = OUTPUT_DIR / f"{task_id}.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(task_id: str, data: dict) -> None:
    """Persist transcription result to disk cache."""
    cache_path = OUTPUT_DIR / f"{task_id}.json"
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass  # cache write failure is non-fatal


def _update_progress(task_id: str, progress: int, message: str = ''):
    """Thread-safe progress update."""
    with _tasks_lock:
        if task_id in tasks:
            tasks[task_id]['progress'] = progress
            if message:
                tasks[task_id]['message'] = message


def run_transcribe(task_id: str, audio_path: str):
    """Run transcription in background thread with cancel support."""
    cancel_event = _cancel_events.get(task_id)
    if not cancel_event:
        cancel_event = threading.Event()
        _cancel_events[task_id] = cancel_event

    _update_progress(task_id, 5, '准备中...')
    try:
        # Check cancel before starting
        if cancel_event.is_set():
            _update_progress(task_id, 0, '已取消')
            with _tasks_lock:
                tasks[task_id]['status'] = 'cancelled'
            return

        _update_progress(task_id, 10, '正在分析音频特征...')

        # transcribe() now accepts a progress callback and cancel_event
        result = transcribe(
            audio_path,
            str(OUTPUT_DIR),
            task_id,
            remove_vocals=True,
            progress_callback=lambda p, m: _update_progress(task_id, p, m),
            cancel_event=cancel_event,
        )

        if cancel_event.is_set():
            _update_progress(task_id, 0, '已取消')
            with _tasks_lock:
                tasks[task_id]['status'] = 'cancelled'
            return

        with _tasks_lock:
            tasks[task_id].update(result)
            tasks[task_id]['progress'] = 100
            tasks[task_id]['message'] = '完成'
        if result.get('status') == 'done':
            _write_cache(task_id, result)
    except Exception as e:
        if not cancel_event.is_set():
            with _tasks_lock:
                tasks[task_id] = {
                    'status': 'error', 'progress': 0, 'message': str(e),
                }
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass
        finally:
            _cancel_events.pop(task_id, None)


@app.post("/api/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    """Upload audio file for transcription."""
    if not audio or not audio.filename:
        raise HTTPException(400, "No file provided")

    ext = Path(audio.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format: {ext} (allowed: {', '.join(sorted(ALLOWED_EXTS))})")

    task_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{task_id}{ext}"

    # File size check (content-length header or stream size)
    content_length = audio.size
    if content_length and content_length > MAX_FILE_SIZE:
        raise HTTPException(413, f"文件过大 (最大 {MAX_FILE_SIZE // 1024 // 1024}MB)")

    try:
        with open(save_path, "wb") as f:
            written = 0
            while True:
                chunk = audio.file.read(8192)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_FILE_SIZE:
                    audio.file.close()
                    os.remove(save_path)
                    raise HTTPException(413, f"文件过大 (最大 {MAX_FILE_SIZE // 1024 // 1024}MB)")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to save uploaded file: {e}")
    finally:
        if audio.file:
            audio.file.close()

    cancel_event = threading.Event()
    _cancel_events[task_id] = cancel_event

    with _tasks_lock:
        tasks[task_id] = {'task_id': task_id, 'status': 'pending', 'progress': 0, 'message': ''}

    thread = threading.Thread(
        target=run_transcribe,
        args=(task_id, str(save_path)),
        daemon=True,
    )
    thread.start()

    return {'task_id': task_id, 'status': 'pending'}


@app.get("/api/status/{task_id}")
async def api_status(task_id: str):
    """Get transcription task status."""
    with _tasks_lock:
        task = tasks.get(task_id)
    if task:
        return task
    # Check disk cache
    cached = _read_cache(task_id)
    if cached:
        return cached
    raise HTTPException(404, f"Task '{task_id}' not found")


@app.get("/api/transcriptions")
async def api_list_transcriptions():
    """List recent cached transcriptions."""
    results = []
    for cache_file in sorted(OUTPUT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            mtime = cache_file.stat().st_mtime
            results.append({
                'task_id': cache_file.stem,
                'note_count': data.get('note_count', 0),
                'duration': data.get('duration', 0),
                'tempo': data.get('tempo', 120),
                'num_bars': data.get('num_bars', 0),
                'timestamp': mtime,  # Unix timestamp, for relative time display
            })
        except Exception:
            pass
    return results[:20]


# Serve output files
try:
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
except Exception:
    pass  # static dir may not exist at import time


@app.post("/api/cancel/{task_id}")
async def api_cancel(task_id: str):
    """Cancel a running transcription task."""
    cancel_event = _cancel_events.get(task_id)
    if not cancel_event:
        with _tasks_lock:
            task = tasks.get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        if task.get('status') in ('done', 'error', 'cancelled'):
            return {"status": task['status'], "message": "任务已结束"}
        return {"status": "ok", "message": "取消信号已发送"}

    cancel_event.set()
    with _tasks_lock:
        if task_id in tasks:
            tasks[task_id]['status'] = 'cancelled'
            tasks[task_id]['message'] = '用户取消'
    return {"status": "cancelled"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info",
        reload=False,
    )

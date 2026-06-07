"""FastAPI server for guitar audio transcription."""
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

# In-memory task store
tasks: dict = {}


def run_transcribe(task_id: str, audio_path: str):
    """Run transcription in background thread."""
    tasks[task_id]['status'] = 'processing'
    tasks[task_id]['progress'] = 10
    try:
        result = transcribe(audio_path, str(OUTPUT_DIR), task_id, remove_vocals=True)
        tasks[task_id].update(result)
        tasks[task_id]['progress'] = 100
        # Cache result to disk so user can reload without re-processing
        if result.get('status') == 'done':
            import json
            _cache_path = OUTPUT_DIR / f"{task_id}.json"
            with open(_cache_path, 'w', encoding='utf-8') as _f:
                json.dump(result, _f, ensure_ascii=False)
    except Exception as e:
        tasks[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': str(e),
        }
    finally:
        # Clean up uploaded file
        try:
            os.remove(audio_path)
        except OSError:
            pass


@app.post("/api/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    """Upload audio file for transcription."""
    if not audio.filename:
        raise HTTPException(400, "No file provided")

    # Validate extension
    ext = Path(audio.filename).suffix.lower()
    if ext not in ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'):
        raise HTTPException(400, f"Unsupported format: {ext}")

    task_id = uuid.uuid4().hex[:12]
    safe_name = f"{task_id}{ext}"
    save_path = UPLOAD_DIR / safe_name

    with open(save_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    tasks[task_id] = {
        'task_id': task_id,
        'status': 'pending',
        'progress': 0,
    }

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
    task = tasks.get(task_id)
    if not task:
        # Check disk cache
        import json
        _cache_path = OUTPUT_DIR / f"{task_id}.json"
        if _cache_path.exists():
            with open(_cache_path, 'r', encoding='utf-8') as _f:
                return json.load(_f)
        raise HTTPException(404, "Task not found")
    return task


@app.get("/api/transcriptions")
async def api_list_transcriptions():
    """List recent cached transcriptions."""
    import json
    results = []
    for _p in sorted(OUTPUT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(_p, 'r', encoding='utf-8') as _f:
                _d = json.load(_f)
            results.append({
                'task_id': _p.stem,
                'note_count': _d.get('note_count', 0),
                'duration': _d.get('duration', 0),
                'tempo': _d.get('tempo', 120),
                'num_bars': _d.get('num_bars', 0),
            })
        except Exception:
            pass
    return results[:20]


# Serve output files
if os.path.isdir(str(OUTPUT_DIR)):
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")

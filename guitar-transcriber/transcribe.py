"""Core transcription pipeline: audio -> MIDI -> guitar tab data"""
import os
import shutil
import subprocess
import traceback

import numpy as np
import pretty_midi

_script_dir = os.path.dirname(os.path.abspath(__file__))

def _safe_remove(path: str) -> None:
    """Remove a file without raising on error."""
    try:
        os.remove(path)
    except OSError:
        pass

# Ensure ffmpeg in the script directory is on PATH (needed by audio-separator)
if _script_dir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _script_dir + os.pathsep + os.environ.get('PATH', '')

# Standard guitar tuning MIDI pitches (high e, B, G, D, A, low E)
STD_TUNING = [64, 59, 55, 50, 45, 40]
MAX_FRET = 24

# Vocal separation model name
UVR_MODEL_NAME = 'UVR_MDXNET_KARA_2.onnx'

# Look for ffmpeg in the same directory, PATH, or imageio_ffmpeg
_FFMPEG = None
for _cand in [os.path.join(_script_dir, 'ffmpeg.exe'), 'ffmpeg.exe', 'ffmpeg']:
    try:
        subprocess.run([_cand, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        _FFMPEG = _cand
        break
    except Exception:
        pass

if not _FFMPEG:
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass


def _ensure_wav(audio_path):
    """Convert M4A/MP4/AAC to WAV if needed. Returns path to WAV file."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ('.wav', '.wave'):
        return audio_path
    if not _FFMPEG:
        raise RuntimeError(f'无法处理 {ext} 格式，需要安装 ffmpeg。请将 ffmpeg.exe 放入本目录。')
    wav_path = audio_path + '.conv.wav'
    subprocess.run(
        [_FFMPEG, '-y', '-i', audio_path, '-ar', '22050', '-ac', '1', wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return wav_path


def _py(v):
    """Convert numpy scalar to Python native type."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def midi_to_fretboard(notes, tuning=None, max_fret=MAX_FRET):
    """Map MIDI notes to (string, fret) pairs.

    notes: list of (pitch, start, end, velocity)
    Returns list of (pitch, start, end, velocity, string, fret)
      string is 1..6 (1=high e, 6=low E)
    """
    if tuning is None:
        tuning = STD_TUNING

    result = []
    for pitch, start, end, velocity in notes:
        best_string = 1
        best_fret = 0
        best_cost = float('inf')

        for i, open_pitch in enumerate(tuning):
            fret = pitch - open_pitch
            if 0 <= fret <= max_fret:
                cost = fret * 2 + abs(i - 2.5) * 0.5
                if cost < best_cost:
                    best_cost = cost
                    best_string = i + 1
                    best_fret = fret

        if best_cost == float('inf'):
            best_string = 6
            best_fret = max(0, pitch - tuning[5])

        result.append((pitch, start, end, velocity, best_string, best_fret))

    return result


def notes_to_bars(notes_fb, beats_per_bar=4, beats_per_minute=120):
    """Organize fretboard notes into bars/beats for tab rendering.

    Returns list of bars, each bar has beats, each beat has notes.
    """
    if not notes_fb:
        return []

    beats_per_second = beats_per_minute / 60.0
    max_end = max(n[2] for n in notes_fb)
    total_beats = max(1, int(np.ceil(max_end * beats_per_second)))
    num_bars = max(1, int(np.ceil(total_beats / beats_per_bar)))

    bars = []
    for bar_idx in range(num_bars):
        bar_start_beat = bar_idx * beats_per_bar  # type: ignore
        bar_end_beat = bar_start_beat + beats_per_bar

        # Collect notes that fall in this bar
        bar_notes = []
        for pitch, start, end, vel, string, fret in notes_fb:
            beat_pos = start * beats_per_second
            if bar_start_beat <= beat_pos < bar_end_beat:
                dur_beats = max(0.25, (end - start) * beats_per_second)
                bar_notes.append({
                    'beat': round(beat_pos - bar_start_beat, 2),
                    'duration': round(dur_beats, 2),
                    'string': int(string),
                    'fret': int(fret),
                    'pitch': int(pitch),
                    'velocity': int(vel),
                })

        bars.append({'index': bar_idx, 'notes': bar_notes})

    return bars


def separate_vocals(audio_path, output_dir=None, cancel_event=None):
    """Remove vocals from audio, return path to instrumental track.

    Uses MDX-Net Karaoke model for vocal/instrumental separation.
    Returns path to the instrumental (no-vocals) WAV file.
    """
    if cancel_event and cancel_event.is_set():
        return audio_path
    if output_dir is None:
        output_dir = os.path.dirname(audio_path)
    from audio_separator.separator import Separator
    # Use models dir on D drive, not C drive temp
    models_dir = os.path.join(_script_dir, '.models')
    os.makedirs(models_dir, exist_ok=True)
    sep = Separator(
        output_dir=output_dir,
        output_format='WAV',
        log_level=1,
        model_file_dir=models_dir,
    )
    sep.load_model(UVR_MODEL_NAME)
    sep.separate(audio_path)
    if cancel_event and cancel_event.is_set():
        return audio_path
    # separate() writes files to output_dir; find the instrumental one
    base = os.path.splitext(os.path.basename(audio_path))[0]
    for f in os.listdir(output_dir):
        if 'instrumental' in f.lower() and base[:8] in f:
            return os.path.join(output_dir, f)
    # Fallback: return original audio
    return audio_path


def transcribe(audio_path, output_dir, task_id, remove_vocals=True,
               progress_callback=None, cancel_event=None):
    """Full transcription pipeline. Saves MIDI and returns tab data + metadata."""
    wav_path = None
    inst_path = None
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        _report = (lambda p, m: None) if not progress_callback else progress_callback

        _report(12, '转换音频格式...')
        if cancel_event and cancel_event.is_set():
            raise KeyboardInterrupt()
        wav_path = _ensure_wav(str(audio_path))

        # Optional vocal removal
        if remove_vocals:
            _report(20, '正在分离人声（首次需要下载模型）...')
            if cancel_event and cancel_event.is_set():
                raise KeyboardInterrupt()
            try:
                inst_path = separate_vocals(wav_path, output_dir, cancel_event)
                if inst_path and os.path.exists(inst_path):
                    wav_path = inst_path
                    _report(40, '人声分离完成')
            except Exception:
                pass  # fall through to original audio if vocal removal fails

        if cancel_event and cancel_event.is_set():
            raise KeyboardInterrupt()
        _report(50, '正在分析音符（basic-pitch）...')
        model_output, midi_data, note_events = predict(
            wav_path,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,
            minimum_frequency=75.0,
            maximum_frequency=2000.0,
            melodia_trick=True,
            midi_tempo=120,
        )

        if cancel_event and cancel_event.is_set():
            raise KeyboardInterrupt()
        _report(75, '估算节奏...')

        # Tempo estimate
        tempo = 120
        if midi_data and hasattr(midi_data, 'estimate_tempo'):
            try:
                tempo = int(midi_data.estimate_tempo())
            except Exception:
                pass

        # Save MIDI
        midi_path = os.path.join(output_dir, f"{task_id}.mid")
        midi_data.write(midi_path)

        _report(85, '生成吉他谱...')

        # Extract notes
        notes_raw = []
        for inst in midi_data.instruments:
            for note in inst.notes:
                notes_raw.append((
                    _py(note.pitch), _py(note.start), _py(note.end),
                    _py(note.velocity),
                ))

        # Map to fretboard
        notes_fb = midi_to_fretboard(notes_raw)

        # Organize into bars
        bars = notes_to_bars(notes_fb, beats_per_bar=4, beats_per_minute=tempo)

        # Count chords (3+ notes at same time)
        chord_count = 0
        notes_by_start = {}
        for n in notes_fb:
            sk = round(n[1] * 4) / 4
            notes_by_start.setdefault(sk, []).append(n)
        for group in notes_by_start.values():
            if len(group) >= 3:
                chord_count += 1

        duration = round(max(n[2] for n in notes_fb), 1) if notes_fb else 0

        _report(95, '生成完成')

        return {
            'status': 'done',
            'midi_url': f"/outputs/{task_id}.mid",
            'note_count': len(notes_fb),
            'chord_count': chord_count,
            'duration': duration,
            'tempo': tempo,
            'num_bars': len(bars),
            'bars': bars,
            'tuning': ['E4', 'B3', 'G3', 'D3', 'A2', 'E2'],  # high to low
        }

    except KeyboardInterrupt:
        return {'status': 'cancelled', 'message': '转录已取消'}
    except Exception as e:
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e),
        }
    finally:
        # Clean up temp WAV files and audio-separator artifacts
        for _p in filter(None, [wav_path, inst_path]):
            # Remove .conv.wav temp files
            if _p.endswith('.conv.wav') and os.path.exists(_p):
                _safe_remove(_p)
            # Remove audio-separator output subdirectories
            _stem_dir = os.path.join(
                os.path.dirname(_p),
                os.path.splitext(os.path.basename(_p))[0],
            )
            if os.path.isdir(_stem_dir) and '_' in os.path.basename(_stem_dir):
                shutil.rmtree(_stem_dir, ignore_errors=True)

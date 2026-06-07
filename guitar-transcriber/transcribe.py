"""Core transcription pipeline: audio -> MIDI -> guitar tab data"""
import os
import shutil
import subprocess
import traceback

import numpy as np
import pretty_midi
import librosa

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


def _best_single_note(pitch, tuning, max_fret):
    """Find best (string, fret) for a single note using simple cost."""
    best_str = 0
    best_fret = 0
    best_cost = float('inf')
    for i, open_pitch in enumerate(tuning):
        fret = pitch - open_pitch
        if 0 <= fret <= max_fret:
            cost = fret * 2 + abs(i - 2.5) * 0.5
            if cost < best_cost:
                best_cost = cost
                best_str = i
                best_fret = fret
    if best_cost == float('inf'):
        best_str = 5
        best_fret = max(0, pitch - tuning[5])
    return best_str, best_fret


def _map_chord_group(notes, tuning, max_fret):
    """Map a chord group: each note gets a distinct string, frets in a playable range.

    notes: list of (pitch, start, end, velocity) that occur near-simultaneously.
    Returns list of (pitch, start, end, velocity, string_index_0..5, fret).
    """
    sorted_n = sorted(notes, key=lambda n: n[0])  # sort by pitch

    # Try several base fret positions (0-12) to find the most playable shape
    best_result = None
    best_cost = float('inf')

    for base in range(13):
        result = []
        used_strings = set()
        cost = 0
        ok = True

        for pitch, start, end, velocity in sorted_n:
            best_i = -1
            best_f = -1
            best_c = float('inf')

            for i in range(6):
                if i in used_strings:
                    continue
                fret = pitch - tuning[i]
                if 0 <= fret <= max_fret:
                    # Encourage frets near base position, discourage string crossing
                    c = abs(fret - base) * 1.5 + (0.2 if fret > 12 else 0)
                    if c < best_c:
                        best_c = c
                        best_i = i
                        best_f = fret

            if best_i >= 0:
                used_strings.add(best_i)
                result.append((pitch, start, end, velocity, best_i, best_f))
                cost += best_c
            else:
                ok = False
                break

        if ok and cost < best_cost:
            best_cost = cost
            best_result = result

    # Fallback: give each note whatever string is free
    if best_result is None:
        result = []
        used = set()
        for pitch, start, end, velocity in sorted_n:
            for i in range(6):
                if i in used:
                    continue
                fret = pitch - tuning[i]
                fret = max(0, min(max_fret, fret))
                used.add(i)
                result.append((pitch, start, end, velocity, i, fret))
                break
        best_result = result

    return best_result


def midi_to_fretboard(notes, tuning=None, max_fret=MAX_FRET, chord_window=0.05):
    """Map MIDI notes to (string, fret) pairs with chord-aware optimization.

    Notes occurring within chord_window seconds of each other are grouped
    and assigned to distinct strings to produce playable chord shapes.

    notes: list of (pitch, start, end, velocity)
    Returns list of (pitch, start, end, velocity, string, fret)
      string is 1..6 (1=high e, 6=low E)
    """
    if tuning is None:
        tuning = STD_TUNING

    # Group notes by start time
    sorted_notes = sorted(notes, key=lambda n: (n[1], n[0]))
    groups = []
    cur_group = []
    cur_time = None

    for n in sorted_notes:
        t = round(n[1], 2)
        if cur_time is None or abs(t - cur_time) <= chord_window:
            cur_group.append(n)
            cur_time = t
        else:
            groups.append(cur_group)
            cur_group = [n]
            cur_time = t
    if cur_group:
        groups.append(cur_group)

    result = []
    for group in groups:
        if len(group) >= 3:
            mapped = _map_chord_group(group, tuning, max_fret)
        else:
            mapped = []
            for n in group:
                s, f = _best_single_note(n[0], tuning, max_fret)
                mapped.append((n[0], n[1], n[2], n[3], s, f))
        for item in mapped:
            pitch, start, end, vel, si, fret = item
            result.append((pitch, start, end, vel, si + 1, fret))

    return result


def notes_to_bars(notes_fb, beats_per_bar=4, beats_per_minute=120, beat_times=None):
    """Organize fretboard notes into bars/beats for tab rendering.

    If beat_times (from librosa beat tracking) is provided, notes are aligned
    to real musical beats. Otherwise falls back to rigid equal-length bars.

    Returns list of bars, each bar has beats, each beat has notes.
    """
    if not notes_fb:
        return []

    if beat_times and len(beat_times) >= 4:
        return _notes_to_bars_real(notes_fb, beat_times, beats_per_bar)
    else:
        return _notes_to_bars_rigid(notes_fb, beats_per_bar, beats_per_minute)


def _notes_to_bars_rigid(notes_fb, beats_per_bar=4, beats_per_minute=120):
    """Original rigid 4/4 bar division (fallback when no real beats available)."""
    beats_per_second = beats_per_minute / 60.0
    max_end = max(n[2] for n in notes_fb)
    total_beats = max(1, int(np.ceil(max_end * beats_per_second)))
    num_bars = max(1, int(np.ceil(total_beats / beats_per_bar)))

    bars = []
    for bar_idx in range(num_bars):
        bar_start_beat = bar_idx * beats_per_bar
        bar_end_beat = bar_start_beat + beats_per_bar

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


def _notes_to_bars_real(notes_fb, beat_times, beats_per_bar=4):
    """Align notes to real musical beats detected by librosa."""
    beat_times = list(beat_times)
    num_bars = max(1, len(beat_times) // beats_per_bar)
    bars = []

    for bar_idx in range(num_bars):
        beat_start_idx = bar_idx * beats_per_bar
        beat_end_idx = min(beat_start_idx + beats_per_bar, len(beat_times))

        if beat_end_idx - beat_start_idx < 2:
            break

        bar_start_time = beat_times[beat_start_idx]
        bar_end_time = beat_times[beat_end_idx - 1]

        # Compute average beat spacing within this bar for duration conversion
        beat_spacing = (bar_end_time - bar_start_time) / (beat_end_idx - beat_start_idx - 1) if beat_end_idx - beat_start_idx > 1 else (60.0 / 120)

        bar_notes = []
        for pitch, start, end, vel, string, fret in notes_fb:
            if bar_start_time <= start < bar_end_time:
                # Find nearest beat index within this bar
                offsets = [abs(start - beat_times[min(idx, len(beat_times) - 1)])
                           for idx in range(beat_start_idx, min(beat_end_idx, len(beat_times)))]
                nearest = min(range(len(offsets)), key=lambda i: offsets[i]) if offsets else 0
                beat_pos = nearest  # 0, 1, 2, or 3 within the bar

                dur_beats = max(0.25, (end - start) / beat_spacing)

                bar_notes.append({
                    'beat': round(beat_pos, 2),
                    'duration': round(dur_beats, 2),
                    'string': int(string),
                    'fret': int(fret),
                    'pitch': int(pitch),
                    'velocity': int(vel),
                })

        bars.append({'index': bar_idx, 'notes': bar_notes})

    return bars


def detect_beats(wav_path):
    """Detect musical beats and tempo using librosa.

    Returns (tempo_bpm, beat_time_list) or (None, None) on failure.
    """
    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True, duration=180)
        if len(y) < sr:  # less than 1 second
            return None, None
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='time')
        beat_times = beat_frames  # already in seconds when units='time'
        if beat_times is None or len(beat_times) < 4:
            return None, None
        return int(round(float(tempo))), beat_times.tolist()
    except Exception:
        return None, None


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

        # ——— Real beat detection with librosa ———
        _report(72, '检测节拍...')
        detected_tempo, beat_times = detect_beats(wav_path)
        tempo = detected_tempo if detected_tempo else 120

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

        # Map to fretboard (chord-aware mapping)
        notes_fb = midi_to_fretboard(notes_raw)

        # Organize into bars using real beats, fall back to rigid if unavailable
        bars = notes_to_bars(
            notes_fb,
            beats_per_bar=4,
            beats_per_minute=tempo,
            beat_times=beat_times,
        )

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

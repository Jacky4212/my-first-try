"""Core transcription pipeline: audio -> MIDI -> guitar tab data"""
import os
import subprocess
import traceback
import numpy as np
import pretty_midi

# Standard guitar tuning MIDI pitches (high e, B, G, D, A, low E)
STD_TUNING = [64, 59, 55, 50, 45, 40]
MAX_FRET = 24

# Look for ffmpeg in the same directory or PATH
_FFMPEG = None
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _cand in [os.path.join(_script_dir, 'ffmpeg.exe'), 'ffmpeg.exe', 'ffmpeg']:
    try:
        subprocess.run([_cand, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        _FFMPEG = _cand
        break
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


def transcribe(audio_path, output_dir, task_id):
    """Full transcription pipeline. Saves MIDI and returns tab data + metadata."""
    wav_path = None
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        wav_path = _ensure_wav(str(audio_path))

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

    except Exception as e:
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e),
        }
    finally:
        if wav_path and wav_path.endswith('.conv.wav') and os.path.exists(wav_path):
            os.remove(wav_path)

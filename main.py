import sounddevice as sd
import numpy as np
import librosa

# Guitar note frequencies (standard tuning, E2 to E4)
NOTE_FREQS = {
    'E2': 82.41, 'A2': 110.00, 'D3': 146.83, 'G3': 196.00, 'B3': 246.94, 'E4': 329.63
}

def record_audio(duration=5, fs=44100, device=None):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=device)
    sd.wait()
    print("Recording finished.")
    return audio.flatten(), fs

def get_fundamental_freq(audio, fs):
    pitches, magnitudes = librosa.piptrack(y=audio, sr=fs)
    # Focus on the middle frames to avoid attack/release noise
    n_frames = pitches.shape[1]
    start = n_frames // 4
    end = 3 * n_frames // 4
    pitches = pitches[:, start:end]
    magnitudes = magnitudes[:, start:end]
    # Find the index of the max magnitude
    idx = np.unravel_index(np.argmax(magnitudes, axis=None), magnitudes.shape)
    pitch = pitches[idx]
    return pitch if pitch > 0 else 0

def map_freq_to_note(freq):
    closest_note = min(NOTE_FREQS, key=lambda note: abs(NOTE_FREQS[note] - freq))
    return closest_note

if __name__ == "__main__":
    audio, fs = record_audio(device=1)
    print("Playing back recorded sound...")
    sd.play(audio, fs)
    sd.wait()
    freq = get_fundamental_freq(audio, fs)
    print(f"Detected frequency: {freq:.2f} Hz")
    note = map_freq_to_note(freq)
    print(f"Detected note: {note}")
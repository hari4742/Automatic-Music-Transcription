import pretty_midi
import librosa
import numpy as np


class AudioProcessor:
    def __init__(self, sample_rate: int, hop_length: int, cqt_bins: int):
        """
        Initialize the audio processor.

        Args:
            sample_rate (int): Sample rate for audio processing.
            hop_length (int): Hop length for CQT computation.
            cqt_bins (int): Number of frequency bins for CQT.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.cqt_bins = cqt_bins

    def wav_to_cqt(self, audio_path: str) -> np.ndarray:
        """
        Convert a WAV file to a Constant-Q Transform (CQT) spectrogram.

        Args:
            audio_path (str): Path to the WAV file.

        Returns:
            np.ndarray: CQT spectrogram of shape (Time, Frequency).
        """
        # Load audio file
        y, _ = librosa.load(audio_path, sr=self.sample_rate)

        # Compute CQT
        cqt = librosa.cqt(
            y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.cqt_bins,
            bins_per_octave=12 * 7  # 7 octaves × 12 bins/octave
        )

        # Transpose to (Time, Frequency)
        return cqt.T


class MIDIProcessor:
    def __init__(self, resolution: int, velocity_threshold: int):
        """
        Initialize the MIDI processor.

        Args:
            resolution (int): Frames per second for piano roll.
            velocity_threshold (int): Minimum velocity for notes.
        """
        self.resolution = resolution
        self.velocity_threshold = velocity_threshold

    def midi_to_pianoroll(self, midi_path: str) -> np.ndarray:
        """
        Convert a MIDI file to a piano roll.

        Args:
            midi_path (str): Path to the MIDI file.

        Returns:
            np.ndarray: Piano roll of shape (Time, Pitch).
        """
        # Load MIDI file
        midi = pretty_midi.PrettyMIDI(midi_path)

        # Compute piano roll
        roll = midi.get_piano_roll(
            fs=self.resolution
        )

        # Transpose to (Time, Pitch)
        return roll.T

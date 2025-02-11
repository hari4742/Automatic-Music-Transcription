import pretty_midi
import librosa
import numpy as np
import math


class AudioProcessor:
    def __init__(self, sample_rate: int, hop_length: int, cqt_bins: int, bins_per_octave: int):
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
        self.bins_per_octave = bins_per_octave

    def wav_to_cqt(self, audio_path: str) -> np.ndarray:
        """
        Convert a WAV file to a Constant-Q Transform (CQT) spectrogram.

        Args:
            audio_path (str): Path to the WAV file.

        Returns:
            np.ndarray: CQT spectrogram of shape (Frequency, Frame).
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Compute CQT
        cqt = librosa.cqt(
            y,
            sr=sr,
            hop_length=self.hop_length,
            n_bins=self.cqt_bins,
            bins_per_octave=self.bins_per_octave
        )

        # Convert to dB for better visualization
        cqt_db = librosa.amplitude_to_db(abs(cqt))

        return cqt_db, sr  # (Frequency, Frame), sample rate


class MIDIProcessor:
    def __init__(self, velocity_threshold: int):
        """
        Initialize the MIDI processor.

        Args:
            velocity_threshold (int): Minimum velocity for notes.
        """
        self.velocity_threshold = velocity_threshold

    def midi_to_pianoroll(self, midi_path: str, sample_rate: float, hop_length: float) -> np.ndarray:
        """
        Convert a MIDI file to a piano roll.

        Args:
            midi_path (str): Path to the MIDI file.
            sample_rate (float): The sample rate used in audio file
            hop_length (float): the consecutive samples used to calculate each frame

        Returns:
            np.ndarray: Piano roll of shape (Pitch, Time).
        """
        # Load MIDI file
        midi = pretty_midi.PrettyMIDI(midi_path)

        # Calculate Frames per Second
        fps = math.ceil(sample_rate / hop_length)

        # Compute piano roll
        roll = midi.get_piano_roll(
            fs=fps
        )

        roll = roll[21:109, :]  # Select only rows 21 to 108 (88 keys)
        roll = (roll > self.velocity_threshold).astype(
            np.float32)  # Convert to binary (Note ON/OFF)

        return roll  # (Pitch, Time)

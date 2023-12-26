import librosa
import soundfile as sf
import numpy as np

def fetch_segments(file_path, segment_duration, stride):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Convert durations to samples
    segment_length = int(segment_duration * sr)
    stride_length = int(stride * sr)

    # Initialize a list to store the segments
    segments = []

    # Loop through the file and extract segments
    for start in range(0, len(audio) - segment_length + 1, stride_length):
        end = start + segment_length
        segment = audio[start:end]
        segments.append(segment)

    return segments, sr

file_path = '/mnt/StorageMedia/dataset_audio/PaddySYNC.wav'

segment_duration = 1.0  # seconds
stride = 1.0  # seconds

audio_segments, sr = fetch_segments(file_path, segment_duration, stride)
processed_audio_segments = []

for i, segment in enumerate(audio_segments):
    # seg_spectrogram = librosa.stft(segment)
    # seg_spectrogram_db = librosa.amplitude_to_db(abs(seg_spectrogram))

    # model

    # output_spectrogram = librosa.db_to_amplitude(seg_spectrogram_db)
    # output_audio = librosa.istft(output_spectrogram)
    processed_audio_segments.append(segment)

concatenated_audio = np.concatenate(processed_audio_segments)
output_path = '/mnt/StorageMedia/dataset_audio/Test.wav'
sf.write(output_path, concatenated_audio, sr)

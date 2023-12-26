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
    seg_spectrogram = librosa.stft(segment, n_fft=2047, hop_length=512,center=False)

    # print (f'shape: {seg_spectrogram.shape}')

    seg_mag = np.abs(seg_spectrogram)
    seg_phase = np.angle(seg_spectrogram)
    two_channel_stft = np.stack((seg_mag, seg_phase), axis=0)

    mag = two_channel_stft[0]
    phase = two_channel_stft[1]

    mag_db = librosa.amplitude_to_db(mag)

    mag_new = librosa.db_to_amplitude(mag_db)

    stft_matrix_recombined = mag_new * np.exp(1j * phase)
    
    print (f'MAG: max: {np.max(mag)}, min: {np.min(mag)}, PHASE: max: {np.max(phase)}, min: {np.min(phase)}')
    
    seg_spectrogram_db = librosa.amplitude_to_db(abs(seg_spectrogram))

    # model

    # output_spectrogram = librosa.db_to_amplitude(seg_spectrogram_db)
    output_audio = librosa.istft(stft_matrix_recombined, n_fft=2047, hop_length=512, center=False)
    processed_audio_segments.append(output_audio)

concatenated_audio = np.concatenate(processed_audio_segments)
output_path = '/mnt/StorageMedia/dataset_audio/Test.wav'
sf.write(output_path, concatenated_audio, sr)

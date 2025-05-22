import numpy as np
import librosa
from app1.utils.logging_config import logging

def count_words(text):
    """
    Count the number of words in a text string.
    """
    return len(text.strip().split())

def extract_pitch(audio_segment, sample_rate):
    """
    Extract the average fundamental frequency (F0) for an audio segment.
    """
    audio_np = audio_segment.numpy()
    if audio_segment.shape[0] > 1:  # Convert to mono if stereo
        audio_np = audio_np[0]
    f0, _, _ = librosa.pyin(audio_np, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sample_rate)
    return np.nanmean(f0)

def extract_spectral_centroid(audio_segment, sample_rate):
    """
    Extract the spectral centroid for an audio segment.
    """
    audio_np = audio_segment.numpy()
    if audio_segment.shape[0] > 1:  # Convert to mono if stereo
        audio_np = audio_np[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate)
    return np.mean(spectral_centroid)

def calculate_speaker_stats(segments, waveform, sample_rate, labels, num_speakers):
    """
    Calculate speaker statistics such as speaking rate, pitch, and spectral centroid.
    """
    try:
        speaker_stats = {f"Speaker_{i}": {"word_count": 0, "duration": 0.0, "f0_values": [], "spectral_centroids": []} for i in range(num_speakers)}

        for i, seg in enumerate(segments):
            speaker = f"Speaker_{labels[i]}"
            text = seg["text"]
            duration = seg["end"] - seg["start"]
            word_count = count_words(text)

            # Extract pitch and spectral centroid for the segment
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = waveform[:, start_sample:end_sample]
            avg_f0 = extract_pitch(segment_audio, sample_rate)
            avg_spectral_centroid = extract_spectral_centroid(segment_audio, sample_rate)

            # Update speaker stats
            speaker_stats[speaker]["word_count"] += word_count
            speaker_stats[speaker]["duration"] += duration
            if not np.isnan(avg_f0):
                speaker_stats[speaker]["f0_values"].append(avg_f0)
            if not np.isnan(avg_spectral_centroid):
                speaker_stats[speaker]["spectral_centroids"].append(avg_spectral_centroid)

        # Calculate speaking rate, average pitch, and average spectral centroid
        for speaker, stats in speaker_stats.items():
            duration = stats["duration"]
            word_count = stats["word_count"]
            stats["speaking_rate(wpm)"] = (word_count / duration) * 60 if duration > 0 else 0.0
            stats["average_pitch"] = np.mean(stats["f0_values"]) if stats["f0_values"] else np.nan
            stats["average_spectral_centroid"] = np.mean(stats["spectral_centroids"]) if stats["spectral_centroids"] else np.nan

        return speaker_stats
    except Exception as e:
        logging.error(f"Error in calculate_speaker_stats: {e}")
        raise



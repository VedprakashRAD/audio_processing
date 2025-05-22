import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torchaudio
from app1.services.sentiment import ensemble_sentiment
from app1.utils.audio_utils import get_speaker_embedding, whisper_model
from app1.utils.logging_config import logging
from app1.services.emotion import predict_emotion

def estimate_num_speakers(embeddings, min_clusters=2, max_clusters=10):
    """
    Estimate the number of speakers dynamically using silhouette score.
    """
    n_samples = len(embeddings)
    
    # Handle case where we don't have enough samples for clustering
    if n_samples <= min_clusters:
        return 1  # Return 1 speaker if we don't have enough samples
    
    best_k = min_clusters
    best_score = -1
    max_possible = min(max_clusters, n_samples - 1)

    for k in range(min_clusters, max_possible + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(embeddings)
        if len(np.unique(labels)) == 1:  # Skip if only one cluster is found
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def diarize_with_speechbrain(audio_path, dynamic_speakers=True, min_clusters=2, max_clusters=10):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

        # Load and validate audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.nelement() == 0:
            raise ValueError("Empty audio file")

        # Get segments with validation
        result = whisper_model.transcribe(audio_path)
        segments = result.get("segments", [])
        if not segments:
            logging.error("No segments found in audio")
            raise ValueError("No segments found in audio")

        # Extract embeddings with validation
        embeddings = []
        valid_segments = []
        
        for seg in segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            
            if end_sample <= start_sample:
                logging.warning(f"Invalid segment boundaries: {seg['start']} to {seg['end']}")
                continue
                
            segment_audio = waveform[:, start_sample:end_sample]
            
            if segment_audio.nelement() == 0:
                logging.warning(f"Empty segment at {seg['start']}-{seg['end']}")
                continue

            try:
                emb = get_speaker_embedding(segment_audio, sample_rate)
                if emb is not None and emb.size > 0:
                    embeddings.append(emb)
                    valid_segments.append(seg)
            except Exception as e:
                logging.warning(f"Failed to get embedding: {e}")
                continue

        if not embeddings:
            logging.error("No valid embeddings extracted from segments")
            raise ValueError("No segments found in audio")

        embeddings = np.vstack(embeddings)
        
        # Handle case where we have very few segments
        if len(embeddings) == 1:
            # Return a simplified result with a single speaker
            diarized_segments = []
            for i, seg in enumerate(valid_segments):
                text = seg["text"]
                combined_sentiment = ensemble_sentiment(text)
                
                # Get audio segment for emotion prediction
                start_sample = int(seg['start'] * sample_rate)
                end_sample = int(seg['end'] * sample_rate)
                
                segment_audio = waveform[:, start_sample:end_sample]
                if segment_audio.numel() > 0:  
                    emotion = predict_emotion(segment_audio[0], sample_rate)
                else:
                    emotion = "neutral"
                
                diarized_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": "Speaker_0",
                    "text": text,
                    "sentiment": combined_sentiment,
                    "emotion": emotion
                })
            
            return diarized_segments

        # Step 4: Estimate number of speakers if dynamic_speakers is True
        if dynamic_speakers:
            num_speakers = estimate_num_speakers(embeddings, min_clusters, max_clusters)
        else:
            num_speakers = min_clusters
            
        # Ensure we don't try to create more clusters than we have samples
        if num_speakers >= len(embeddings):
            num_speakers = 1

        # Step 5: Cluster embeddings into the determined number of speakers
        labels = KMeans(n_clusters=num_speakers, random_state=0).fit_predict(embeddings)

        # Step 6: Assign speaker labels and sentiments
        diarized_segments = []
        for i, seg in enumerate(valid_segments):
            speaker = f"Speaker_{labels[i]}"
            text = seg["text"]
            combined_sentiment = ensemble_sentiment(text)

            # Get audio segment for emotion prediction
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            
            segment_audio = waveform[:, start_sample:end_sample]
            if segment_audio.numel() > 0:  
                emotion = predict_emotion(segment_audio[0], sample_rate)
            else:
                emotion = "neutral"

            diarized_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker,
                "text": text,
                "sentiment": combined_sentiment,
                "emotion": emotion
            })

        return diarized_segments

    except Exception as e:
        logging.error(f"Error in diarization: {e}")
        raise



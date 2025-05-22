import os
import whisper
import torchaudio
import numpy as np
from app1.utils.logging_config import logging
from sklearn.cluster import KMeans
from app1.services.diarization import diarize_with_speechbrain
from app1.utils.audio_utils import LANGUAGE_MAP, get_speaker_embedding
from app1.models.response_model import AudioAnalysisResponse, Segment
from app1.services.summarization import generate_summary_with_groq
from app1.services.sentiment import ensemble_sentiment
from app1.services.speaker_stats import calculate_speaker_stats

# Load Whisper model
whisper_model = whisper.load_model("base")

async def transcribe_audio_file(file):
    """
    Transcribes an audio file using Whisper.
    Accepts either a file-like object or a file path as a string.
    """
    if isinstance(file, str):  
        temp_file_path = file
    else:  
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

    try:
        # Transcribe audio
        result = whisper_model.transcribe(temp_file_path)
        language_code = result["language"]
        language_name = LANGUAGE_MAP.get(language_code, language_code)

        # Perform diarization with dynamic speaker detection
        diarized_segments = diarize_with_speechbrain(temp_file_path, dynamic_speakers=True, min_clusters=2, max_clusters=10)

        # Convert diarized segments to Segment model
        segments_with_metadata = [
            Segment(
                start=seg["start"],
                end=seg["end"],
                speaker=seg["speaker"],
                text=seg["text"],
                sentiment=seg["sentiment"],
                emotion=seg["emotion"]
            )
            for seg in diarized_segments
        ]

        # Extract numeric labels from speaker names (e.g "Speaker_1" -> 1)
        numeric_labels = [int(seg["speaker"].split("_")[1]) for seg in diarized_segments]

        # Calculate speaker stats
        waveform, sample_rate = torchaudio.load(temp_file_path)
        speaker_stats = calculate_speaker_stats(diarized_segments, waveform, sample_rate, numeric_labels, num_speakers=len(set(numeric_labels)))

        # Prepare extracted information for summarization
        extracted_info = {
            "transcription": result["text"],
            "segments": segments_with_metadata,
            "overall_sentiment": ensemble_sentiment(result["text"]),
            "speaker_stats": speaker_stats
        }

        try:
            # Generate structured summary
            structured_summary = generate_summary_with_groq(extracted_info)

            # Extract details from structured summary
            speaker_identification = structured_summary.get("speaker_roles", {})
            important_details = structured_summary.get("important_details", {})
            summary_text = structured_summary.get("summary", "")

            # Create response using AudioAnalysisResponse model
            response = AudioAnalysisResponse(
                language=language_name,
                transcription=result["text"],
                segments=segments_with_metadata,
                overall_sentiment=extracted_info["overall_sentiment"],
                speaker_stats=extracted_info["speaker_stats"],
                speaker_identification=speaker_identification,
                important_details=important_details,
                summary=summary_text
            )

            return response.dict()
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return {
                "error": "An error occurred during transcription. Please try again later."
            }
    
    finally:
        if not isinstance(file, str) and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


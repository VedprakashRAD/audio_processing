from fastapi import APIRouter, HTTPException, status, Body
from fastapi.responses import JSONResponse
from app1.services.transcription import transcribe_audio_file
from app1.services.all_graphs import plot_audio_with_speakers
from app1.utils.logging_config import logging
import tempfile
import os
import json
import librosa
import numpy as np
import requests
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AudioURLInput(BaseModel):
    """Model for audio URL input"""
    url: str

router = APIRouter(
    prefix="/api/v1",
    tags=["audio"]
)

async def calculate_dynamic_lufs_threshold(file_path):
    """
    Calculate a dynamic LUFS threshold based on audio content.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        float: Recommended LUFS threshold value
    """
    # Get default threshold from environment variable or use 18.0
    default_threshold = float(os.environ.get("LUFS_DEFAULT_THRESHOLD", 18.0))
    min_threshold = float(os.environ.get("LUFS_MIN_THRESHOLD", 15.0))
    max_threshold = float(os.environ.get("LUFS_MAX_THRESHOLD", 22.0))
    
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Calculate percentiles for dynamic thresholding
        rms_25 = np.percentile(rms, 25)  # Quieter parts
        rms_75 = np.percentile(rms, 75)  # Louder parts
        rms_95 = np.percentile(rms, 95)  # Very loud parts (potential emotional speech)
        
        # Convert to dB scale (approximation of LUFS)
        if rms_75 > 0:  # Avoid log of zero
            db_75 = 20 * np.log10(rms_75)
            
            # Adjust threshold based on audio characteristics
            if db_75 < -30:  # Very quiet audio
                return min_threshold  # More sensitive threshold
            elif db_75 > -15:  # Very loud audio
                return max_threshold  # Less sensitive threshold
            else:
                # Linear mapping between -30 and -15 dB to threshold range min-max
                return min_threshold + (max_threshold - min_threshold) * (db_75 + 30) / 15.0
        
        return default_threshold  # Default if calculation fails
    except Exception as e:
        logging.error(f"Error calculating dynamic LUFS threshold: {e}")
        return default_threshold  # Default fallback

async def download_file(url, target_path):
    """
    Download file from URL to target path
    
    Args:
        url: URL to download from
        target_path: Path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        logging.error(f"Error downloading file from {url}: {e}")
        return False

@router.post("/audioanalysis/", summary="Analyze audio file from URL", description="Provide a URL to an audio file for transcription, speaker diarization, sentiment analysis, and visualization")
async def audio_analysis(
    audio_data: AudioURLInput = Body(..., description="URL to audio file")
):
    """
    Endpoint to analyze audio from a URL.
    """
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file_path = temp_file.name
            
            # Handle URL to audio file
            audio_url = audio_data.url
            if not audio_url.endswith('.mp3') and not audio_url.endswith('.wav'):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "message": "Only MP3 or WAV audio files are supported",
                        "data": None
                    }
                )
            
            # Download the file
            download_success = await download_file(audio_url, temp_file_path)
            if not download_success:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "message": f"Failed to download audio file from URL: {audio_url}",
                        "data": None
                    }
                )

        try:
            # Determine LUFS threshold value automatically
            used_threshold = await calculate_dynamic_lufs_threshold(temp_file_path)
            logging.info(f"Automatically determined LUFS threshold: {used_threshold}")
            
            # Transcription
            transcription_result = await transcribe_audio_file(temp_file_path)

            # Plot
            plot_json = await plot_audio_with_speakers(
                temp_file_path, lufs_threshold_value=used_threshold
            )
            plot = json.loads(plot_json)

            return JSONResponse(
                content={
                    "success": True,
                    "message": "Audio analysis completed successfully",
                    "data": {
                        "transcription": transcription_result,
                        "plot": plot,
                        "used_threshold": used_threshold,
                        "input_source": "url",
                        "input_name": audio_data.url
                    }
                }
            )

        except Exception as e:
            logging.error(f"Error in /audioanalysis/ endpoint: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "message": f"An error occurred during audio analysis: {str(e)}",
                    "data": None
                }
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    except Exception as e:
        logging.error(f"File handling error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": f"File processing error: {str(e)}",
                "data": None
            }
        )

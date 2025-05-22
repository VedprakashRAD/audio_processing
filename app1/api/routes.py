from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse
from app1.services.transcription import transcribe_audio_file
from app1.services.all_graphs import plot_audio_with_speakers
from app1.utils.logging_config import logging
import tempfile
import os
import json
import librosa
import numpy as np

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
                return 15.0  # More sensitive threshold
            elif db_75 > -15:  # Very loud audio
                return 22.0  # Less sensitive threshold
            else:
                # Linear mapping between -30 and -15 dB to threshold range 15-22
                return 15.0 + (22.0 - 15.0) * (db_75 + 30) / 15.0
        
        return 18.0  # Default if calculation fails
    except Exception as e:
        logging.error(f"Error calculating dynamic LUFS threshold: {e}")
        return 18.0  # Default fallback

@router.post("/audioanalysis/", summary="Analyze audio file", description="Upload an audio file for transcription, speaker diarization, sentiment analysis, and visualization")
async def audio_analysis(
    file: UploadFile = File(..., description="Audio file to analyze (MP3 format recommended)")
):
    """
    Combined endpoint to return transcription and plot.
    """
    # Validate file
    if not file.filename or not (file.filename.endswith('.mp3') or file.filename.endswith('.wav')):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "Only MP3 or WAV audio files are supported",
                "data": None
            }
        )
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            contents = await file.read()
            if not contents:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "message": "Empty file uploaded",
                        "data": None
                    }
                )
            temp_file.write(contents)
            temp_file_path = temp_file.name

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
                        "used_threshold": used_threshold
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

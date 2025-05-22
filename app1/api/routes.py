from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse
from app1.services.transcription import transcribe_audio_file
from app1.services.all_graphs import plot_audio_with_speakers
from app1.utils.logging_config import logging
import tempfile
import os
import json

router = APIRouter(
    prefix="/api/v1",
    tags=["audio"]
)

@router.post("/audioanalysis/", summary="Analyze audio file", description="Upload an audio file for transcription, speaker diarization, sentiment analysis, and visualization")
async def audio_analysis(
    file: UploadFile = File(..., description="Audio file to analyze (MP3 format recommended)"),
    lufs_threshold_value: float = Query(18.0, description="LUFS threshold for plot")
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
            # Transcription
            transcription_result = await transcribe_audio_file(temp_file_path)

            # Plot
            plot_json = await plot_audio_with_speakers(
                temp_file_path, lufs_threshold_value=lufs_threshold_value
            )
            plot = json.loads(plot_json)

            return JSONResponse(
                content={
                    "success": True,
                    "message": "Audio analysis completed successfully",
                    "data": {
                        "transcription": transcription_result,
                        "plot": plot
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

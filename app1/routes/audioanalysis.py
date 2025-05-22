from fastapi import APIRouter, HTTPException, UploadFile, File
from app1.services.diarization import diarize_with_speechbrain

router = APIRouter()

@router.post("/audioanalysis/")
async def audio_analysis_endpoint(audio_file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        temp_audio_path = f"temp_{audio_file.filename}"
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(audio_file.file.read())

        # Perform diarization
        result = diarize_with_speechbrain(temp_audio_path)

        # Clean up the temporary file
        os.remove(temp_audio_path)

        return {"result": result}
    except ValueError as ve:
        # Return the specific error message (e.g., "No segments found in audio")
        return {"error": str(ve)}
    except Exception as e:
        # For unexpected errors, return a generic message
        return {"error": "An unexpected error occurred: " + str(e)}
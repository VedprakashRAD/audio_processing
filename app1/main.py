import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app1.api.routes import router

# Initialize language models on startup
import nltk
import spacy
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')

# Create FastAPI app
app = FastAPI(
    title="Audio Analysis API",
    description="""
    API for audio transcription, speaker diarization, sentiment analysis, and more.
    
    ## Features
    
    - **Audio Transcription**: Converts audio files into text using OpenAI's Whisper model.
    - **Speaker Diarization**: Dynamically detects and labels speakers in the audio.
    - **Sentiment Analysis**: Analyzes the sentiment of transcribed text for each speaker segment.
    - **Text Summarization**: Generates concise summaries of the transcribed text.
    - **Speaker Statistics**: Provides detailed statistics for each speaker.
    - **Graphical Analysis**: Generates visual representations of audio data.
    - **Dynamic LUFS Thresholding**: Automatically determines optimal loudness thresholds based on audio content.
    
    ## LUFS Threshold Explanation
    
    The LUFS (Loudness Units Full Scale) threshold is used to detect significant audio events and speaker changes:
    
    - **Lower threshold** (15.0): More sensitive, better for quiet recordings or detecting subtle changes
    - **Default threshold** (18.0): Balanced for most audio content
    - **Higher threshold** (22.0): Less sensitive, better for noisy recordings or focusing on louder speech
    
    The API can automatically determine the optimal threshold based on the audio characteristics.
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "url": "https://github.com/yourusername/audioanalysis",
    },
    license_info={
        "name": "MIT License",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include router
app.include_router(router)

# Root endpoint for health checks
@app.get("/", tags=["health"])
async def root():
    return {
        "success": True,
        "message": "Audio Analysis API is running",
        "data": {
            "version": "1.0.0",
            "documentation": "/docs"
        }
    }

# Get PORT from environment variable for Railway
port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app1.main:app", host="0.0.0.0", port=port, reload=False)



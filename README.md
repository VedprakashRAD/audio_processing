# Audio Analysis

This project provides an API for audio transcription, speaker diarization, sentiment analysis, summarization, and graphical analysis. It leverages state-of-the-art machine learning models like Whisper for transcription and SpeechBrain for speaker recognition.

## Features

- **Audio Transcription**: Converts audio files into text using OpenAI's Whisper model.
- **Speaker Diarization**: Dynamically detects and labels speakers in the audio using embeddings and clustering.
- **Sentiment Analysis**: Analyzes the sentiment of transcribed text for each speaker segment.
- **Text Summarization**: Generates concise summaries of the transcribed text.
- **Speaker Statistics**: Provides detailed statistics for each speaker, including speaking rate and pitch.
- **Graphical Analysis**: Generates visual representations of audio data, such as speaker activity graphs and loudness detection.

## Project Structure

```
audio
│
├── app1
│   ├── __init__.py             # Marks this directory as a Python package
│   ├── main.py                 # Entry point for the FastAPI application
│   ├── api
│   │   ├── __init__.py         # Marks this directory as a Python package
│   │   └── routes.py           # FastAPI routes for handling API requests
│   ├── models
│   │   ├── __init__.py         # Marks this directory as a Python package
│   │   └── response_model.py   # Pydantic models for API responses
│   ├── services
│   │   ├── __init__.py         # Marks this directory as a Python package
│   │   ├── diarization.py      # Speaker diarization logic
│   │   ├── summarization.py    # Text summarization logic
│   │   ├── transcription.py    # Audio transcription and integration logic
│   │   ├── sentiment.py        # Sentiment analysis logic
│   │   ├── speaker_stats.py    # Speaker statistics calculation
│   │   └── all_graphs.py       # Additional graph-related logic (e.g., visualizations)
│   ├── utils
│       ├── __init__.py         # Marks this directory as a Python package
│       ├── audio_utils.py      # Utility functions for audio processing
│       └── logging_config.py   # Configuration for logging
│
├── requirements.txt            # All the dependencies
├── README.md                   # Project documentation

```

## How It Works

1. **Transcription**:
   - The `transcription.py` service uses Whisper to transcribe audio files into text.
   - Language detection is performed automatically.

2. **Speaker Diarization**:
   - The `diarization.py` service dynamically estimates the number of speakers using silhouette scores.
   - Speaker embeddings are extracted, and clustering is performed to label speakers.

3. **Sentiment Analysis**:
   - The `sentiment.py` service analyzes the sentiment of each speaker's transcribed text.

4. **Summarization**:
   - The `summarization.py` service generates a concise summary of the entire transcription.

5. **Speaker Statistics**:
   - The `speaker_stats.py` service calculates speaking rate and pitch for each speaker.

6. **Graphical Analysis**:
   - The `all_graphs.py` service provides tools to generate visualizations, such as:
     - Speaker activity over time.
     - Loudness detection.

## API Endpoints

### `/transcribe/` (POST)
- **Description**: Transcribes an audio file and returns detailed information, including speaker diarization, sentiment analysis, and summarization.
- **Request**:
  - `file`: Audio file to be transcribed (e.g., `.mp3`, `.wav`).
- **Response**:
  - `language`: Detected language of the audio.
  - `transcription`: Full transcription of the audio.
  - `segments`: List of speaker segments with start time, end time, speaker label, text, and sentiment.
  - `overall_sentiment`: Overall sentiment of the transcription.
  - `summary`: Summary of the transcription.
  - `speaker_stats`: Statistics for each speaker (e.g., speaking rate, average pitch).
  - `speaker_identification`: Mapping of speaker labels to their roles (e.g., "Executive", "Customer").
  - `important_details`: Key extracted details, such as:
    - `Phone_number`: Contact number if mentioned.
    - `Place_of_service`: Location where assistance is required.
    - `Vehicle_information`: Details about the vehicle (e.g., model, year, registration number).
    - `Service_related_details`: Services requested (e.g., battery jump starts, fuel delivery).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd audio
   ```

2. Create a virtual environment:
   ```bash
   python -m venv myenv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install required language models:
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader vader_lexicon
   ```

## Usage

1. Start the FastAPI server:
   ```bash
   uvicorn app1.main:app --reload
   ```

2. Access the API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```

3. Use the `/audioanalysis/` endpoint to upload audio files and receive transcription results.

## Dependencies
- fastapi
- uvicorn
- whisper
- torchaudio
- scikit-learn
- speechbrain
- nltk
- numpy
- spacytextblob
- spacy
- nltk
- librosa
- pyloudnorm
- python-multipart
- ffmpeg-python


install this in command prompt: 
- python -m spacy download en_core_web_sm
- python -m nltk.downloader vader_lexicon

## Environment Variables

This application supports dynamic configuration through environment variables. Below are the key environment variables you can use:

### Server Configuration
- `PORT`: The port to run the server on (default: 8000)
- `HOST`: The host to run the server on (default: 0.0.0.0)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed origins for CORS (default: *)

### Model Configuration
- `WHISPER_MODEL`: The Whisper model size to use for transcription (options: tiny, base, small, medium, large; default: base)

### LUFS Threshold Settings
- `LUFS_DEFAULT_THRESHOLD`: Default LUFS threshold value if dynamic calculation fails (default: 18.0)
- `LUFS_MIN_THRESHOLD`: Minimum LUFS threshold for very quiet audio (default: 15.0)
- `LUFS_MAX_THRESHOLD`: Maximum LUFS threshold for very loud audio (default: 22.0)

### API Keys
- `GROQ_API_KEY`: Your GROQ API key for text summarization services

## Deployment

This project is configured for deployment on Railway and other container platforms:

1. Set up the required environment variables on your platform
2. Deploy using the Dockerfile or railway.json configuration
3. The application will automatically run with the specified settings

### Prerequisites

1. Create a [Railway](https://railway.app/) account
2. Install the [Railway CLI](https://docs.railway.app/develop/cli) (optional)

### Deployment Steps

#### Option 1: Deploy via Railway Dashboard

1. Create a new project in Railway
2. Connect your GitHub repository
3. Add the required environment variables:
   - `GROQ_API_KEY`: Your Groq API key
4. Deploy the project

#### Option 2: Deploy via Railway CLI

1. Login to Railway:
   ```bash
   railway login
   ```

2. Link to your project:
   ```bash
   railway link
   ```

3. Add environment variables:
   ```bash
   railway variables set GROQ_API_KEY=your_groq_api_key_here
   ```

4. Deploy the project:
   ```bash
   railway up
   ```

### Configuration Files

The project includes the following configuration files for Railway:

- `Procfile`: Defines the command to start the application
- `runtime.txt`: Specifies the Python version
- `railway.json`: Configures the build and deployment process
- `setup.py`: Installs required language models during deployment

                                 

# Audio Analysis API - Example Curl Commands

## Testing with URL to audio file

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/audioanalysis/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://public-prod-ready.s3.ap-south-1.amazonaws.com/call-recordings/VCejtqIlV_11833.mp3"
  }'
```

## Testing health endpoint

```bash
curl -X 'GET' \
  'http://localhost:8000/' \
  -H 'accept: application/json'
```

## Notes

- Replace `localhost:8000` with your actual server address
- Make sure the URL points to a valid MP3 or WAV file
- The API will automatically determine the optimal LUFS threshold based on the audio content
- Results include transcription, speaker diarization, sentiment analysis, and visual plots 
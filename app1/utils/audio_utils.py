import torchaudio.transforms as T
from speechbrain.inference import SpeakerRecognition
import whisper

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load SpeechBrain ECAPA-TDNN Speaker Encoder
speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir"
)

# Resample transform for audio to 16kHz mono
resampler = T.Resample(orig_freq=44100, new_freq=16000)

LANGUAGE_MAP = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "ml": "Malayalam",
    "mr": "Marathi",
    "kn": "Kannada",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "as": "Assamese",
    "or": "Odia",
    "en": "English"
}

def get_speaker_embedding(audio_segment, sample_rate=16000):
    if audio_segment.shape[0] > 1:
        audio_segment = audio_segment[0:1, :]
    if sample_rate != 16000:
        audio_segment = resampler(audio_segment)
    return speaker_model.encode_batch(audio_segment).squeeze(0).detach().numpy()




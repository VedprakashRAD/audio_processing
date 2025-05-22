import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from app1.utils.logging_config import logging

# Load model and feature extractor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(device)
# Ensure FP32 on CPU
if device == "cpu":
    model = model.float()

def predict_emotion(waveform, sample_rate):
    try:
        # Enhanced input validation with detailed logging
        if waveform is None or not isinstance(waveform, torch.Tensor):
            logging.warning(f"Invalid waveform input type: {type(waveform)}")
            return "neutral"

        if waveform.nelement() == 0:
            logging.warning("Empty waveform received")
            return "neutral"

        # Convert stereo to mono and ensure minimum length
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        
        # Minimum length should be at least 1 second at target sample rate
        min_length = 16000  # 1 second at 16kHz
        if len(waveform) < min_length:
            logging.warning(f"Waveform too short: {len(waveform)} samples")
            return "neutral"

        # Enhanced normalization with checks
        max_val = waveform.abs().max()
        if max_val > 0:  # Prevent division by zero
            waveform = waveform / max_val
        else:
            logging.warning("Silent audio segment detected")
            return "neutral"

        # Resample with validation
        if sample_rate != 16000:
            try:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=16000
                )
                waveform = resampler(waveform)
            except Exception as e:
                logging.error(f"Resampling failed: {e}")
                return "neutral"

        # Feature extraction with enhanced error handling
        try:
            waveform = waveform.cpu()
            waveform_np = waveform.numpy()
            
            if not np.isfinite(waveform_np).all():
                logging.warning("Non-finite values detected in audio")
                return "neutral"
                
            inputs = feature_extractor(
                waveform_np, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(device).float() for k, v in inputs.items()}  # Ensure FP32

            # Prediction with validation
            with torch.no_grad():
                if inputs['input_values'].size(1) == 0:
                    logging.error("Empty segment detected")
                    return "neutral"
                    
                outputs = model(**inputs)
                if not hasattr(outputs, 'logits'):
                    logging.error("Model output missing logits")
                    return "neutral"
                    
                predicted_id = torch.argmax(outputs.logits, dim=-1).item()
                emotion = model.config.id2label.get(predicted_id, "neutral")
                
            return emotion

        except Exception as e:
            logging.error(f"Feature extraction or prediction failed: {str(e)}")
            return "neutral"

    except Exception as e:
        logging.error(f"Unexpected error in emotion prediction: {str(e)}")
        return "neutral"
    
    
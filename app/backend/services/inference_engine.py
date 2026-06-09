import os
import time
import tempfile
import requests
import asyncio
import numpy as np
import librosa
import torch
from core.config import JAMENDO_TAGS
from models import MusicCNNAttention, MusicCRNN

ai_models = {}
device = torch.device("cpu")

def load_model(model_name: str):
    global ai_models
    
    if model_name in ai_models:
        return ai_models[model_name]
        
    num_classes = len(JAMENDO_TAGS)
    if model_name == "cnn_self_attention":
        model_path = "models/weights/cnn_self_attention/best_model.pth"
        model = MusicCNNAttention(num_classes=num_classes)
    elif model_name == "crnn":
        model_path = "models/weights/crnn/best_model.pth"
        model = MusicCRNN(num_classes=num_classes)
    else:
        print(f"ERROR: Unknown model '{model_name}'.")
        return None

    if os.path.exists(model_path):
        print(f"Loading {model_name} into memory...")
        checkpoint = torch.load(model_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        ai_models[model_name] = model
    else:
        print(f"CRITICAL ERROR: Model file not found at {model_path}")
        return None
        
    return ai_models[model_name]

def warmup_models():
    print("Warming up models to avoid Cold Start delay...")
    t0 = time.time()
    try:
        # Dummy audio: 30 seconds of silence
        y = np.zeros(22050 * 30, dtype=np.float32)
        sr = 22050
        
        # Force Numba JIT compilation for Librosa
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        audio_tensor = torch.tensor(spectrogram_db).unsqueeze(0).unsqueeze(0).float()

        # Force PyTorch C++ / CPU threads initialization
        for model_name in ["cnn_self_attention", "crnn"]:
            model = load_model(model_name)
            if model:
                with torch.no_grad():
                    _ = model(audio_tensor)
                    
        print(f"Warmup completed in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Warmup failed: {e}")

def _evaluate_song_sync(song: dict, emotion: str, model_name: str) -> dict:
    loaded_ai_model = load_model(model_name)
    if not loaded_ai_model:
        return {"error": "Missing .pth file"}

    temp_path = None
    try:
        t_start = time.time()

        # Fetch Audio from API
        audio_response = requests.get(song["preview_url"])
        _, temp_path = tempfile.mkstemp(suffix=".mp3")
        with open(temp_path, "wb") as f:
            f.write(audio_response.content)
        t_fetch = time.time()

        # Calculate Mel-Spectrogram
        y, sr = librosa.load(temp_path, sr=22050, offset=15.0, duration=30.0)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        os.remove(temp_path)
        temp_path = None
        t_spectrogram = time.time()

        # Inference
        audio_tensor = torch.tensor(spectrogram_db).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad(): 
            logits_output = loaded_ai_model(audio_tensor)
            probabilities = torch.sigmoid(logits_output)[0] 
            
            idx_requested_emotion = JAMENDO_TAGS.index(emotion)
            requested_confidence = probabilities[idx_requested_emotion].item()

            max_index = torch.argmax(probabilities).item()
            dominant_emotion = JAMENDO_TAGS[max_index]
        
        t_ia = time.time()

        print(f"Analyzing: {song['title']}")
        print(f"  - API Audio Fetch: {t_fetch - t_start:.2f}s")
        print(f"  - Spectrogram Calc: {t_spectrogram - t_fetch:.2f}s")
        print(f"  - AI Inference: {t_ia - t_spectrogram:.2f}s")
        print(f"  -> Request '{emotion}'. Confidence: {requested_confidence*100:.1f}% (Dominant: {dominant_emotion})")
        print("-" * 30)

        # Acceptance criteria
        is_approved = (requested_confidence >= 0.50 or dominant_emotion == emotion)
        
        return {
            "is_approved": is_approved,
            "confidence": requested_confidence,
            "dominant": dominant_emotion,
            "song": song,
            "error": None
        }

    except Exception as e:
        print(f"Error processing {song['title']}: {e}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return {"error": str(e)}

async def evaluate_song(song: dict, emotion: str, model_name: str) -> dict:
    return await asyncio.to_thread(_evaluate_song_sync, song, emotion, model_name)

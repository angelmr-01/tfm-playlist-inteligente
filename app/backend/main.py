import os
import tempfile
import numpy as np
import librosa
import torch
import requests
import random
import time
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from models import MusicCNNAttention, MusicCRNN
from enum import Enum


# Server Configuration
app = FastAPI(title="TFM - AI Music Emotions API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Configuration
# Maintain the same index order as the Colab training script
JAMENDO_TAGS = [
    'calm', 
    'dark', 
    'emotional', 
    'energetic', 
    'epic', 
    'happy', 
    'melancholic', 
    'powerful', 
    'relaxing', 
    'romantic', 
    'sad', 
    'upbeat'
]

ai_models = {}
device = torch.device("cpu")

def load_model(model_name: str):
    global ai_models
    
    if model_name in ai_models:
        return ai_models[model_name]
        
    num_classes = len(JAMENDO_TAGS)
    if model_name == "cnn_self_attention":
        model_path = "checkpoint/cnn_self_attention/best_model.pth"
        model = MusicCNNAttention(num_classes=num_classes)
    elif model_name == "crnn":
        model_path = "checkpoint/crnn/best_model.pth"
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

# Global Eager Loading
print("Eager Loading: Loading models into RAM globally...")
load_model("cnn_self_attention")
load_model("crnn")
print("Global model loading completed.")

# Jamendo Configuration
JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID")
if not JAMENDO_CLIENT_ID:
    print("WARNING: JAMENDO_CLIENT_ID missing in .env")
else:
    print("Jamendo credentials detected.")


class AvailableModels(str, Enum):
    cnn_self_attention = "cnn_self_attention"
    crnn = "crnn"

class AvailableEmotions(str, Enum):
    calm = "calm"
    dark = "dark"
    emotional = "emotional"
    energetic = "energetic"
    epic = "epic"
    happy = "happy"
    melancholic = "melancholic"
    powerful = "powerful"
    relaxing = "relaxing"
    romantic = "romantic"
    sad = "sad"
    upbeat = "upbeat"

@app.get("/api/playlist/{emotion}")
async def generate_playlist(emotion: AvailableEmotions, model: AvailableModels):
    emotion = emotion.value
    if not JAMENDO_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Missing JAMENDO_CLIENT_ID")

    # Heuristic Mapping: Broad genres to filter candidates
    jamendo_mapping = {
        "happy":       {"tags": "pop", "speed": "high"},
        "sad":         {"tags": "piano", "speed": "low"},
        "energetic":   {"tags": "electronic", "speed": "high"},
        "relaxing":    {"tags": "ambient", "speed": "low"},
        "dark":        {"tags": "cinematic", "speed": "low"},
        "romantic":    {"tags": "acoustic", "speed": "low"},
        "emotional":   {"tags": "soundtrack", "speed": "low"},
        "upbeat":      {"tags": "dance", "speed": "high"},
        "epic":        {"tags": "orchestral", "speed": "high"},
        "melancholic": {"tags": "classical", "speed": "low"},
        "calm":        {"tags": "ambient", "speed": "low"},
        "powerful":    {"tags": "rock", "speed": "high"}
    }

    url_base = "https://api.jamendo.com/v3.0/tracks/"
    
    # Random offsets to avoid repetitive tracks
    random_offset_a = random.randint(0, 150)
    random_offset_b = random.randint(0, 500)

    try:
        # Query A: Heuristic sampling
        params_a = {
            "client_id": JAMENDO_CLIENT_ID, "format": "json", "limit": 15,
            "include": "musicinfo", "offset": random_offset_a
        }
        params_a.update(jamendo_mapping[emotion])
        response_a = requests.get(url_base, params=params_a).json()

        # Query B: Popularity sampling (Noise)
        params_b = {
            "client_id": JAMENDO_CLIENT_ID, "format": "json", "limit": 15,
            "order": "popularity_total", "offset": random_offset_b
        }
        response_b = requests.get(url_base, params=params_b).json()

        combined_tracks = response_a.get('results', []) + response_b.get('results', [])
        raw_candidates = [
            {
                "id": t['id'], "title": t['name'], "artist": t['artist_name'],
                "preview_url": t['audio'], "image": t['image']
            } for t in combined_tracks if t.get('audio')
        ]
        random.shuffle(raw_candidates)

        # Real-time async evaluation generator
        async def song_generator():
            buffer = []
            last_send_time = time.time()
            total_approved = 0
            
            loaded_ai_model = load_model(model.value)

            for song in raw_candidates:
                if total_approved >= 10:
                    break
                    
                if not loaded_ai_model:
                    yield "data: {\"error\": \"Falta el archivo .pth\"}\n\n"
                    break

                try:
                    t_start = time.time()

                    audio_response = requests.get(song["preview_url"])
                    _, temp_path = tempfile.mkstemp(suffix=".mp3")
                    with open(temp_path, "wb") as f:
                        f.write(audio_response.content)
                    t_download = time.time()

                    y, sr = librosa.load(temp_path, sr=22050, duration=30.0)
                    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
                    os.remove(temp_path)
                    t_librosa = time.time()

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
                    print(f"  - Download MP3: {t_download - t_start:.2f}s")
                    print(f"  - Librosa:      {t_librosa - t_download:.2f}s")
                    print(f"  - AI Inference: {t_ia - t_librosa:.2f}s")
                    print(f"  -> Request '{emotion}'. Confidence: {requested_confidence*100:.1f}% (Dominant: {dominant_emotion})")
                    print("-" * 30)

                    # Acceptance criteria
                    if requested_confidence >= 0.50 or dominant_emotion == emotion:
                        buffer.append(song)
                        total_approved += 1

                except Exception as e:
                    print(f"Error processing {song['title']}: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                elapsed_time = time.time() - last_send_time
                if len(buffer) >= 2 or (len(buffer) == 1 and elapsed_time > 10):
                    yield f"data: {json.dumps(buffer)}\n\n"
                    buffer = [] 
                    last_send_time = time.time() 

            if buffer:
                yield f"data: {json.dumps(buffer)}\n\n"
            yield "data: [DONE]\n\n"

        # Return the async generator wrapped in a StreamingResponse
        return StreamingResponse(song_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
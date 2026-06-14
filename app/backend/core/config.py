import os
from enum import Enum

JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID")

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

class AvailableModels(str, Enum):
    cnn = "cnn"
    crnn = "crnn"
    mobilenet = "mobilenet"

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

# Heuristic Mapping: Broad genres to filter candidates
JAMENDO_MAPPING = {
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

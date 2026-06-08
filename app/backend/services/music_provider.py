import requests
import random
import asyncio
from core.config import JAMENDO_CLIENT_ID, JAMENDO_MAPPING

def _fetch_candidates_sync(emotion: str) -> list:
    url_base = "https://api.jamendo.com/v3.0/tracks/"
    
    # Random offsets to avoid repetitive tracks
    random_offset_a = random.randint(0, 150)
    random_offset_b = random.randint(0, 500)

    # Query A: Heuristic sampling
    params_a = {
        "client_id": JAMENDO_CLIENT_ID, "format": "json", "limit": 15,
        "include": "musicinfo", "offset": random_offset_a
    }
    params_a.update(JAMENDO_MAPPING[emotion])
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
    return raw_candidates

async def fetch_candidates(emotion: str) -> list:
    if not JAMENDO_CLIENT_ID:
        raise ValueError("Missing JAMENDO_CLIENT_ID")
        
    return await asyncio.to_thread(_fetch_candidates_sync, emotion)

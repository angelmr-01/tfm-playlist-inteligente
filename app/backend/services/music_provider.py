import requests
import random
import asyncio
from core.config import JAMENDO_CLIENT_ID, JAMENDO_MAPPING

def _fetch_candidates_sync(emotion: str) -> list:
    url_base = "https://api.jamendo.com/v3.0/tracks/"
    
    # Random offsets to avoid repetitive tracks
    heuristic_offset = random.randint(0, 150)
    popularity_offset = random.randint(0, 500)

    # Query A: Heuristic sampling
    heuristic_params = {
        "client_id": JAMENDO_CLIENT_ID, "format": "json", "limit": 15,
        "include": "musicinfo", "offset": heuristic_offset
    }
    heuristic_params.update(JAMENDO_MAPPING[emotion])
    heuristic_response = requests.get(url_base, params=heuristic_params).json()

    # Query B: Popularity sampling (Noise)
    popularity_params = {
        "client_id": JAMENDO_CLIENT_ID, "format": "json", "limit": 15,
        "order": "popularity_total", "offset": popularity_offset
    }
    popularity_response = requests.get(url_base, params=popularity_params).json()

    combined_tracks = heuristic_response.get('results', []) + popularity_response.get('results', [])
    raw_candidates = [
        {
            "id": track['id'], "title": track['name'], "artist": track['artist_name'],
            "preview_url": track['audio'], "image": track['image']
        } for track in combined_tracks if track.get('audio')
    ]
    random.shuffle(raw_candidates)
    return raw_candidates

async def fetch_candidates(emotion: str) -> list:
    if not JAMENDO_CLIENT_ID:
        raise ValueError("Missing JAMENDO_CLIENT_ID")
        
    return await asyncio.to_thread(_fetch_candidates_sync, emotion)

import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from core.config import AvailableModels, AvailableEmotions
from services.music_provider import fetch_candidates
from services.inference_engine import evaluate_song, load_model

# Server Configuration
app = FastAPI(title="TFM - AI Music Emotions API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Eager Loading
print("Eager Loading: Loading models into RAM globally...")
load_model("cnn_self_attention")
load_model("crnn")
print("Global model loading completed.")


@app.get("/api/playlist/{emotion}")
async def generate_playlist(emotion: AvailableEmotions, model: AvailableModels):
    emotion_val = emotion.value
    model_val = model.value

    try:
        # 1. Fetch Candidates (Async)
        raw_candidates = await fetch_candidates(emotion_val)

        # 2. Real-time async evaluation generator
        async def song_generator():
            buffer = []
            total_approved = 0

            for song in raw_candidates:
                if total_approved >= 10:
                    break

                # 3. Evaluate each song concurrently without blocking the event loop
                result = await evaluate_song(song, emotion_val, model_val)

                if result.get("error"):
                    print(f"Error processing {song['title']}: {result['error']}")
                    # If it's a critical missing file error, we inform the UI
                    if "Falta el archivo" in result["error"]:
                        yield f"data: {{\"error\": \"{result['error']}\"}}\n\n"
                        break
                    continue

                if result["is_approved"]:
                    buffer.append(song)
                    total_approved += 1

                # Send immediately if buffer is not empty to improve UX responsiveness
                if len(buffer) >= 1:
                    yield f"data: {json.dumps(buffer)}\n\n"
                    buffer = []

            if buffer:
                yield f"data: {json.dumps(buffer)}\n\n"
            yield "data: [DONE]\n\n"

        # Return the async generator wrapped in a StreamingResponse
        return StreamingResponse(song_generator(), media_type="text/event-stream")

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
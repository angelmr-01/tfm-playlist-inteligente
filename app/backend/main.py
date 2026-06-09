import json
import time
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from core.config import AvailableModels, AvailableEmotions
from services.music_provider import fetch_candidates
from services.inference_engine import evaluate_song, load_model, warmup_models

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
print("Eager Loading: Loading models into RAM...")
load_model("cnn_self_attention")
load_model("crnn")
print("Model loading completed.")
warmup_models()


@app.get("/health")
async def health_check():
    return {"status": "ready"}

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
            
            # Concurrency limit to evaluate at most 4 songs at once (preventing CPU/RAM crash)
            concurrency_limiter = asyncio.Semaphore(4)

            async def bounded_evaluate(song_item):
                async with concurrency_limiter:
                    return await evaluate_song(song_item, emotion_val, model_val)

            # Fire off all evaluations concurrently
            evaluation_tasks = [asyncio.create_task(bounded_evaluate(song)) for song in raw_candidates]

            # Yield as soon as any task completes
            for completed_task in asyncio.as_completed(evaluation_tasks):
                if total_approved >= 10:
                    break

                result = await completed_task

                if result.get("error"):
                    song_title = result.get("song", {}).get("title", "Unknown")
                    print(f"Error processing {song_title}: {result['error']}")
                    # If it's a critical missing file error, we inform the UI
                    if "Missing .pth" in result["error"]:
                        yield f"data: {{\"error\": \"{result['error']}\"}}\n\n"
                        break
                    continue

                if result.get("is_approved"):
                    buffer.append(result["song"])
                    total_approved += 1

                # Send immediately if buffer is not empty to improve UX responsiveness
                if len(buffer) >= 1:
                    yield f"data: {json.dumps(buffer)}\n\n"
                    buffer = []

            # Cleanup pending tasks if we stopped early (e.g., reached 10 songs)
            for t in evaluation_tasks:
                if not t.done():
                    t.cancel()

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
# Master's Thesis: AI Music Playlist Engine

This repository contains the source code for a Master's Thesis focused on the intelligent generation of musical playlists based on the user's mood (emotions).

The system uses Artificial Intelligence models (Convolutional Neural Networks and Transformers) to analyze audio spectrograms in real-time and filter songs retrieved from the public Jamendo API.

## System Architecture

The project is divided into two main components:

1. **Backend (FastAPI + PyTorch)**:
   - Exposes a Server-Sent Events (SSE) endpoint for real-time analysis.
   - Uses `librosa` for Mel-spectrogram extraction.
   - Includes two trained AI models: `CNN + Self-Attention` (SOTA) and classical `CRNN`.
   - Fully containerized with Docker.

2. **Frontend (React + Vite)**:
   - Single Page Application (SPA) to consume the backend's SSE stream.
   - Allows the user to select the desired emotion and the AI model to use.
   - Includes preview and playback of the generated audio tracks.

## Prerequisites

- **Docker** and **Docker Compose** installed on your system (to run the backend).
- **Node.js** (v18+) installed (to run the frontend locally).
- A Jamendo API Key (`JAMENDO_CLIENT_ID`).

## Installation and Execution

### 1. Environment Variables
Create a `.env` file in the root folder of the project (next to `docker-compose.yml`) and include your credentials:
```env
JAMENDO_CLIENT_ID=your_api_key_here
```

### 2. Starting the Backend (Docker)
Open a terminal in the root of the project and run:
```bash
docker-compose up --build
```
This will download the necessary Python dependencies, install system libraries (like `ffmpeg` and `libsndfile`), and start the FastAPI server at `http://localhost:8000`.

*Note: The first time it starts, it will load the AI model weights into RAM (Eager Loading) to optimize the performance of subsequent inferences.*

### 3. Starting the Frontend (React)
Open another terminal, navigate to the frontend folder, and start the development server:
```bash
cd app/frontend
npm install
npm run dev
```
The user interface will be available by default at `http://localhost:5173`.

## Artificial Intelligence Models

The models available in this repository are responsible for classifying audio into 12 possible emotions:
- `calm`, `dark`, `emotional`, `energetic`, `epic`, `happy`, `melancholic`, `powerful`, `relaxing`, `romantic`, `sad`, `upbeat`.

The trained weights (`.pth`) must reside in `app/backend/checkpoint/` for the backend to instantiate them correctly in memory.

import { useState } from 'react'
import './App.css'

const EMOTIONS = [
  "calm", "dark", "emotional", "energetic",
  "epic", "happy", "melancholic", "powerful",
  "relaxing", "romantic", "sad", "upbeat"
]

function App() {
  const [selectedEmotion, setSelectedEmotion] = useState('happy')
  const [selectedModel, setSelectedModel] = useState('cnn')
  const [isLoadingPlaylist, setIsLoadingPlaylist] = useState(false)
  const [generatedPlaylist, setGeneratedPlaylist] = useState(null)
  const [playlistError, setPlaylistError] = useState(null)

  const generatePlaylist = (e) => {
    if (e) e.preventDefault()
    setIsLoadingPlaylist(true)
    setPlaylistError(null)
    setGeneratedPlaylist([])

    // Auto-scroll downwards to show the loading scanner smoothly
    setTimeout(() => {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth"
      });
    }, 100);

    const eventSource = new EventSource(`http://localhost:8000/api/playlist/${selectedEmotion}?model=${selectedModel}`);

    eventSource.onmessage = (event) => {
      if (event.data === "[DONE]") {
        eventSource.close();
        setIsLoadingPlaylist(false);
        return;
      }

      try {
        const dataParsed = JSON.parse(event.data);
        if (dataParsed.error) {
          setPlaylistError(dataParsed.error);
          eventSource.close();
          setIsLoadingPlaylist(false);
          return;
        }
        setGeneratedPlaylist((prev) => [...prev, ...dataParsed]);
      } catch (err) {
        console.error("Stream error:", err);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      setPlaylistError("Connection error with AI API.");
      setIsLoadingPlaylist(false);
    };
  }

  return (
    <div className="container">
      <div className="header-section">
        <h1>AI Music Engine</h1>
        <p>Real-time neural network inference for emotion-based playlists</p>
      </div>

      <div className="card">
        <div className="model-selector-wrapper">
          <label>Neural Network Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="select-input"
          >
            <option value="cnn">CNN</option>
            <option value="crnn">CRNN</option>
            <option value="mobilenet">MobileNet</option>
          </select>
        </div>

        <div className="emotion-section-title">Target Emotion</div>
        <div className="emotion-grid">
          {EMOTIONS.map(emotion => (
            <div
              key={emotion}
              className={`emotion-chip ${selectedEmotion === emotion ? 'selected' : ''}`}
              onClick={() => !isLoadingPlaylist && setSelectedEmotion(emotion)}
            >
              {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
            </div>
          ))}
        </div>

        <button
          onClick={generatePlaylist}
          className="generate-btn"
          disabled={isLoadingPlaylist}
        >
          {isLoadingPlaylist ? 'Initializing Inference...' : 'Generate Playlist'}
        </button>
      </div>

      {playlistError && <div className="error-message">Error: {playlistError}</div>}

      {isLoadingPlaylist && (!generatedPlaylist || generatedPlaylist.length === 0) && (
        <div className="card loading-container">
          <div className="loader-pulse"></div>
          <div className="loading-text">Scanning audio spectrograms...</div>
        </div>
      )}

      {generatedPlaylist && generatedPlaylist.length > 0 && (
        <div className="card">
          <div className="results-header">
            <h3>Inference Results</h3>
            <span className="track-count">{generatedPlaylist.length} tracks approved</span>
          </div>

          <div className="playlist-grid">
            {generatedPlaylist.map((song, index) => (
              <div
                key={song.id}
                className="track-card"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="track-info">
                  {song.image && (
                    <img src={song.image} alt="Cover" className="track-image" />
                  )}
                  <div className="track-details">
                    <span className="track-title">{song.title}</span>
                    <span className="track-artist">{song.artist}</span>
                  </div>
                </div>
                <audio controls src={song.preview_url} className="track-audio">
                  Browser does not support audio.
                </audio>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
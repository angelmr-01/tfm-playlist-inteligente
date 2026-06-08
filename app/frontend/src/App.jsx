import { useState } from 'react'
import './App.css'

function App() {
  // ==========================================
  // Logic: Streaming Integration (Jamendo API & AI)
  // ==========================================
  const [selectedEmotion, setSelectedEmotion] = useState('happy')
  const [selectedModel, setSelectedModel] = useState('cnn_self_attention')
  const [isLoadingPlaylist, setIsLoadingPlaylist] = useState(false)
  const [generatedPlaylist, setGeneratedPlaylist] = useState(null)
  const [playlistError, setPlaylistError] = useState(null)

  const generatePlaylist = (e) => {
    e.preventDefault()
    setIsLoadingPlaylist(true)
    setPlaylistError(null)
    setGeneratedPlaylist([])

    // Establish SSE connection with the backend
    const eventSource = new EventSource(`http://localhost:8000/api/playlist/${selectedEmotion}?model=${selectedModel}`);

    // Process incoming chunks from the backend stream
    eventSource.onmessage = (event) => {
      // Close the connection if the server signals completion
      if (event.data === "[DONE]") {
        eventSource.close();
        setIsLoadingPlaylist(false);
        return;
      }

      try {
        // Read the incoming payload
        const dataParsed = JSON.parse(event.data);

        if (dataParsed.error) {
          setPlaylistError(dataParsed.error);
          eventSource.close();
          setIsLoadingPlaylist(false);
          return;
        }

        // Append new tracks to the existing state
        setGeneratedPlaylist((previousList) => [...previousList, ...dataParsed]);
      } catch (err) {
        console.error("Stream reading error:", err);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      setPlaylistError("Connection error with AI API.");
      setIsLoadingPlaylist(false);
    };
  }

  // ==========================================
  // UI Rendering
  // ==========================================
  return (
    <div className="container">
      <h1>AI Music Engine</h1>

      <div className="pantalla-activa">
        <h2>Emotion Inference Engine</h2>
        <p>The AI is evaluating spectrograms in real-time to find songs that match your mood.</p>

        <form onSubmit={generatePlaylist} className="formulario">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="selector-emocion"
            style={{ marginBottom: '10px' }}
          >
            <option value="cnn_self_attention">Model: CNN + Self Attention (SOTA)</option>
            <option value="crnn">Model: CRNN (Sequential)</option>
          </select>

          <select
            value={selectedEmotion}
            onChange={(e) => setSelectedEmotion(e.target.value)}
            className="selector-emocion"
          >
            <option value="calm">Calm</option>
            <option value="dark">Dark</option>
            <option value="emotional">Emotional</option>
            <option value="energetic">Energetic</option>
            <option value="epic">Epic</option>
            <option value="happy">Happy</option>
            <option value="melancholic">Melancholic</option>
            <option value="powerful">Powerful</option>
            <option value="relaxing">Relaxing</option>
            <option value="romantic">Romantic</option>
            <option value="sad">Sad</option>
            <option value="upbeat">Upbeat</option>
          </select>

          <button type="submit" disabled={isLoadingPlaylist}>
            {isLoadingPlaylist ? 'Analyzing...' : 'Generate Playlist'}
          </button>
        </form>

        {playlistError && <p className="error">Error: {playlistError}</p>}

        {generatedPlaylist && (
          <div className="resultados">
            <h3>Selected Candidates:</h3>
            <ul className="lista-playlist">
              {generatedPlaylist.map(song => (
                <li key={song.id} style={{ marginBottom: '1.5rem', borderBottom: '1px solid #ccc', paddingBottom: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
                    {song.image && <img src={song.image} alt="Portada" style={{ width: '50px', height: '50px', borderRadius: '5px' }} />}
                    <div>
                      <strong>{song.title}</strong> <br />
                      <small>{song.artist}</small>
                    </div>
                  </div>
                  <audio controls src={song.preview_url} style={{ width: '100%', height: '30px' }}>
                    Your browser does not support the audio element.
                  </audio>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
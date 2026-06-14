import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Ignoramos warnings de PySoundFile que a veces saltan con mp3 corruptos
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y CONSTANTES
# ==========================================
RUTA_TSV = 'data/tfm_catalogo_limpio.tsv'
RUTA_AUDIOS = './datos_tfm_audio_limpio'
RUTA_OUT_NPY = './espectrogramas_tfm'

os.makedirs(RUTA_OUT_NPY, exist_ok=True)

EMOCIONES = ['happy', 'sad', 'energetic', 'relaxing', 'dark', 'romantic', 
             'emotional', 'upbeat', 'epic', 'melancholic', 'calm', 'powerful']

SR = 22050
DURACION_OBJETIVO = 30.0
LONGITUD_OBJETIVO = int(SR * DURACION_OBJETIVO)
MIN_DURACION_ACEPTABLE = 20.0  # Si hay menos de 20s de música, se descarta
LONGITUD_MINIMA = int(SR * MIN_DURACION_ACEPTABLE)

# ==========================================
# 2. FUNCIÓN NÚCLEO (WORKER PARA MULTIPROCESAMIENTO)
# ==========================================
def procesar_cancion(row):
    """
    Procesa una única canción. Extrae sus 3 ventanas, calcula los Mel-Spectrograms
    y devuelve una lista de diccionarios con los metadatos de los chunks válidos.
    """
    track_str = row['TRACK_ID']
    id_limpio = str(int(track_str.replace('track_', '')))
    
    posible_mp3 = os.path.join(RUTA_AUDIOS, f"{id_limpio}.mp3")
    posible_flac = os.path.join(RUTA_AUDIOS, f"{id_limpio}.flac")
    ruta_audio = posible_mp3 if os.path.exists(posible_mp3) else posible_flac
    
    if not os.path.exists(ruta_audio):
        return [] # Retornamos vacío si no existe el archivo

    chunks_validos = []
    offsets = [30.0, 60.0, 90.0]
    
    for chunk_idx, offset in enumerate(offsets):
        try:
            # Cargamos solo la ventana de 30 segundos
            y, sr = librosa.load(ruta_audio, sr=SR, offset=offset, duration=DURACION_OBJETIVO)
            
            # REGLA DE NEGOCIO: ¿Hay suficiente música real?
            if len(y) < LONGITUD_MINIMA:
                continue # Descartamos este chunk por tener demasiados silencios
                
            # Padding: Si tiene entre 20 y 30 segundos, rellenamos con ceros hasta llegar a 30s
            if len(y) < LONGITUD_OBJETIVO:
                y = np.pad(y, (0, LONGITUD_OBJETIVO - len(y)), mode='constant')
                
            # DSP: Transformada de Fourier a Mel-Spectrogram (128 bandas)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Guardamos el tensor numpy (.npy)
            nombre_chunk = f"{id_limpio}_chunk_{chunk_idx}.npy"
            np.save(os.path.join(RUTA_OUT_NPY, nombre_chunk), mel_spec_db)
            
            # Generamos su registro de metadatos
            dict_chunk = {
                'TRACK_ID': track_str,
                'CHUNK_FILE': nombre_chunk
            }
            # Mapeamos las etiquetas (0 o 1) para las 12 emociones según el Ground Truth
            # El TSV original de Jamendo tiene formato "mood/theme---happy"
            for emo in EMOCIONES:
                etiqueta_jamendo = f"mood/theme---{emo}"
                # Comprobamos si la emoción exacta está en la columna TAGS
                if etiqueta_jamendo in str(row['TAGS']):
                    dict_chunk[emo] = 1.0
                else:
                    dict_chunk[emo] = 0.0
                    
            chunks_validos.append(dict_chunk)
            
        except Exception:
            # Fallos de decodificación o offset fuera de los límites del audio
            pass
            
    return chunks_validos

# ==========================================
# 3. ORQUESTADOR PRINCIPAL
# ==========================================
def main():
    print("Cargando TSV de metadatos...")
    df = pd.read_csv(RUTA_TSV, sep='\t')
    
    # Convertimos el DataFrame a una lista de diccionarios (filas) para el multiprocesamiento
    filas = [row for _, row in df.iterrows()]
    
    registro_total_chunks = []
    
    # Detectamos núcleos óptimos (dejamos 1 libre para no congelar tu PC)
    nucleos = max(1, os.cpu_count() - 1)
    print(f"Iniciando procesamiento en paralelo usando {nucleos} núcleos...")
    
    with ProcessPoolExecutor(max_workers=nucleos) as executor:
        # Mapeamos la función a todas las filas con una barra de progreso
        futuros = {executor.submit(procesar_cancion, fila): fila for fila in filas}
        
        for futuro in tqdm(as_completed(futuros), total=len(filas), desc="Procesando audios"):
            try:
                chunks_cancion = futuro.result()
                registro_total_chunks.extend(chunks_cancion)
            except Exception as e:
                print(f"Error procesando un archivo: {e}")

    # Guardamos el nuevo catálogo maestro para el entrenamiento
    df_resultado = pd.DataFrame(registro_total_chunks)
    ruta_csv_final = 'metadatos_espectrogramas.csv'
    df_resultado.to_csv(ruta_csv_final, index=False)
    
    print("-" * 40)
    print(f"¡Proceso completado con éxito!")
    print(f"Espectrogramas válidos generados: {len(df_resultado)}")
    print(f"Metadatos guardados en: {ruta_csv_final}")
    print("\nSiguiente paso: Comprime la carpeta 'espectrogramas_tfm' en un .zip y súbela a tu Google Drive.")

if __name__ == '__main__':
    main()
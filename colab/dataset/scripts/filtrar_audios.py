import pandas as pd
import os
import shutil

# ==========================================
# CONSTANTES
# ==========================================
RUTA_ORIGEN = './datos_tfm_audio'
RUTA_DESTINO = './datos_tfm_audio_limpio'
RUTA_TSV = 'data/tfm_catalogo_limpio.tsv'

def main():
    print("Cargando la lista de canciones válidas...")
    df = pd.read_csv(RUTA_TSV, sep='\t')
    
    os.makedirs(RUTA_DESTINO, exist_ok=True)
    
    canciones_movidas = 0
    canciones_perdidas = 0
    
    print("Extrayendo canciones usando la lógica de Bucketing (Direct Access)...")
    
    # Recorremos la lista de nuestras canciones válidas
    for track_string in df['TRACK_ID']:
        # Formato: "track_0002263"
        id_rellenado = track_string.replace('track_', '')
        carpeta = id_rellenado[-2:]
        id_limpio = str(int(id_rellenado))
        
        # Construimos las posibles rutas
        ruta_mp3 = os.path.join(RUTA_ORIGEN, carpeta, f"{id_limpio}.mp3")
        ruta_flac = os.path.join(RUTA_ORIGEN, carpeta, f"{id_limpio}.flac")
        
        if os.path.exists(ruta_mp3):
            shutil.move(ruta_mp3, os.path.join(RUTA_DESTINO, f"{id_limpio}.mp3"))
            canciones_movidas += 1
        elif os.path.exists(ruta_flac):
            shutil.move(ruta_flac, os.path.join(RUTA_DESTINO, f"{id_limpio}.flac"))
            canciones_movidas += 1
        else:
            canciones_perdidas += 1

    print("-" * 40)
    print(f"¡Misión Cumplida! Canciones rescatadas: {canciones_movidas}")
    if canciones_perdidas > 0:
        print(f"Hubo {canciones_perdidas} canciones que no estaban en las carpetas.")
    print("Ya puedes borrar la carpeta 'datos_tfm_audio' original de 151 GB.")

if __name__ == '__main__':
    main()
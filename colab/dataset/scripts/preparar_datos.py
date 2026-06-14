import pandas as pd
import os
import csv

# ==========================================
# CONSTANTES
# ==========================================
URL_CATALOGO = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/autotagging_moodtheme.tsv"
TOP_12 = {
    'mood/theme---happy', 'mood/theme---sad', 'mood/theme---energetic', 
    'mood/theme---relaxing', 'mood/theme---dark', 'mood/theme---romantic', 
    'mood/theme---emotional', 'mood/theme---upbeat', 'mood/theme---epic', 
    'mood/theme---melancholic', 'mood/theme---calm', 'mood/theme---powerful'
}

def limpiar_etiquetas(etiquetas_str: str) -> str:
    """
    Filtra las etiquetas de Jamendo quedándose solo con las que pertenecen al TOP_12.
    """
    etiquetas = set(str(etiquetas_str).split(','))
    etiquetas_utiles = etiquetas.intersection(TOP_12)
    
    if etiquetas_utiles:
        return ','.join(etiquetas_utiles)
    return None

def main():
    print("Descargando el catálogo maestro de Jamendo (puede tardar unos segundos)...")
    df = pd.read_csv(URL_CATALOGO, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    total_original = len(df)
    
    print("Limpiando ruido semántico y descartando canciones no deseadas...")
    df['TAGS'] = df['TAGS'].apply(limpiar_etiquetas)
    
    df_limpio = df.dropna(subset=['TAGS']).copy()
    total_limpio = len(df_limpio)
    
    os.makedirs('data', exist_ok=True)
    ruta_salida = 'data/tfm_catalogo_limpio.tsv'
    df_limpio.to_csv(ruta_salida, sep='\t', index=False)
    
    print("-" * 40)
    print(f"Canciones Originales (Basura incluida): {total_original}")
    print(f"Canciones Válidas (Para tu TFM):        {total_limpio}")
    print(f"Canciones Descartadas (Ahorro de disco): {total_original - total_limpio}")
    print(f"Nuevo catálogo guardado en: {ruta_salida}")

if __name__ == '__main__':
    main()